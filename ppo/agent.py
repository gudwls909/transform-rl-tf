import os
import tensorflow as tf
import numpy as np
import copy

from environment import MnistEnvironment, Environment
from replay_memory import ReplayMemory
from ddpg_model import PPO
from origin_model.mnist_solver import Network

class Agent(object):
    def __init__(self, args, sess):
        # CartPole 환경
        self.sess = sess
        self.model = Network(sess, phase='train')  # mnist accurcacy model
        self.env = MnistEnvironment(self.model) 
        self.state_size = self.env.state_size
        self.action_size = self.env.action_size
        self.a_bound = self.env.a_bound
        self.train_size = len(self.env.train_images)
        self.test_size = len(self.env.test_images)
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.discount_factor = args.discount_factor
        self.epsilon = args.epsilon
        self.epochs = args.epochs

        self.num_actor = 32  # N
        self.timesteps = 20  # T
        self.gae_parameter = 0.95  # lambda
        self.num_train = 8  # K

        self.ENV = Environment(self.env, self.state_size, self.action_size)
        self.replay = ReplayMemory(self.state_size, self.batch_size, self.num_actor * self.timesteps)
        self.ppo = PPO(self.state_size, self.action_size, self.sess, self.learning_rate,
                       self.discount_factor, self.replay, self.epsilon, self.a_bound)

        self.save_dir = args.save_dir
        self.render_dir = args.render_dir
        self.play_dir = args.play_dir

        # initialize
        sess.run(tf.global_variables_initializer())  # tensorflow graph가 다 만들어지고 난 후에 해야됨

        # load pre-trained mnist model
        self.env.model.checkpoint_load()
        
        self.saver = tf.train.Saver()
        pass

    def select_action(self, state):
        policy = self.sess.run(self.ppo.sampled_action, feed_dict={self.ppo.state: state})[0][0]
        policy_clip = np.clip(policy, -self.a_bound, self.a_bound)
        return policy_clip
        pass

    def make_delta(self, memory):
        states, rewards, next_states = [], [], []
        for i in range(len(memory)):
            states.append(memory[i][0])
            rewards.append(memory[i][2])
            next_states.append(memory[i][3])
        current_v = self.sess.run(self.ppo.critic, feed_dict={self.ppo.state: states})
        next_v = self.sess.run(self.ppo.critic, feed_dict={self.ppo.state: next_states})
        delta = [r_t + self.discount_factor * v_next - v for r_t, v_next, v in zip(rewards, next_v, current_v)]
        return delta
        pass

    def make_gae(self, memory):
        delta = self.make_delta(memory)
        gae = copy.deepcopy(delta)
        for t in reversed(range(len(gae) - 1)):
            gae[t] = gae[t] + self.gae_parameter * self.discount_factor * gae[t + 1]
            # memory[t].append(gae[t])
        # memory[len(gae)-1].append(gae[len(gae)-1])

        gae = np.array(gae).astype(np.float32)
        gae = (gae - gae.mean()) / (gae.std() + 1e-10)
        for t in range(len(gae)):
            memory[t].append(gae[t])
        pass

    def memory_to_replay(self, memory):
        self.make_gae(memory)
        for i in range(len(memory)):
            self.replay.add(memory[i][0], memory[i][1], memory[i][2], memory[i][3], memory[i][4], memory[i][5])
        pass

    def train(self):
        scores, losses, scores2, losses2, idx_list = [], [], [], [], []
        self.ppo.update_target_network()
        for e in range(self.epochs):
            for i, idx in enumerate(np.random.permutation(self.train_size)):
                idx_list.append(idx)
                if (i+1) % self.num_actor == 0:
                    for j in range(self.num_actor):
                        memory, states, rewards, next_states = [], [], [], []
                        terminal = False
                        score = 0
                        state = self.ENV.new_episode(idx_list[j])
                        for _ in range(self.timesteps):
                            state = np.reshape(state, [1, self.state_size])
                            action = self.select_action(state)
                            next_state, reward, terminal = self.ENV.act(action)
                            state = state[0]
                            memory.append([state, action, reward, next_state, terminal])
                            score += reward
                            state = next_state

                            if terminal:
                                break

                        scores.append(score)
                        self.memory_to_replay(memory)

                    for _ in range(self.num_train):
                        losses.append(self.ppo.train_network())

                    self.ppo.update_target_network()
                    self.replay.clear()
                    scores2.append(np.mean(scores))
                    losses2.append(np.mean(losses))

                    losses.clear()
                    scores.clear()
                    idx_list.clear()

                if (i+1)%50 == 0 and (i+1) >= self.num_actor:
                    print('epoch', e+1, 'iter:', f'{i+1:05d}', ' score:', f'{scores2[-1]:.03f}',
                          ' last 10 mean score', f'{np.mean(scores2[-min(10, len(scores2)):]):.03f}',
                          ' loss', f'{losses2[-1]:.03f}', f'sequence: {self.env.sequence}')
                if (i+1)%200 == 0:
                    self.ENV.render_worker(os.path.join(self.render_dir, f'{(i+1):05d}.png'))
                if (i+1)%1000 == 0:
                    self.save()
        pass

    def play(self):
        cor_before_lst, cor_after_lst = [], []
        for idx in range(self.test_size): 
            state = self.ENV.new_episode(idx, phase='test')
            state = np.reshape(state, [1, self.state_size])
    
            terminal = False
            score = 0
            while not terminal:
                action = self.select_action(state)
                next_state, reward, terminal = self.ENV.act(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                score += reward
                state = next_state
#                 time.sleep(0.02)
                if terminal:
                    (cor_before, cor_after) = self.ENV.compare_accuracy()
                    cor_before_lst.append(cor_before)
                    cor_after_lst.append(cor_after)

                    self.ENV.render_worker(os.path.join(self.play_dir, f'{(idx+1):04d}.png'))
                    print(f'{(idx+1):04d} image score: {score}\n')
        print('====== NUMBER OF CORRECTION =======')
        print(f'before: {np.sum(cor_before_lst)}, after: {np.sum(cor_after_lst)}')
    pass

    def save(self):
        checkpoint_dir = os.path.join(self.save_dir, 'ckpt')
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'trained_agent'))

    def load(self):
        checkpoint_dir = os.path.join(self.save_dir, 'ckpt')
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'trained_agent'))

