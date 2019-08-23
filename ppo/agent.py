import os
import tensorflow as tf
import numpy as np
import copy
from math import pi

from environment import MnistEnvironment, Environment
from ppo.replay_memory import ReplayMemory
from ppo.ppo_model import PPO
from origin_model.mnist_solver import Network

import time
from scipy.stats import norm


class Agent(object):
    def __init__(self, args, sess):
        self.sess = sess
        self.model = Network(sess, phase='test')  # pre-trained mnist accuracy model
        self.env = MnistEnvironment(self.model, args.env, args.reward_type)
        self.state_size = self.env.state_size
        self.state_shape = self.env.state_shape
        self.action_size = self.env.action_size
        self.a_bound = self.env.a_bound
        self.train_size = len(self.env.train_images)
        self.test_size = len(self.env.test_images)
        self.learning_rate = args.learning_rate
        self.batch_size = args.batch_size
        self.discount_factor = args.discount_factor
        self.epsilon = args.epsilon
        self.epochs = args.epochs
        self._make_std()

        self.num_actor = 256  # N
        self.timesteps = 20  # T
        self.gae_parameter = 0.99  # lambda
        self.num_train = 64  # K

        self.ENV = Environment(self.env, self.state_size, self.action_size)
        self.replay = ReplayMemory(self.state_size, self.batch_size, self.num_actor * self.timesteps)
        self.ppo = PPO(self.state_size, self.action_size, self.sess, self.learning_rate,
                       self.discount_factor, self.replay, self.epsilon, self.a_bound, self.state_shape)

        self.continue_train = args.continue_train
        self.save_dir = args.save_dir
        self.render_dir = args.render_dir
        self.play_dir = args.play_dir

        # initialize
        sess.run(tf.global_variables_initializer())  # tensorflow graph가 다 만들어지고 난 후에 해야됨

        # load pre-trained mnist model
        self.env.model.checkpoint_load()

        self.saver = tf.train.Saver()

        # continue_train
        if self.continue_train:
            self.load()
        pass

    def _policy_action_bound(self, policy):
        a_range = (self.a_bound[:, 1] - self.a_bound[:, 0]) / 2.
        a_mean = (self.a_bound[:, 0] + self.a_bound[:, 1]) / 2.

        return policy * np.transpose(a_range) + np.transpose(a_mean)

    def select_action(self, state, phase):
        if phase == 'step':
            policy = self.sess.run(self.ppo.sampled_action,
                                   feed_dict={self.ppo.state: state, self.ppo.std: self.std_step})[0]
        elif phase == 'test':
            policy = self.sess.run(self.ppo.sampled_action,
                                   feed_dict={self.ppo.state: state, self.ppo.std: self.std_test})[0]
        else:
            raise PhaseError('Phase is not train or test')
        policy = self._policy_action_bound(policy)
        return policy
        pass

    def _get_old_policy(self, state, action):
        a_range = (self.a_bound[:, 1] - self.a_bound[:, 0]) / 2.
        a_mean = (self.a_bound[:, 0] + self.a_bound[:, 1]) / 2.

        action = (action - a_mean) / a_range
        actor_output = self.sess.run(self.ppo.actor,
                                     feed_dict={self.ppo.state: state, self.ppo.std: self.std_step})[0]
        # old_policy = self.sess.run(self.ppo.normal.log_prob(action - actor_output),
        #                           feed_dict={self.ppo.state: state, self.ppo.std: self.std_step})[0]
        old_policy = norm.logpdf(action - actor_output, loc=0, scale=self.std_step[0])
        return old_policy
        pass

    def _make_std(self):
        # make std for step, train and test
        # a_range = self.a_bound[:, 1:] - self.a_bound[:, :1]
        self.std_step = np.ones([1, self.action_size])
        self.std_train = np.ones([self.batch_size, self.action_size])
        # self.std_train = np.multiply(self.std_train, np.transpose(a_range)) / 2.
        self.std_test = self.std_train / 5.

    '''
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

        # normalize gae
        gae = np.array(gae).astype(np.float32)
        gae = (gae - gae.mean()) / (gae.std() + 1e-10)
        for t in range(len(gae)):
            memory[t].append(gae[t])
        pass
    '''

    def make_gae(self, memory):
        rewards = [m[2] for m in memory]
        masks = [m[4] for m in memory]  # terminals
        values = [m[6] for m in memory]
        returns = np.zeros_like(rewards)
        advants = np.zeros_like(rewards)

        running_returns = 0
        previous_value = 0
        running_advants = 0

        for t in reversed(range(0, len(rewards))):
            running_returns = rewards[t] + self.discount_factor * running_returns * masks[t]
            running_tderror = rewards[t] + self.discount_factor * previous_value * masks[t] - values[t]
            running_advants = running_tderror + self.discount_factor * self.gae_parameter * running_advants * masks[t]

            returns[t] = running_returns
            previous_value = values[t]
            advants[t] = running_advants

        if len(rewards) > 1:
            if (advants.std() == [0 for _ in range(len(rewards))]).all():
                pass
            else:
                advants = (advants - advants.mean()) / advants.std()

        for t in range(len(rewards)):
            memory[t].append(advants[t])
            memory[t].append(returns[t])
        pass

    def memory_to_replay(self, memory):
        self.make_gae(memory)
        for i in range(len(memory)):
            self.replay.add(memory[i])
        pass

    def train(self):
        scores, losses, scores2, losses2, idx_list = [], [], [], [], []
        count = 0
        for e in range(self.epochs):
            for i, idx in enumerate(np.random.permutation(self.train_size)):
                count += 1
                idx_list.append(idx)
                if count % self.num_actor == 0:
                    for j in range(self.num_actor):
                        memory, states, rewards, next_states = [], [], [], []
                        score = 0
                        state = self.ENV.new_episode(idx_list[j])
                        for _ in range(self.timesteps):
                            state = np.reshape(state, [1, self.state_size])
                            action = self.select_action(state, 'step')
                            next_state, reward, terminal = self.ENV.act(action)
                            old_policy = self._get_old_policy(state, action)
                            old_value = self.sess.run(self.ppo.critic,
                                                      feed_dict={self.ppo.state: state})[0]
                            state = state[0]
                            memory.append([state, action, reward, next_state, terminal, old_policy, old_value])
                            score += reward
                            state = next_state

                            if terminal:
                                break

                        scores.append(score)
                        self.memory_to_replay(memory)

                    for _ in range(self.num_train):
                        losses.append(self.ppo.train_network(self.std_train))

                    self.replay.clear()
                    scores2.append(np.mean(scores))
                    losses2.append(np.mean(losses, axis=0))

                    losses.clear()
                    scores.clear()
                    idx_list.clear()

                if count % 300 == 0 and count >= self.num_actor:
                    print('epoch', e + 1, 'iter:', f'{count:05d}', ' score:', f'{scores2[-1]:.03f}',
                          ' actor loss', f'{losses2[-1][0]:.03f}', ' critic loss', f'{losses2[-1][1]:.03f}',
                          f'sequence: {self.env.sequence}')
                if count % 300 == 0 and count >= self.num_actor:
                    self.ENV.render_worker(os.path.join(self.render_dir, f'{count:05d}.png'))
                if count % 1000 == 0:
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
                action = self.select_action(state, 'test')
                next_state, reward, terminal = self.ENV.act(action)
                next_state = np.reshape(next_state, [1, self.state_size])
                score += reward
                state = next_state
                #                 time.sleep(0.02)
                if terminal:
                    (cor_before, cor_after) = self.ENV.compare_accuracy()
                    cor_before_lst.append(cor_before)
                    cor_after_lst.append(cor_after)

                    if (idx + 1) % 200 == 0:
                        self.ENV.render_worker(os.path.join(self.play_dir, f'{(idx + 1):04d}.png'))
                        print(f'{(idx + 1):04d} image score: {score}')
        print('====== NUMBER OF CORRECTION =======')
        print(f'before: {np.sum(cor_before_lst)}, after: {np.sum(cor_after_lst)}')

    pass

    def save(self):
        checkpoint_dir = os.path.join(self.save_dir, 'ckpt')
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        self.saver.save(self.sess, os.path.join(checkpoint_dir, 'trained_agent'))

    def load(self):
        print('=== loading ckeckpoint... ===')
        checkpoint_dir = os.path.join(self.save_dir, 'ckpt')
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'trained_agent'))
