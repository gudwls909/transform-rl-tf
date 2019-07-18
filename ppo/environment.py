import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import util


class MnistEnvironment(object):
    def __init__(self, model):
        self.model = model
        self.mc = 10
        self.threshold = 3e-3
        self._max_episode_steps = 10
        
        self.state_size = 784
        self.action_size = 1
        self.a_bound = 30
        
        self.data_load()
    
    def data_load(self):
        mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)
        
        self.train_images = mnist.train.images.reshape([-1,28,28,1])
        self.train_labels = mnist.train.labels
        self.test_images = mnist.test.images.reshape([-1,28,28,1])[:200]
        self.test_labels = mnist.test.labels[:200]
            
    def reset(self, idx, phase='train'):
        self.phase = phase
        if self.phase == 'train':
            self.img = self.train_images[idx] # 28*28*1
            self.img = util.random_degrade(self.img)
            self.label = self.train_labels[idx]
        else: # self.phase == 'test'
            self.img = self.test_images[idx]
            self.img = util.random_degrade(self.img)
            self.label = self.test_labels[idx]

        # initialize
        self.sequence = 0
        self.batch_imgs = [self.img] # save the rotated images 
        self.del_angles = [0] # save the rotated angle sequentially
        prob_set = util.all_prob(self.model, np.expand_dims(self.img, axis=0), self.mc)
        self.uncs = [util.get_mutual_informations(prob_set)[0]] # save the uncertainty
        self.label_hats = [prob_set.mean(axis=0).argmax(axis=1)[0]] # save predicted label

        return self.img.flatten()
    
    def step(self, rotate_angle):
        # sequence
        self.sequence += 1
        self.del_angles.append(rotate_angle)

        # next_state
        next_img = util.np_rotate(self.img, sum(self.del_angles))
        
        # calculate uncertainty
        prob_set = util.all_prob(self.model, np.expand_dims(next_img, axis=0), self.mc)
        unc_after = util.get_mutual_informations(prob_set)[0]
        unc_before = self.uncs[-1]

        # save the values
        self.uncs.append(unc_after)
        self.label_hats.append(prob_set.mean(axis=0).argmax(axis=1)[0])
        self.batch_imgs.append(next_img)
        
        # terminal
        if self.phase == 'train':
            if (unc_after < self.threshold and self.label_hats[-1] == self.label) \
               or self.sequence >= self._max_episode_steps:
                terminal = True
            else:
                terminal = False
        else: # self.phase == 'test'
            if unc_after < self.threshold or self.sequence >= self._max_episode_steps:
                terminal = True
            else:
                terminal = False

        # reward
        reward_after = np.clip(-np.log(unc_after), a_min=None, a_max=-np.log(self.threshold))
        reward_before = np.clip(-np.log(unc_before), a_min=None, a_max=-np.log(self.threshold))
        if terminal:
            reward = reward_after - reward_before - 1.0
#             reward = 0
        else:
            reward = reward_after - reward_before - 1.0
#             reward = -0.2
        
        return next_img.flatten(), reward, terminal, 0
        
    def render(self, fname):
        self.batch_imgs = np.stack(self.batch_imgs)
        img_width = self.batch_imgs.shape[2]
        
        self.batch_imgs = util.make_grid(self.batch_imgs, len(self.batch_imgs), 2)
        print(self.uncs,'\n')
        tick_labels = [f'{angle:.02f}\n{unc:.04f}\n{label_hat}'
                       for (angle, unc, label_hat) 
                       in zip(self.del_angles, self.uncs, self.label_hats)]
        util.save_batch_fig(fname, self.batch_imgs, img_width, tick_labels)

    def compare_accuracy(self):
        return (self.label_hats[0] == self.label, self.label_hats[-1] == self.label)


class Environment(object):
    def __init__(self, env, state_size, action_size):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        pass

    def new_episode(self, idx, phase='train'):
        state = self.env.reset(idx, phase)
        return state
        pass

    def act(self, action):
        next_state, reward, terminal, _ = self.env.step(*action)
        return next_state, reward, terminal
        pass

    def render_worker(self, fname):
        self.env.render(fname)
        pass

    def compare_accuracy(self):
        return self.env.compare_accuracy()
        pass

