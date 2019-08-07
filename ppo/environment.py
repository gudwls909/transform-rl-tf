import pickle
import numpy as np
from os.path import join
# from tensorflow.examples.tutorials.mnist import input_data

import util


class MnistEnvironment(object):
    def __init__(self, model):
        self.model = model
        self.mc = 20
        self.threshold = 3e-3
        self._max_episode_steps = 5

        self.state_shape = [40, 40, 1]
        self.state_size = 1600
        self.action_size = 6
        self.a_bound = np.array([])# [r, sh1, sh2, sc1, sc2, t1, t2]
        
        self.data_load()
    
    def data_load(self):
        with open(join('..','data','affMNIST.pickle'),'rb') as f:
            train_dataset, test_dataset = pickle.load(f)
        
        # images.shape = (10000,40,40,1), labels onehot=False
        self.train_images, self.train_labels = train_dataset
        self.test_images, self.test_labels = test_dataset
        self.test_images, self.test_labels = self.test_images[:200], self.test_labels[:200]
            
    def reset(self, idx, phase='train'):
        self.phase = phase
        if self.phase == 'train':
            self.img = self.train_images[idx] # 40*40*1
            self.label = self.train_labels[idx]
        else: # self.phase == 'test'
            self.img = self.test_images[idx]
            self.label = self.test_labels[idx]

        # initialize
        self.sequence = 0
        self.batch_imgs = [self.img] # save the transformed images 
        self.del_thetas = [np.array((1.,0.,0.,0.,1.,0.))] # save theta sequentially
        img_28size = util.theta2affine_img(self.img, self.del_thetas[-1], resize=(28,28))
        prob_set = util.all_prob(self.model, np.expand_dims(img_28size, axis=0), self.mc)
        self.uncs = [util.get_mutual_informations(prob_set)[0]] # save the uncertainty
        self.label_hats = [prob_set.mean(axis=0).argmax(axis=1)[0]] # save predicted label
        self.rewards = [0]

        return self.img.flatten()
    
    def step(self, theta):
        # sequence
        self.sequence += 1
###### function
        self.del_thetas.append(theta)

        # next_state
        del_theta = util.integrate_thetas(self.del_thetas)
        next_img = util.theta2affine_img(self.img, del_theta)
        
        # calculate uncertainty
        img_28size = util.theta2affine_img(self.img, del_theta, (28,28))
        prob_set = util.all_prob(self.model, np.expand_dims(img_28size, axis=0), self.mc)
        unc_after = util.get_mutual_informations(prob_set)[0]
        unc_before = self.uncs[-1]

        # save the values
        self.uncs.append(unc_after)
        self.label_hats.append(prob_set.mean(axis=0).argmax(axis=1)[0])
        self.batch_imgs.append(next_img)
        rew_prob = prob_set.mean(axis=0)[0][self.label]
        #rew_prob = np.clip(-np.log(1-rew_prob), a_min=None, a_max=-np.log(self.threshold))
        
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
        if np.sum(next_img) < 20:
            reward = -5
            terminal = True
        else:
            reward_after = np.clip(-np.log(unc_after), 
                                   a_min=None, a_max=-np.log(self.threshold))
            reward_before = np.clip(-np.log(unc_before), 
                                    a_min=None, a_max=-np.log(self.threshold))
            reward = reward_after - reward_before - 1.0

        self.rewards.append(reward)

        return next_img.flatten(), reward, terminal, 0
        
    def render(self, fname):
        self.batch_imgs = np.stack(self.batch_imgs)
        img_width = self.batch_imgs.shape[2]
        
        self.batch_imgs = util.make_grid(self.batch_imgs, len(self.batch_imgs), 2)
        print(self.uncs,'\n')
        tick_labels =  [str([float(f'{v:.01f}') for v in theta[:3]]) + '\n' + 
                        str([float(f'{v:.01f}') for v in theta[3:]]) +  f'\n{unc:.04f}\n{label_hat}\n{reward:.04f}'
                       for (theta, unc, label_hat, reward)
                       in zip(self.del_thetas, self.uncs, self.label_hats, self.rewards)]
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
        next_state, reward, terminal, _ = self.env.step(action)
        return next_state, reward, terminal
        pass

    def render_worker(self, fname):
        self.env.render(fname)
        pass

    def compare_accuracy(self):
        return self.env.compare_accuracy()
        pass

