import pickle
import numpy as np
import os.path
from os.path import join
# from tensorflow.examples.tutorials.mnist import input_data

import util
import affMNIST_generator


class MnistEnvironment(object):
    def __init__(self, model, env_type, rew_type):
        self.model = model
        if env_type in ['r', 'rsc', 'rsh', 'rss', 'rsst']:
            self.type = env_type
        else:
            print(env_type)
            raise TypeError('env type error')
        if rew_type in [1, 2, 3]:
            self.rew_type = rew_type
        else:
            print(rew_type)
            raise TypeError('rew type error')
        self.mc = 20
        # self.threshold = 3e-3 if self.type == 'r' else 8e-3
        self.threshold = 0.99 if self.type == 'r' else 0.99
        self._max_episode_steps = 10
        if env_type == 'rsst':
            self.state_shape = [40, 40, 1]
            self.state_size = 1600
        else:
            self.state_shape = [28, 28, 1]
            self.state_size = 784

        if self.type == 'r':
            self.action_size = 1
            self.a_bound = np.array([[-30., 30.]])
        elif self.type == 'rsc':
            self.action_size = 3
            self.a_bound = np.array([[-30., 30.],
                                     [0.9, 1.1],
                                     [0.9, 1.1]])
        elif self.type == 'rsh':
            self.action_size = 3
            self.a_bound = np.array([[-30., 30.],
                                     [-0.3, 0.3],
                                     [-0.3, 0.3]])
        elif self.type == 'rss':
            self.action_size = 5
            self.a_bound = np.array([[-30., 30.],
                                     [-0.3, 0.3],
                                     [-0.3, 0.3],
                                     [0.9, 1.1],
                                     [0.9, 1.1]])
        else:  # self.type = 'rsst'
            self.action_size = 7
            self.a_bound = np.array([[-30., 30.],
                                     [-0.2, 0.2],
                                     [-0.2, 0.2],
                                     [0.9, 1.1],
                                     [0.9, 1.1],
                                     [-4., 4],
                                     [-4., 4]])

        self.data_load()

    def data_load(self):
        if not os.path.isfile('data/affMNIST_28' + self.type + '.pickle'):
            print("=== No Train Data File Exist, Let's Generate it first ===")
            affMNIST_generator.main(self.type)
        with open(join('data', 'affMNIST_28' + self.type + '.pickle'), 'rb') as f:
            train_dataset, test_dataset = pickle.load(f)

        # images.shape = (10000,28,28,1) or (10000,40,40,1), labels onehot=False
        self.train_images, self.train_labels = train_dataset
        self.test_images, self.test_labels = test_dataset
        self.test_images, self.test_labels = self.test_images[:10000], self.test_labels[:10000]

    def reset(self, idx, phase='train'):
        self.phase = phase
        if self.phase == 'train':
            self.img = self.train_images[idx]  # 28*28*1
            self.label = self.train_labels[idx]
        else:  # self.phase == 'test'
            self.img = self.test_images[idx]
            self.label = self.test_labels[idx]

        # initialize
        self.sequence = 0
        self.batch_imgs = [self.img]  # save the transformed images
        self.del_thetas = [np.array((1., 0., 0., 0., 1., 0.))]  # save theta sequentially
        if self.type == 'r':
            self.del_params = [[0.]]
        elif self.type == 'rsc':
            self.del_params = [[0., 1., 1.]]
        elif self.type == 'rsh':
            self.del_params = [[0., 0., 0.]]
        elif self.type == 'rss':
            self.del_params = [[0., 0., 0., 1., 1.]]
        else:  # self.type == 'rsst'
            self.del_params = [[0., 0., 0., 1., 1., 0., 0.]]

        img_28size = util.theta2affine_img(self.img, self.del_thetas[-1], (28, 28))
        # prob_set = util.all_prob(self.model, np.expand_dims(img_28size, axis=0), self.mc)
        # self.uncs = [util.get_mutual_informations(prob_set)[0]]  # save the uncertainty
        prob = np.clip(self.model.test(np.expand_dims(img_28size, axis=0))[0], 0, 0.9999)
        self.accs = [prob[self.label]]
        # self.label_hats = [prob_set.mean(axis=0).argmax(axis=1)[0]]  # save predicted label
        self.label_hats = [prob.argmax()]
        self.max_probs = [prob.max()]
        self.rewards = [0]

        return self.img.flatten()

    def step(self, param):
        # sequence
        self.sequence += 1
        theta = util.param2theta(param, self.type)
        self.del_thetas.append(theta)
        self.del_params.append(param)

        # next_state
        del_theta = util.integrate_thetas(self.del_thetas)
        next_img = util.theta2affine_img(self.img, del_theta)

        # calculate uncertainty
        img_28size = util.theta2affine_img(self.img, del_theta, (28, 28))
        # prob_set = util.all_prob(self.model, np.expand_dims(img_28size, axis=0), self.mc)
        # unc_after = util.get_mutual_informations(prob_set)[0]
        # unc_before = self.uncs[-1]
        acc_before = self.accs[-1]
        prob = self.model.test(np.expand_dims(img_28size, axis=0))[0]
        acc_after = np.clip(prob[self.label], 0, 0.9999)
        rew_prob = -np.log(1-acc_after)
        rew_prob_before = -np.log(1-acc_before)

        # save the values
        # self.uncs.append(unc_after)
        self.accs.append(acc_after)
        # self.label_hats.append(prob_set.mean(axis=0).argmax(axis=1)[0])
        self.label_hats.append(prob.argmax())
        self.max_probs.append(prob.max())
        self.batch_imgs.append(next_img)

        # terminal
        self.success = False
        if self.phase == 'train':
            # if unc_after < self.threshold and self.label_hats[-1] == self.label:
            if acc_after > self.threshold and self.label_hats[-1] == self.label:
                terminal = True
                self.success = True
            elif self.sequence >= self._max_episode_steps:
                terminal = True
            else:
                terminal = False
        else:  # self.phase == 'test'
            # if unc_after < self.threshold or self.sequence >= self._max_episode_steps:
            if self.max_probs[-1] > 0.995 or self.sequence >= self._max_episode_steps:
                terminal = True
            else:
                terminal = False

        # reward
        if np.sum(next_img) < 20:
            reward = -5
            terminal = True
        else:
            # reward = self._make_reward()
            reward = rew_prob - rew_prob_before - 1

        reward += 1. if self.success else 0
        self.rewards.append(reward)

        return next_img.flatten(), reward, terminal, 0

    def render(self, fname):
        self.batch_imgs = np.stack(self.batch_imgs)
        img_width = self.batch_imgs.shape[2]

        self.batch_imgs = util.make_grid(self.batch_imgs, len(self.batch_imgs), 2)
        # print(self.uncs, '\n')
        # print(self.accs, '\n')
        if self.action_size == 1:
            tick_labels = [str([float(f'{p:.01f}') for p in param]) + f'\n{unc:.04f}\n{label_hat}\n{reward:.04f}'
                           for (param, unc, label_hat, reward)
                           # in zip(self.del_params, self.uncs, self.label_hats, self.rewards)]
                           in zip(self.del_params, self.accs, self.label_hats, self.rewards)]
        elif self.action_size == 3:
            tick_labels = [str([float(f'{p:.01f}') for p in param[:1]]) + '\n' +
                           str([float(f'{p:.03f}') for p in param[1:3]]) + f'\n{unc:.04f}\n{label_hat}\n{reward:.04f}'
                           for (param, unc, label_hat, reward)
                           # in zip(self.del_params, self.uncs, self.label_hats, self.rewards)]
                           in zip(self.del_params, self.accs, self.label_hats, self.rewards)]
        elif self.action_size == 5:
            tick_labels = [str([float(f'{p:.01f}') for p in param[:1]]) + '\n' +
                           str([float(f'{p:.03f}') for p in param[1:3]]) + '\n' +
                           str([float(f'{p:.03f}') for p in param[3:5]]) + f'\n{unc:.04f}\n{label_hat}\n{reward:.04f}'
                           for (param, unc, label_hat, reward)
                           # in zip(self.del_params, self.uncs, self.label_hats, self.rewards)]
                           in zip(self.del_params, self.accs, self.label_hats, self.rewards)]
        else:  # self.action_size == 7
            tick_labels = [str([float(f'{p:.01f}') for p in param[:1]]) + '\n' +
                           str([float(f'{p:.03f}') for p in param[1:3]]) + '\n' +
                           str([float(f'{p:.03f}') for p in param[3:5]]) + '\n' +
                           str([float(f'{p:.01f}') for p in param[5:7]]) + f'\n{unc:.04f}\n{label_hat}\n{reward:.04f}'
                           for (param, unc, label_hat, reward)
                           # in zip(self.del_params, self.uncs, self.label_hats, self.rewards)]
                           in zip(self.del_params, self.accs, self.label_hats, self.rewards)]
        util.save_batch_fig(fname, self.batch_imgs, img_width, tick_labels)

    def compare_accuracy(self):
        # label_hats = self.label_hats[-1] if self.success else self.label_hats[np.argmax(self.max_probs)]
        return self.label_hats[0] == self.label, self.label_hats[-1] == self.label
    '''
    def _make_reward(self):
        if self.rew_type == 1:
            # tuned reward function
            unc_before = self.uncs[-2]
            unc_after = self.uncs[-1]
            pred_before = self.label_hats[-2] == self.label
            pred_after = self.label_hats[-1] == self.label
            pred = (pred_before, pred_after)
            rew_before = np.clip(-np.log(unc_before), a_min=None, a_max=-np.log(self.threshold))
            rew_after = np.clip(-np.log(unc_after), a_min=None, a_max=-np.log(self.threshold))

            reward = rew_after - rew_before

            if np.abs(unc_after - unc_before) < 0.001:
                if pred == (0, 0):
                    reward = reward
                elif pred == (0, 1):
                    reward = reward + 1
                elif pred == (1, 0):
                    reward = reward - 1
                elif pred == (1, 1):
                    reward = reward
            else:
                if unc_before < unc_after:
                    if pred == (0, 0):
                        reward = -reward
                    elif pred == (0, 1):
                        reward = -reward + 1
                    elif pred == (1, 0):
                        reward = reward - 1
                    elif pred == (1, 1):
                        reward = reward
                else:
                    if pred == (0, 0):
                        reward = -reward - 1
                    elif pred == (0, 1):
                        reward = reward + 2
                    elif pred == (1, 0):
                        reward = -reward - 2
                    elif pred == (1, 1):
                        reward = reward

        elif self.rew_type == 2:
            # not tuned reward function
            unc_before = self.uncs[-2]
            unc_after = self.uncs[-1]
            rew_before = np.clip(-np.log(unc_before), a_min=None, a_max=-np.log(self.threshold))
            rew_after = np.clip(-np.log(unc_after), a_min=None, a_max=-np.log(self.threshold))

            reward = rew_after - rew_before

        else:  # self.re_type == 3
            # reward only if task succeeds
            reward = 0
            pass

        # reward gets -1 every timestep
        reward -= 1
        return reward
    '''


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
