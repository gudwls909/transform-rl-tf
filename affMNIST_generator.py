import pickle
import numpy as np
import argparse
from os.path import join
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

import util

parser = argparse.ArgumentParser(description="Pendulum")
parser.add_argument('--add55', default=False, action='store_true')
parser.add_argument('--env', default='r', type=str)
args = parser.parse_args()

np2pil = lambda img: Image.fromarray((img.squeeze(2)*255).astype(np.uint8))
pil2np = lambda img: np.expand_dims((np.array(img) / 255.), axis=2)


def main(env_type, add55=False):
    img_size = 40 if env_type in ['rst', 'rsst'] else 28

    # load MNIST
    print('=== load MNIST.... ===')
    mnist = input_data.read_data_sets(join("data", "MNIST_data"), one_hot=False)

    train_images = mnist.train.images.reshape([-1, 28, 28, 1])
    test_images = mnist.test.images.reshape([-1, 28, 28, 1])

    train_labels = mnist.train.labels
    test_labels = mnist.test.labels

    # generate affine MNIST
    print('=== generate affine MNIST.... ===')
    train_inputs = np.zeros([1000, img_size, img_size, 1])
    train_targets = np.zeros([1000, ], dtype=int)
    test_inputs = np.zeros([10000, img_size, img_size, 1])
    test_targets = np.zeros([10000, ], dtype=int)

    for i in range(train_inputs.shape[0]):
        img = train_images[i]
        aff_img = util.random_affine_image(img, env_type)
        train_inputs[i] = aff_img
        train_targets[i] = train_labels[i]

    for i in range(test_inputs.shape[0]):
        img = test_images[i]
        aff_img = util.random_affine_image(img, env_type)
        # aff_img = util.random_affine_image(img, env_type,
        #                                    r_bound=[50, 60],
        #                                    sh_bound=[0.3, 0.6],
        #                                    sc_bound=[1.1, 1.5],
        #                                    t_bound=[-15, -9])
        test_inputs[i] = aff_img
        test_targets[i] = test_labels[i]

    if add55 and env_type in ['rst', 'rsst']:
        train_images40 = np.zeros([train_images.shape[0], 40, 40, 1])
        for i in range(train_images.shape[0]):
            img = train_images[i]
            aff_theta = np.eye(3)[:2, :].flatten()
            pil_img = np2pil(img)
            pil_img = pil_img.transform((40, 40), Image.AFFINE, aff_theta, resample=Image.BICUBIC)
            train_images40[i] = pil2np(pil_img)

    if not add55:
        affMNIST_dataset = [(train_inputs, train_targets), (test_inputs, test_targets)]
    else:
        if env_type not in ['rst', 'rsst']:
            train_inputs = np.concatenate((train_images, train_inputs), axis=0)
        else:
            train_inputs = np.concatenate((train_images40, train_inputs), axis=0)
        train_targets = np.concatenate((train_labels, train_targets), axis=0)
        affMNIST_dataset = [(train_inputs, train_targets), (test_inputs, test_targets)]

    # save affMNIST
    print('=== save affine MNIST... ===')
    if not add55:
        with open(join('data', 'affMNIST_28' + env_type + '.pickle'), 'wb') as f:
            pickle.dump(affMNIST_dataset, f)
    else:
        with open(join('data', 'affMNIST_28' + env_type + '_add56.pickle'), 'wb') as f:
            pickle.dump(affMNIST_dataset, f)


if __name__ == '__main__':
    main(args.env, args.add55)
