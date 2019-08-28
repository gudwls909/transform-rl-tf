import pickle
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from tensorflow.keras.datasets.cifar10 import load_data

import util


def main(env_type):
    img_size = 40 if env_type == 'rsst' else 28

    # load CIFAR10
    print('=== load CIFAR10.... ===')
    (train_images, train_targets), (test_images, test_targets) = load_data()

    train_images = train_images / 255.
    test_images = test_images / 255.

    train_targets = train_targets.squeeze()
    test_targets = test_targets.squeeze()

    B_train, H, W, C = train_images.shape
    B_test, H, W, C = test_images.shape

    # generate affine CIFAR10
    print('=== generate affine CIFAR10.... ===')

    if env_type == 'rsst':
        H, W = 50, 50

    train_inputs = np.zeros([B_train, H, W, C])
    test_inputs = np.zeros([B_test, H, W, C])

    for i in range(B_train):
        img = train_images[i]
        aff_img = util.random_affine_image(img, env_type)
        train_inputs[i] = aff_img

    for i in range(B_test):
        img = test_images[i]
        aff_img = util.random_affine_image(img, env_type)
        # aff_img = util.random_affine_image(img, env_type,
        #                                    r_bound=[50, 60],
        #                                    sh_bound=[0.3, 0.6],
        #                                    sc_bound=[1.1, 1.5],
        #                                    t_bound=[-15, -9])
        test_inputs[i] = aff_img

    affine_dataset = [(train_inputs, train_targets), (test_inputs, test_targets)]

    # save affined dataset
    print('=== save affine CIFAR10... ===')
    with open(join('data', 'affCIFAR_32' + env_type + '.pickle'), 'wb') as f:
        pickle.dump(affine_dataset, f)


if __name__ == '__main__':
    main(env_type)
