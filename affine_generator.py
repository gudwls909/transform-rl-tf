import pickle
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from tensorflow.keras.datasets.cifar10 import load_data as cifar10_load_data

import torchvision as tv

import util


def main(env_type, data_type):

    if data_type == 'cifar10':
        # load CIFAR10
        print('=== load CIFAR10.... ===')
        (train_images, train_targets), (test_images, test_targets) = cifar10_load_data()

        train_images = train_images / 255.
        test_images = test_images / 255.
        train_targets = train_targets.squeeze()
        test_targets = test_targets.squeeze()

        aff_filename = 'affCIFAR_32'
        
    elif data_type == 'svhn':
        # load SVHN
        print('=== load SVHN.... ===')
        svhn_train = tv.datasets.SVHN(join('data','svhn'), split='train', download=True)
        svhn_test = tv.datasets.SVHN(join('data','svhn'), split='test', download=True)

        train_images = svhn_train.data.transpose(0,2,3,1) / 255.
        test_images = svhn_test.data.transpose(0,2,3,1) / 255.
        train_targets = ((svhn_train.labels-1)%10).astype(np.uint8)
        test_targets = ((svhn_test.labels-1)%10).astype(np.uint8)

        aff_filename = 'affSVHN_32'

    elif data_type == 'stl10':
        # load STL10
        print('=== load STL10... ===')
        train_images = stl10_train.data.transpose(0,2,3,1) / 255.
        test_images = stl10_test.data.transpose(0,2,3,1) / 255.

        train_targets = stl10_train.labels.astype(np.uint8)
        test_targets = stl10_test.labels.astype(np.uint8)

        aff_filename = 'affSTL_96'

    B_train, H, W, C = train_images.shape
    B_test, H, W, C = test_images.shape

    # generate affine dataset
    print('=== generate affined dataset.... ===')

    if env_type == 'rsst':
        if data_type in ['cifar10','svhn']:
            H, W = 50, 50
        elif data_type in ['stl10']:
            H, W = 110, 110

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
    print('=== save affined dataset... ===')
    with open(join('data', aff_filename + env_type + '.pickle'), 'wb') as f:
        pickle.dump(affine_dataset, f)


if __name__ == '__main__':
    main(env_type)
