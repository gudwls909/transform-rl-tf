import pickle
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

import util


def main(num_traindata):
    # load MNIST
    print('=== load MNIST.... ===')
    mnist = input_data.read_data_sets(join("data", "MNIST_data"), one_hot=False)

    train_images = mnist.train.images.reshape([-1, 28, 28, 1])[:num_traindata]
    test_images = mnist.test.images.reshape([-1, 28, 28, 1])[:10000]

    train_targets = mnist.train.labels[:num_traindata]
    test_targets = mnist.test.labels[:10000]

    affMNIST_dataset = [(train_images, train_targets), (test_images, test_targets)]

    # save affMNIST
    print('=== save affine MNIST... ===')
    with open(join('data', 'MNIST_' + str(num_traindata) + '.pickle'), 'wb') as f:
        pickle.dump(affMNIST_dataset, f)


if __name__ == '__main__':
    main(num_traindata)
