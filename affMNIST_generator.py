import pickle
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

from ppo import util


def main():
    # load MNIST
    print('=== load MNIST.... ===')
    mnist = input_data.read_data_sets(join("data","MNIST_data"), one_hot=False)
    
    train_images = mnist.train.images.reshape([-1,28,28,1])[:10000]
    test_images = mnist.test.images.reshape([-1,28,28,1])[:10000]

    train_targets = mnist.train.labels[:10000]
    test_targets = mnist.test.labels[:10000]

    # generate affine MNIST
    print('=== generate affine MNIST.... ===')
    train_inputs = np.zeros([10000,40,40,1])
    test_inputs = np.zeros([10000,40,40,1])

    for i in range(train_inputs.shape[0]):
        img = train_images[i]
        aff_img = util.random_affine_image(img)
        train_inputs[i] = aff_img

    for i in range(test_inputs.shape[0]):
        img = test_images[i]
        aff_img = util.random_affine_image(img)
        test_inputs[i] = aff_img

    affMNIST_dataset = [(train_inputs, train_targets), (test_inputs, test_targets)]

    # save affMNIST
    print('=== save affine MNIST... ===')
    with open(join('data','affMNIST.pickle'),'wb') as f:
        pickle.dump(affMNIST_dataset, f)


if __name__ == '__main__':
    main() 