import os
from os.path import join

import pickle
import argparse
import tensorflow as tf

import util
from origin_model.mnist_solver import Network

parser = argparse.ArgumentParser(description="Pendulum")
parser.add_argument('--gpu_number', default='0', type=str)
parser.add_argument('--learner', default='cnn', type=str, help='cnn or stn')
parser.add_argument('--epochs', default=30, type=int)
parser.add_argument('--save_dir', default='r', type=str)
parser.add_argument('--env', default='r', type=str)
parser.add_argument('--test', default=False, action='store_true')
args = parser.parse_args()

args.save_dir = os.path.join('save', args.learner, args.save_dir)

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_number


def main(args):
    ### session
    config = tf.ConfigProto()
    config.log_device_placement = False
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    ### load data print('\n=== Load Data ===')
    env = args.env
    img_size = 40 if env in ['rsst'] else 28
    with open(join('data', 'affMNIST_28' + env + '.pickle'), 'rb') as f:
        train_dataset, test_dataset = pickle.load(f)

    train_images, train_labels = train_dataset[0], train_dataset[1]
    test_images, test_labels = test_dataset[0], test_dataset[1]

    ### train
    if not args.test:
        print(f'\n=== Start Train, {args.learner} learner ===')
        model = Network(sess, input_size=img_size, learner=args.learner, phase='train')
        model.ckpt_dir = join(args.save_dir, 'checkpoint')
        model.epochs = args.epochs
        model.batch_size = 32
        model.checkpoint_load()
        model.train(train_images, train_labels, test_images, test_labels)

    ### test
    tf.reset_default_graph()
    sess = tf.Session(config=config)

    print(f'\n=== Start Test, {args.learner} learner ===')
    model = Network(sess, input_size=img_size, learner=args.learner, phase='test')
    model.ckpt_dir = join(args.save_dir, 'checkpoint')
    model.batch_size = 32
    model.checkpoint_load()
    accuracy, _, _ = model.test_accuracy(test_images, test_labels)
    print(f'Accuracy: {accuracy}')


if __name__ == '__main__':
    main(args)