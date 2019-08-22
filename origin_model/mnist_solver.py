import os
import numpy as np
import random
import scipy.misc as scm
import tensorflow as tf

from collections import namedtuple
from tqdm import tqdm
from glob import glob
from tensorflow.examples.tutorials.mnist import input_data

from . import mnist_model


class Network(object):

    def __init__(self, sess, input_size=28, learner='cnn', phase='test'):
        self.sess = sess
        self.phase = phase
        self.data_dir = 'MNIST_data'
        self.ckpt_dir = 'origin_model/checkpoint'
        self.learner = learner
        self.batch_size = 128
        self.input_size = input_size
        self.image_c = 1
        self.label_n = 10
        self.nf = 32
        self.lr = 1e-4
        self.epochs = 3

        # hyper parameter for building module
        OPTIONS = namedtuple('options', ['nf', 'label_n', 'input_size', 'phase'])
        self.options = OPTIONS(self.nf, self.label_n, self.input_size, self.phase)

        # build model & make checkpoint saver
        self.build_model()

    def build_model(self):
        # placeholder
        self.input_images = tf.placeholder(tf.float32,
                                           [None, self.input_size, self.input_size, self.image_c],
                                           name='input_images')
        self.labels = tf.placeholder(tf.int64, [None], name='labels')

        # loss funciton
        self.score = mnist_model.classifier(self.input_images, self.options, learner=self.learner, name='convnet')
        self.pred = tf.nn.softmax(self.score, axis=1)
        self.loss = mnist_model.cls_loss(logits=self.score, labels=self.labels)
        self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # accuracy
        corr = tf.equal(tf.argmax(self.pred, 1), self.labels)
        self.accr_count = tf.reduce_sum(tf.cast(corr, "float"))

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

    #         # load checkpoint
    #         self.checkpoint_load()

    def test(self, inputs):
        '''
        inputs: size = [batch_size, 28, 28, 1]
        targets: size = [batch_size]
        '''
        feeds = {
            self.input_images: inputs
        }
        preds = self.sess.run(self.pred, feed_dict=feeds)

        return preds

    def test_accuracy(self, images, labels):
        """
        Args:
            images(np.array): shape = [B, H, W, C]
            labels(np.array): shape = [B,]
        """
        count, corrects = 0., 0.
        for i in range(images.shape[0] // self.batch_size):
            inputs = images[i * self.batch_size:(i + 1) * self.batch_size]
            targets = labels[i * self.batch_size:(i + 1) * self.batch_size]

            preds = self.test(inputs)
            labels_hat = preds.argmax(axis=1)

            count += self.batch_size
            corrects += sum(labels_hat == targets)

        return corrects / count, corrects, count

    def train(self, train_images, train_labels, test_images, test_labels):
        best_acc, _, _ = self.test_accuracy(test_images, test_labels)
        count = 0
        for ep in range(self.epochs):
            shuffle_idxs = np.random.permutation(train_images.shape[0])
            train_images = np.take(train_images, shuffle_idxs, axis=0)
            train_labels = np.take(train_labels, shuffle_idxs, axis=0)

            for i in range(train_images.shape[0] // self.batch_size):
                count += 1
                inputs = train_images[i * self.batch_size:(i + 1) * self.batch_size]
                targets = train_labels[i * self.batch_size:(i + 1) * self.batch_size]

                feeds = {self.input_images: inputs, self.labels: targets}
                _, loss = self.sess.run([self.optim, self.loss], feed_dict=feeds)

                if count % 100 == 0:
                    valid_acc, _, _ = self.test_accuracy(test_images, test_labels)
                    print(f'Epoch: {ep + 1}, Iter: {count:04d}, Best Acc: {valid_acc:.05f}, Loss: {loss:.04f}')

                    if valid_acc > best_acc:
                        self.checkpoint_save()
                        best_acc = valid_acc
                        print(f'best =====> {best_acc:.05f}')

    def checkpoint_save(self):
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        model_name = "net.model"
        self.saver.save(self.sess,
                        os.path.join(self.ckpt_dir, model_name))

    def checkpoint_load(self):
        print(" [*] Reading checkpoint...")

        ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.ckpt_dir, ckpt_name))
            print(" [*] checkpoint load SUCCESS ")
            return True
        else:
            print(" [!] checkpoint load failed ")
            return False
