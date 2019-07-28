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
    
    def __init__(self, sess, phase='test'):
        self.sess = sess
        self.phase = phase 
        self.data_dir = 'MNIST_data' 
        self.ckpt_dir = 'origin_model/checkpoint'
        self.batch_size = 128
        self.input_size = 28
        self.image_c = 1
        self.label_n = 10
        self.nf = 32
#         self.lr = 1e-4 
        
        # hyper parameter for building module
        OPTIONS = namedtuple('options', ['nf', 'label_n', 'phase'])
        self.options = OPTIONS(self.nf, self.label_n, self.phase)
        
        # build model & make checkpoint saver
        self.build_model()
        
    
    def build_model(self):
        # placeholder
        self.input_images = tf.placeholder(tf.float32, 
                                          [None,self.input_size,self.input_size,self.image_c],
                                          name='input_images')
        self.labels = tf.placeholder(tf.int64, [None], name='labels')
        
        # loss funciton
        self.score = mnist_model.classifier(self.input_images, self.options, reuse=True, name='convnet')
        self.pred = tf.nn.softmax(self.score, axis=1)
        self.loss = mnist_model.cls_loss(logits=self.score, labels=self.labels)
        
        # accuracy
        corr = tf.equal(tf.argmax(self.pred, 1), self.labels)    
        self.accr_count = tf.reduce_sum(tf.cast(corr, "float"))

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver()

#         # load checkpoint
#         if self.checkpoint_load():
#             print(" [*] checkpoint load SUCCESS ")
#         else:
#             print(" [!] checkpoint load failed ")

 
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
        
