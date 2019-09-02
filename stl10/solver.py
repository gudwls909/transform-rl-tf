import os
import numpy as np
import random
import scipy.misc as scm
from tqdm import tqdm
from glob import glob
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision as tv

from stl10 import selector


class Network(object):

    def __init__(self, sess, input_size=96, learner='cnn', phase='test'):
        self.sess = sess
        self.phase = phase
        self.learner = learner
        self.batch_size = 32
        self.input_size = input_size
        self.image_c = 3
        self.label_n = 10
        self.nf = 32
        self.lr = 1e-4
        self.epochs = 3

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.transform=tv.transforms.Compose([
            tv.transforms.Lambda(lambda img: Image.fromarray(img)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        # build model & make checkpoint saver
        self.build_model()

    def build_model(self):
        self.model = selector.select('stl10')
        self.model = self.model.to(self.device)
        self.model.eval()

    def test(self, inputs):
        '''
        Args:
            images(np.array): shape = [B, H, W, C], value 0~1
                              Unnormalized..
        Returns:
            predictions(np.array): shape = [B,C]
        '''
        inputs = (inputs * 255.).astype(np.uint8)
        torch_inputs = torch.zeros_like(torch.tensor(inputs), dtype=torch.float32).permute(0,3,1,2)
        for i in range(inputs.shape[0]):
            torch_inputs[i] = self.transform(inputs[i])
        torch_inputs = torch_inputs.to(self.device)
            
        scores = self.model(torch_inputs)
        preds = F.softmax(scores, dim=1)
        preds = preds.cpu().detach().numpy()

        return preds

    def test_accuracy(self, images, labels):
        """
        Args:
            images(np.array): shape = [B, H, W, C], value 0~1
            labels(np.array): shape = [B,]
        """

        count, corrects = 0., 0.
        for i in range(images.shape[0] // self.batch_size):
            inputs = images[i * self.batch_size:(i + 1) * self.batch_size]
            targets = labels[i * self.batch_size:(i + 1) * self.batch_size]

            preds = self.test(inputs)
            labels_hat = preds.argmax(axis=1)

            count += self.batch_size
            corrects += sum(labels_hat==targets)

        return corrects / count, corrects, count

