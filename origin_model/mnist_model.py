import tensorflow as tf
import tensorflow.contrib.slim as slim
from stn import spatial_transformer_network as transformer


def classifier(images, options, learner='cnn', name='classifier'):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        #         x = relu(conv2d(images, options.nf, ks=5, s=1, name='conv1'))  # 28*28*nf
        #         if learner == 'stn':
        #             theta = linear(tf.reshape(x, [-1, int(options.input_size * options.input_size * options.nf)]), 128,
        #                            name='loc_linear1')
        #             theta = linear(theta, 6, name='loc_linear2')
        #             x = transformer(x, theta)

        if learner == 'stn':
            theta = linear(tf.layers.flatten(images), 128, name='loc_linear1')
            theta = linear(theta, 6, name='loc_linear2')
            x = transformer(images, theta, [options.input_size, options.input_size])
            x = relu(conv2d(x, options.nf, ks=5, s=1, name='conv1'))  # 28*28*nf
        else:
            x = relu(conv2d(images, options.nf, ks=5, s=1, name='conv1'))  # 28*28*nf

        x = relu(conv2d(x, 2 * options.nf, ks=3, s=2, name='conv2'))  # 14*14*(2*nf)
        x = relu(conv2d(x, 4 * options.nf, ks=3, s=2, name='conv3'))  # 7*7*(4*nf)

        x = linear(tf.layers.flatten(x), 128, name='linear1')
        x = dropout(x, 0.5, options.phase)
        x = linear(x, options.label_n, name='linear2')
        return x


def cls_loss(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))


def conv2d(input_, output_dim, ks=3, s=1, padding='SAME', name='conv2d'):
    with tf.variable_scope(name):
        return slim.conv2d(input_, output_dim, ks, s, padding=padding,
                           weights_initializer=tf.contrib.layers.xavier_initializer())


def linear(input_, output_dim, stddev=0.02, name='linear'):
    with tf.variable_scope(name):
        return slim.fully_connected(input_, output_dim, activation_fn=None,
                                    weights_initializer=tf.contrib.layers.xavier_initializer())


def dropout(x, rate, phase):
    if phase is 'train':
        phase = True
    else:
        phase = False
    return tf.layers.dropout(inputs=x, rate=rate, training=phase)


def average_pooling(x, ks=[2, 2], s=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=ks, strides=s, padding=padding)


def max_pooling(x, ks=[3, 3], s=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=ks, strides=s, padding=padding)


def flatten(x):
    return tf.contrib.layers.flatten(x)


def relu(x, name='relu'):
    return tf.nn.relu(x)