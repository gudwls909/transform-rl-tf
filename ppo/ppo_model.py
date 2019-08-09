import numpy as np
import tensorflow as tf


class PPO(object):
    def __init__(self, state_size,  action_size, sess, learning_rate, discount_factor, replay, epsilon, a_bound, state_shape):
        self.state_size = state_size
        self.action_size = action_size
        self.sess = sess
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.replay = replay
        self.eps = epsilon
        self.action_limit = a_bound
        self.action_range = self.action_limit[:, 1:] - self.action_limit[:, :1]

        self.state_shape = state_shape

        self.state = tf.placeholder(tf.float32, [None, self.state_size])
        self.target = tf.placeholder(tf.float32, [None, 1])
        self.advantage = tf.placeholder(tf.float32, [None, 1])
        self.actions = tf.placeholder(tf.float32, [None, self.action_size])
        self.std = tf.placeholder(tf.float32, [None, self.action_size])

        self.actor, self.sampled_action = self.build_actor('actor_eval', True)
        self.actor_target, _ = self.build_actor('actor_target', False)
        self.critic = self.build_critic('critic_eval', True,)
        self.critic_target = self.build_critic('critic_target', False)

        self.actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_eval')
        self.actor_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_target')
        self.critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_eval')
        self.critic_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_target')

        self.replace = [tf.assign(t, e)
                   for t, e in zip(self.actor_target_vars + self.critic_target_vars, self.actor_vars + self.critic_vars)]

        self.train, self.loss = self.optimizer()
        pass

    def build_actor(self, scope, trainable):
        with tf.variable_scope(scope):
            state = tf.reshape(self.state, [-1, self.state_shape[0], self.state_shape[1], self.state_shape[2]])
            conv1 = tf.layers.conv2d(inputs=state, filters=32, kernel_size=[3, 3], padding='SAME',
                                     activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, padding="SAME")

            conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding='SAME',
                                     activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding='SAME')

            flat = tf.reshape(pool2, [-1, self.state_shape[0]*self.state_shape[1]*4])
            dense3 = tf.layers.dense(inputs=flat, units=30, activation=tf.nn.relu)

            # output action mean constrained by action limit
            m = tf.layers.dense(dense3, self.action_size, activation=tf.nn.tanh, trainable=trainable)
            m = tf.multiply(m, tf.cast(tf.transpose(self.action_range), tf.float32)) / 2. \
                + tf.cast(tf.transpose(tf.reduce_mean(self.action_limit, axis=1, keepdims=True)), tf.float32)

            output = tf.contrib.distributions.Normal(loc=m, scale=self.std)
            sampled_output = tf.clip_by_value(output.sample(),
                                              self.action_limit[:, 0], self.action_limit[:, 1])
            return output, sampled_output  # [batch_size, action_size]
            pass

    def build_critic(self, scope, trainable):
        with tf.variable_scope(scope):
            state = tf.reshape(self.state, [-1, self.state_shape[0], self.state_shape[1], self.state_shape[2]])
            conv1 = tf.layers.conv2d(inputs=state, filters=32, kernel_size=[3, 3], padding='SAME',
                                     activation=tf.nn.relu)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, padding="SAME")

            conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding='SAME',
                                     activation=tf.nn.relu)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding='SAME')

            flat = tf.reshape(pool2, [-1, self.state_shape[0]*self.state_shape[1]*4])
            output = tf.layers.dense(inputs=flat, units=1, activation=tf.nn.relu)
            return output

    def optimizer(self):
        policy = tf.clip_by_value(self.actor.prob(self.actions), 1e-10, float('nan'))
        policy_old = tf.clip_by_value(self.actor_target.prob(self.actions), 1e-10, float('nan'))

        ratio = policy / policy_old

        min_a = ratio * self.advantage
        min_b = tf.clip_by_value(ratio, 1-self.eps, 1+self.eps) * self.advantage

        actor_loss = tf.reduce_mean(tf.math.minimum(min_a, min_b))
        critic_loss = tf.losses.mean_squared_error(labels=self.target, predictions=self.critic)

        loss = -actor_loss + 0.5 * critic_loss
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        return train_op, loss
        pass

    def train_network(self, std):
        # sample mini-batch data from replay buffer
        states, actions, rewards, next_states, terminals, gaes = self.replay.mini_batch()

        # calculate target Q value for critic update
        next_target_v = self.sess.run(self.critic_target, feed_dict={self.state: next_states})
        target = []
        for i in range(self.replay.batch_size):
            if terminals[i]:
                target.append(rewards[i])
            else:
                target.append(rewards[i] + self.discount_factor * next_target_v[i])
        target = np.reshape(target, [self.replay.batch_size, 1])

        # train
        self.sess.run(self.train, feed_dict={self.state: states, self.advantage: gaes,
                                             self.actions: actions, self.target: target, self.std: std})

        # return train loss
        return self.sess.run(self.loss, feed_dict={self.state: states, self.advantage: gaes,
                                                   self.actions: actions, self.target: target, self.std: std})
        pass

    def update_target_network(self):
        self.sess.run(self.replace)
        pass
