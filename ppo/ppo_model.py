import numpy as np
import tensorflow as tf


class PPO(object):
    def __init__(self, state_size, action_size, sess, learning_rate, discount_factor, replay, epsilon, a_bound,
                 state_shape):
        self.state_size = state_size
        self.action_size = action_size
        self.sess = sess
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.replay = replay
        self.eps = epsilon
        self.action_limit = a_bound
        self.action_range = (self.action_limit[:, 1:] - self.action_limit[:, :1]) / 2.
        self.action_limit_mean = (self.action_limit[:, :1] + self.action_limit[:, 1:]) / 2.

        self.state_shape = state_shape

        self.state = tf.placeholder(tf.float32, [None, self.state_size])
        self.advantage = tf.placeholder(tf.float32, [None, 1])
        self.actions = tf.placeholder(tf.float32, [None, self.action_size])
        self.std = tf.placeholder(tf.float32, [None, self.action_size])
        self.old_policy = tf.placeholder(tf.float32, [None, self.action_size])
        self.old_value = tf.placeholder(tf.float32, [None, 1])
        self.returns = tf.placeholder(tf.float32, [None, 1])

        self.normal = tf.contrib.distributions.Normal(loc=0., scale=self.std)
        # self.actor_target, _ = self.build_actor('actor_target', False)
        self.actor, self.sampled_action = self.build_actor('actor_eval', True)
        # self.critic_target = self.build_critic('critic_target', False)
        self.critic = self.build_critic('critic_eval', True)

        # self.actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_eval')
        # self.actor_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_target')
        # self.critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_eval')
        # self.critic_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_target')

        # self.replace = [tf.assign(t, e) for t, e in zip(self.actor_tmp_vars + self.critic_tmp_vars,
        #                                                self.actor_vars + self.critic_vars)]

        self.train, self.loss = self.optimizer()
        pass

    def build_actor(self, scope, trainable):
        with tf.variable_scope(scope):
            state = tf.reshape(self.state, [-1, self.state_shape[0], self.state_shape[1], self.state_shape[2]])
            conv1 = tf.layers.conv2d(inputs=state, filters=32, kernel_size=[3, 3], padding='SAME',
                                     activation=tf.nn.relu, trainable=trainable)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, padding="SAME")
            conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding='SAME',
                                     activation=tf.nn.relu, trainable=trainable)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding='SAME')
            flat = tf.reshape(pool2, [-1, self.state_shape[0] * self.state_shape[1] * 4])
            dense3 = tf.layers.dense(inputs=flat, units=32, activation=tf.nn.relu, trainable=trainable, name='test')

            # with tf.variable_scope('test', reuse=tf.AUTO_REUSE):
            #    self.w = tf.get_variable('kernel')

            m = tf.layers.dense(dense3, self.action_size, activation=tf.nn.tanh, trainable=trainable)
            # output action mean constrained by action limit
            # m = tf.multiply(m, tf.cast(tf.transpose(self.action_range), tf.float32)) \
            #    + tf.cast(tf.transpose(tf.reduce_mean(self.action_limit, axis=1, keepdims=True)), tf.float32)

            # sampled_output = tf.clip_by_value(output.sample(), -1, 1)

            # reparameterization trick
            sampled_output = m + self.normal.sample()  # no clip action
            # sampled_output = tf.clip_by_value(m + self.normal.sample(), -1, 1)  # clip action
            return m, sampled_output  # [batch_size, action_size]
            pass

    def build_critic(self, scope, trainable):
        with tf.variable_scope(scope):
            state = tf.reshape(self.state, [-1, self.state_shape[0], self.state_shape[1], self.state_shape[2]])
            conv1 = tf.layers.conv2d(inputs=state, filters=32, kernel_size=[3, 3], padding='SAME',
                                     activation=tf.nn.relu, trainable=trainable)
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, padding="SAME")
            conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding='SAME',
                                     activation=tf.nn.relu, trainable=trainable)
            pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2, padding='SAME')
            flat = tf.reshape(pool2, [-1, self.state_shape[0] * self.state_shape[1] * 4])
            dense1 = tf.layers.dense(inputs=flat, units=32, activation=tf.nn.relu, trainable=trainable, name='test2')

            # with tf.variable_scope('test2', reuse=True):
            #    self.w2 = tf.get_variable('kernel')

            output = tf.layers.dense(inputs=dense1, units=1, trainable=trainable)

            return output

    def optimizer(self):
        policy = self.normal.log_prob(self.actions - self.actor)
        # old_policy = self.actor_target.log_prob(self.actions)
        # self.policy = policy

        ratio = tf.exp(policy - self.old_policy)
        # self.ratio = ratio

        actor_loss1 = self.advantage * ratio
        actor_loss2 = self.advantage * tf.clip_by_value(ratio, 1 - self.eps, 1 + self.eps)
        actor_loss = tf.reduce_mean(tf.math.minimum(actor_loss1, actor_loss2))

        clipped_values = self.old_value + tf.clip_by_value(self.critic - self.old_value,
                                                           -self.eps,
                                                           self.eps)
        critic_loss1 = tf.losses.mean_squared_error(labels=clipped_values, predictions=self.returns)
        critic_loss2 = tf.losses.mean_squared_error(labels=self.critic, predictions=self.returns)
        critic_loss = tf.reduce_mean(tf.math.maximum(critic_loss1, critic_loss2))

        # no need to use entropy in continuous action space
        # entropy = tf.reduce_mean(self.actor.entropy())
        # self.entropy = tf.log(2*np.pi*np.exp(1)*tf.math.square(self.std))
        # self.entropy = tf.reduce_mean(self.normal.entropy())

        # loss = -actor_loss + 0.5 * critic_loss - 0.01 * entropy
        loss = -actor_loss + 0.5 * critic_loss
        train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        return train_op, [actor_loss, critic_loss]
        pass

    def train_network(self, std):
        # sample mini-batch data from replay buffer
        states, actions, rewards, next_states, terminals, old_policies, old_values, gaes, returns = self.replay.mini_batch()

        # action bound to (-1, 1)
        actions = (actions - np.transpose(self.action_limit_mean)) / np.transpose(self.action_range)

        # normalize gae
        # gaes = np.asarray(gaes, dtype=np.float32)
        # gaes = (gaes - gaes.mean(axis=0)) / gaes.std(axis=0)

        # calculate target Q value for critic update
        # next_target_v = self.sess.run(self.critic_target, feed_dict={self.state: next_states})
        # target = []
        # for i in range(self.replay.batch_size):
        #    if terminals[i]:
        #        target.append(rewards[i])
        #    else:
        #        target.append(rewards[i] + self.discount_factor * next_target_v[i])
        # target = np.reshape(target, [self.replay.batch_size, 1])

        # train
        _, loss = self.sess.run([self.train, self.loss], feed_dict={self.state: states, self.advantage: gaes,
                                                                    self.actions: actions,
                                                                    self.old_policy: old_policies,
                                                                    self.old_value: old_values, self.returns: returns,
                                                                    self.std: std})

        # a = self.sess.run([self.actor],
        #                  feed_dict={self.state: states, self.advantage: gaes,
        #                             self.actions: actions, self.old_policy: old_policies,
        #                             self.old_value: old_values, self.returns: returns, self.std: std})

        return loss
        pass

    def update_target_network(self):
        # update target from tmp network
        self.sess.run(self.replace)
        pass

    def update_tmp_network(self):
        # update tmp from main network
        self.sess.run(self.replace_to_tmp)
        pass
