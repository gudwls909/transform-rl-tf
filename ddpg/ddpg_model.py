import numpy as np
import tensorflow as tf

class DDPG(object):
    def __init__(self, state_size,  action_size, sess, learning_rate,
                 replay, discount_factor, a_bound):
        self.state_size = state_size
        self.action_size = action_size
        self.sess = sess
        self.lr_actor = learning_rate
        self.lr_critic = learning_rate
        self.replay = replay
        self.discount_factor = discount_factor
        self.action_limit = a_bound
        self.action_range = self.action_limit[:, 1:] - self.action_limit[:, :1]

        self.state = tf.placeholder(tf.float32, [None, self.state_size])
        self.target = tf.placeholder(tf.float32, [None, 1])

        self.actor = self.build_actor('actor_eval', True)
        self.actor_target = self.build_actor('actor_target', False)
        self.critic = self.build_critic('critic_eval', True, self.actor)
        self.critic_target = self.build_critic('critic_target', False, self.actor_target)

        self.actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_eval')
        self.actor_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='actor_target')
        self.critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_eval')
        self.critic_target_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='critic_target')

        self.replace = [tf.assign(t, (1 - 0.01) * t + 0.01 * e) 
                        for t, e in zip(self.actor_target_vars + self.critic_target_vars, self.actor_vars + self.critic_vars)]

        self.train_actor = self.actor_optimizer()
        self.train_critic = self.critic_optimizer()
        pass

    def build_actor(self, scope, trainable):
        actor_hidden_size = 30
        with tf.variable_scope(scope):
            hidden1 = tf.layers.dense(self.state, actor_hidden_size, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(hidden1, self.action_size, activation=tf.nn.tanh, name='a', trainable=trainable)

#             a = tf.multiply(a, tf.cast(tf.transpose(self.action_range), tf.float32)) / 2. \
#                 + tf.cast(tf.transpose(tf.reduce_mean(self.action_limit, axis=1, keepdims=True)), tf.float32)

            return a

    def build_critic(self, scope, trainable, a):
        with tf.variable_scope(scope):
            critic_hidden_size =30
            hidden1 = tf.layers.dense(self.state, critic_hidden_size, name='s1', trainable=trainable) \
                      + tf.layers.dense(a, critic_hidden_size, name='a1', trainable=trainable) \
                      + tf.get_variable('b1', [1, critic_hidden_size], trainable=trainable)
            hidden1 = tf.nn.relu(hidden1)
            return tf.layers.dense(hidden1, 1, trainable=trainable)

    def actor_optimizer(self):
        loss = tf.reduce_mean(self.critic)
        train_op = tf.train.AdamOptimizer(-self.lr_actor).minimize(loss, var_list=self.actor_vars)

        return train_op
        pass

    def critic_optimizer(self):
        loss = tf.losses.mean_squared_error(labels=self.target, predictions=self.critic)
        #loss = tf.reduce_mean(tf.square(self.target - self.critic))
        train_op = tf.train.AdamOptimizer(self.lr_critic).minimize(loss, var_list=self.critic_vars)
        return train_op
        pass

    def train_network(self):
        states, actions, rewards, next_states, terminals = self.replay.mini_batch()

        next_target_q = self.sess.run(self.critic_target, feed_dict={self.state: next_states})

        target = []
        for i in range(self.replay.batch_size):
            if terminals[i]:
                target.append(rewards[i])
            else:
                target.append(rewards[i] + self.discount_factor * next_target_q[i])
        target = np.reshape(target, [self.replay.batch_size, 1])

        self.sess.run(self.train_actor, feed_dict={self.state: states})
        self.sess.run(self.train_critic, feed_dict={self.state: states, self.target: target, self.actor: actions})
        pass

    def update_target_network(self):
        self.sess.run(self.replace)
        pass
