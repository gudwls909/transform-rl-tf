import random
import numpy as np
from collections import deque


class ReplayMemory(object):
    def __init__(self, state_size, batch_size, length):
        self.length = length
        self.memory = deque(maxlen=self.length)
        self.state_size = state_size
        self.batch_size = batch_size
        pass

    def add(self, var_list):
        # state, action, reward, next_state, terminal, old_policy, old_value, gae, return
        self.memory.append(tuple(var_list))
        pass

    def mini_batch(self):
        mini_batch = random.sample(self.memory, self.batch_size)  # memory에서 random하게 sample
        states = np.array([m[0] for m in mini_batch])
        actions = np.array([m[1] for m in mini_batch])
        rewards = [m[2] for m in mini_batch]  # [batch_size,]
        next_states = np.array([m[3] for m in mini_batch])
        terminals = [m[4] for m in mini_batch]  # [batch_size,]
        old_policies = [m[5] for m in mini_batch]
        old_values = [m[6] for m in mini_batch]
        gaes = [[m[7]] for m in mini_batch]
        returns = [[m[8]] for m in mini_batch]
        return states, np.asarray(actions), rewards, next_states, terminals, old_policies, old_values, gaes, returns
        pass

    def clear(self):
        self.memory.clear()
        pass
