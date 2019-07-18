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

    def add(self, state, action, reward, next_state, terminal, gae):
        self.memory.append((state, action, reward, next_state, terminal, gae))
        pass

    def mini_batch(self):
        mini_batch = random.sample(self.memory, self.batch_size)  # memory에서 random하게 sample

        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, terminals, gaes = [], [], [], []
        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            next_states[i] = mini_batch[i][3]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            terminals.append(mini_batch[i][4])
            gaes.append(mini_batch[i][5])
        return states, np.asarray(actions), rewards, next_states, terminals, gaes
        pass

    def clear(self):
        self.memory.clear()
        pass
