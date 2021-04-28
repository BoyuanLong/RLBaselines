import sys
from contextlib import closing

import numpy as np
from io import StringIO


class MDP(object):
    def __init__(self, ns, na, tran_path=None, r_path=None):
        super().__init__()
        self.ns = ns
        self.na = na
        self.s = 0

        if tran_path is None:
            self.matrix = np.random.rand(ns, na, ns)
            sum = self.matrix.sum(axis=-1, keepdims=True)
            self.matrix = self.matrix / sum
        else:
            self.matrix = np.load(tran_path)
        
        self.rewards = np.zeros(self.ns)
        self.rewards[self.ns-1] = 1
        # if r_path is None:
        #     self.rewards = np.random.rand(ns, na)
        # else:
        #     self.rewards = np.load(r_path)
        
    def reset(self):
        self.s = 0
        return self.s

    def step(self, a):
        next_s = np.random.choice(self.ns, p=self.matrix[self.s, a])
        r = self.rewards[next_s]
        return next_s, r, False, None

if __name__ == '__main__':
    np.random.seed(111)
    env = MDP(10, 5, 'test.npy', 'r_file.npy')

    s = env.s
    print(env.rewards)
    # print(env.matrix)
    print(env.s)
    print(env.step(s, 3))
    print(env.matrix.shape)