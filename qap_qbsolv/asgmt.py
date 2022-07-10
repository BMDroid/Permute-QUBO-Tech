'''
Description: Random QUBO class with controllable density weight matrix
Author: BO Jianyuan
Date: 2021-07-18 16:04:29
LastEditors: BO Jianyuan
LastEditTime: 2021-08-20 11:55:30
'''

import numpy as np

class Assignment(object):
    
    def __init__(self, i, j, ratio, low=0, up=1000):
        self.weights = np.random.uniform(low, up, i * j).reshape(i, j).astype(int)
        extra = np.random.uniform(low, up, j * j).reshape(j, j).astype(int)
        self.extra = 0.5 * (extra + extra.T)
        np.fill_diagonal(self.extra, 0)
        # control the density
        ratio = Assignment.get_ratio(ratio)
        bar = low + (1 - ratio) * (up - low)
        flags = self.extra < bar
        self.extra[flags] = 0
        self.A = i
        self.T = j
        self._max_edge = np.max(self.extra)
        self.pairs = Assignment.create_random_job_pairs(self.A, self.T)
        
        
    @staticmethod
    def create_random_job_pairs(A, T, m=3, n=10):
        pairs = []
        flag = False
        # p = np.random.randint(0, T, min(A, (T // m)) * min(3, (T // n))).reshape(min(A, (T // m)), min(3, (T // n)))
        p = np.random.randint(0, T, min(A // 2, (T // m)) * min(3, max(2, (T // n)))).reshape(min(A // 2, (T // m)), min(3, max(2, (T // n))))
        while not flag:
            for row in range(p.shape[0]):
                idx = list(set(p[row, :].tolist()))
                if len(idx) > 1:
                    pairs.append(idx)
                    flag = True
        return pairs
    
    @staticmethod
    def get_ratio(ratio):
        if ratio > 0.5 and ratio != 1:
            return ratio - 0.1
        elif ratio < 0.5:
            return ratio + 0.1
        return ratio