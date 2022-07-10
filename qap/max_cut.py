'''
Description: Class of Max Cut Instance
Author: BO Jianyuan
Date: 2021-08-19 15:45:11
LastEditors: BO Jianyuan
LastEditTime: 2021-08-24 17:41:44
'''

import numpy as np

class Max_Cut(object):
    
    def __init__(self, txt, n=0, low=0, up=2, ratio=1, k1=0.25, k2=0.25):
        # default low and up will create an weighted max cut instance
        if txt is None:
            self.V = n
            self.low, self.up = low, up
            self.ratio = ratio
            self.k1, self.k2 = k1, k2
            self.W = np.random.randint(low, up, self.V**2).reshape(self.V, self.V)
            self.W = 0.5 * (self.W + self.W.T)
            self.W = self.W.astype(int)
            np.fill_diagonal(self.W, 0)
            # control the density
            ratio = Max_Cut.get_ratio(ratio)
            bar = low + (1 - ratio) * (up - low)
            flags = self.W < bar
            self.W[flags] = 0
        else:
            self.V = int(txt[0].strip().split()[0])
            self.ratio = int(txt[0].strip().split()[1]) / (self.V**2)
            self.W = np.zeros((self.V, self.V))
            for line in txt[1:]:
                i, j, w_ij = map(int, line.split())
                self.W[i - 1, j -1] = w_ij
            self.W = self.W + self.W.T
            self.low, self.up = np.min(self.W), np.max(self.W)
            self.k1, self.k2 = k1, k2
        self._max_edge = np.max(self.W)
        self.S, self.D = Max_Cut.create_random_vertices_pairs(self.V, k1, k2)
        
    @staticmethod
    def create_random_vertices_pairs(V, k1=0.25, k2=0.25):
        S = np.random.choice(range(V), 2 * int(V * k1), replace=False).reshape(int(V * k1), 2).tolist()
        candidates = np.random.choice(range(V), int(V * k2), replace=False).tolist()
        D = []
        count = 0
        while len(candidates) > 3 and count <= len(S):
            l = np.random.randint(2, 4)
            if all([pair not in candidates[:l] for pair in S]):
                D.append(candidates[:l])
                candidates = candidates[l:]
            else:
                random.shuffle(candidates)
                count += 1
        if all([pair not in candidates for pair in S]):
            D.append(candidates)
            return S, D
        return S, D
    
    @staticmethod
    def get_ratio(ratio):
        if ratio > 0.5 and ratio != 1:
            return ratio - 0.1
        elif ratio < 0.5:
            return ratio + 0.1
        return ratio
    