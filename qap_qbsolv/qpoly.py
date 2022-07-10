'''
Description: 
Author: BO Jianyuan
Date: 2020-11-23 14:59:14
LastEditors: BO Jianyuan
LastEditTime: 2021-09-03 22:08:09
'''

import numpy as np

class QPoly(object):
    """Quadratic Polynomial class
       Arguments:
           n:                 The number of variables that can be handled by this QPoly.
       Attributes:
           array:             The numpy array showing this QPoly.
           constant:          The constant value of this QPoly.
    """
    def __init__(self, n=1024):
        self.array = np.zeros((n, n), dtype=np.float64)  # note input must be int, i have change it to float
        self.constant = 0
        self._size = n

    def add_term(self, c, i, j=None):
        """Add a term 'c * x_i * x_j' to this QPoly"""
        if j is None:
            j = i
        if i >= self._size or j >= self._size:
            raise RuntimeError('wrong var number')
        if i > j:
            self.array[j][i] += c
        else:
            self.array[i][j] += c

    def add_constant_term(self, c):
        """Add a constant term 'c' to this QPoly"""
        self.constant += c

    def power_(self, p=2):
        """Raise this QPoly to the second power"""
        diag = np.diag(self.array)
        if np.count_nonzero(self.array - np.diag(diag)) > 0 or p != 2:
            raise RuntimeError('not quadratic')
        a = np.outer(diag, diag)
        self.array = np.triu(a, k=1) + np.triu(a.T) + (2 * self.constant * np.diag(diag))
        self.constant = self.constant ** 2
        
    def power(self, p=2):
        """Raise this QPoly to the second power"""
        diag = np.diag(self.array)
        if np.count_nonzero(self.array - np.diag(diag)) > 0 or p != 2:
            raise RuntimeError('not quadratic')
        a = np.outer(diag, diag)
        self.array = a + (2 * self.constant * np.diag(diag))
        self.constant = self.constant ** 2

    def multiply_quadratic_binary_polynomial(self, poly):
        """Multiply this QPoly with a Quadratic Polynomial 'poly'"""
        diag0 = np.diag(self.array)
        diag1 = np.diag(poly.array)
        if diag0.size != diag1.size:
            raise RuntimeError('wrong array size')
        if np.count_nonzero(self.array - np.diag(diag0)) > 0 or np.count_nonzero(poly.array - np.diag(diag1)) > 0:
            raise RuntimeError('not quadratic')
        a = np.outer(diag0, diag1)
        self.array = np.triu(a, k=1) + np.triu(a.T) + (self.constant * np.diag(diag1)) + (poly.constant * np.diag(diag0))
        self.constant *= poly.constant

    def multiply_by_factor(self, f):
        """Multiply all terms in this QPoly by a constant value 'f'"""
        self.array *= f
        self.constant *= f

    def sum(self, p):
        """Add a Quadratic Polynomial 'p' to this QPoly"""
        if self._size != p._size:
            raise RuntimeError('wrong array size')
        self.array += p.array
        self.constant += p.constant

    def build_polynomial(self):
        """Make a copy of itself"""
        return copy.deepcopy(self)

    def export_dict(self):
        """Convert this QPoly to a dictionary"""
        cells = np.where(self.array != 0)
        ts = [{"coefficient": float(self.array[i][j]), "polynomials": [int(i), int(j)]} for i, j in zip(cells[0], cells[1])]
        if self.constant != 0:
            ts.append({"coefficient": float(self.constant), "polynomials": []})
        return {'binary_polynomial': {'terms': ts}}

    def reset(self):
        """Clear this QPoly"""
        self.array.fill(0)
        self.constant = 0

    def eval(self, conf):
        """Evaluate this Poly with a configuration 'conf'"""
        if self._size < len(conf):
            raise RuntimeError('wrong configuration size')
        val = self.constant
        for i, c in enumerate(conf):
            for j, d in enumerate(conf[i:]):
                if c and d:
                    val += self.array[i][j + i]
        return val

    def remove_var(self, var):
        if var < 0 or self._size <= var:
            raise RuntimeError('wrong var number')
        self.array[:, var] = 0
        self.array[var, :] = 0

    def clone(self):
        qubo = QPoly(self._size)
        qubo.constant = self.constant
        qubo.array = np.copy(self.array)
        return qubo