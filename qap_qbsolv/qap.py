'''
Description: QAP
Author: BO Jianyuan
Date: 2021-01-14 14:27:52
LastEditors: BO Jianyuan
LastEditTime: 2021-09-01 01:53:37
'''

import numpy as np
import itertools


class QAP(object):
    def __init__(self, d, f, perturbed): # perturbed is bool
        
        def _calculate_product(n, d, f):
            product_term = d.reshape(n**2, 1) @ f.reshape(1, n**2)
            product_total = np.sum(product_term)
            return product_term, product_total

        def _create_board(n, d, f):
            total_D = np.sum(d)
            total_F = np.sum(f)
            # item_counter = n ** 2
            # board = {(i, j): i * n+j for (i, j) in itertools.product(range(n), range(n))}
            # reversed_board = {v: k for k, v in board.items()}
            return total_D, total_F # item_counter, board, reversed_board

        self.N = max(len(d), len(f))
        self.D = d
        self.F = f
        self.product_term, self.product_total = _calculate_product(
            self.N, self.D, self.F)
        self.total_D, self.total_F = _create_board(self.N, self.D, self.F)
        self.delta_D, self.delta_F = QAP.perturb(self)
        self.perturbed_product_term, self.max_perturbed_product_term = QAP.perturb_product(
            self)
        self.perturbed = perturbed
        self.max_product = QAP.calculate_parameter(self)

    @staticmethod
    def perturb(qap):
        n = qap.N
        a_D = np.full((n**2, n**2),
                    -n**2) + (n**4 - n**2) * np.identity(n**2)

        a_F = np.full((n**2, n**2),
                    -n**2) + (n**4 - n**2) * np.identity(n**2)

        v_D = qap.product_total - n**2 * qap.total_D * qap.D
        v_D = v_D.reshape(n**2,)

        v_F = qap.product_total - n**2 * qap.total_F * qap.F
        v_F = v_F.reshape(n**2,)
        
        delta_D = np.linalg.lstsq(a_D, v_D, rcond=-1)[0]
        delta_F = np.linalg.lstsq(a_F, v_F, rcond=-1)[0]

        return delta_D, delta_F

    @staticmethod
    def perturb_product(qap):
        perturbed_product_term = qap.product_term + np.tile(qap.delta_D, (qap.N**2, 1)) + np.tile(qap.delta_F.reshape(-1, 1), (1, qap.N**2))
        max_perturbed_product_term = np.max(np.abs(perturbed_product_term))
        perturbed_product_term /= max_perturbed_product_term
        return perturbed_product_term, max_perturbed_product_term

    @staticmethod
    def calculate_parameter(qap):
        if not qap.perturbed:
            product_term = getattr(qap, 'product_term')
        else:
            product_term = getattr(qap, 'perturbed_product_term')
        n = qap.N
        row = np.sum(product_term, axis=0).reshape(n, n)
        col = np.sum(product_term, axis=1).reshape(n, n)
        row = np.sum(row, axis=0)
        col = np.sum(col, axis=1)
        return np.max(row.reshape(n, 1) @ col.reshape(1, n))