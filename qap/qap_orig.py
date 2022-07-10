'''
Description: QAP
Author: BO Jianyuan
Date: 2021-01-14 14:27:52
LastEditors: BO Jianyuan
LastEditTime: 2021-09-01 01:24:52
'''

import numpy as np
import itertools


class QAP(object):
    def __init__(self, d, f, perturbed): # perturbed is bool
        def _calculate_product(n, f, d):
            product_term = {}
            product_total = 0
            for (i, j, k, l) in itertools.product(range(n), range(n), range(n),
                                                  range(n)):
                product_term[(i, j, k, l)] = f[i][j] * d[k][l]
                product_total += product_term[(i, j, k, l)]
            return product_term, product_total

        def _create_board(n, d, f):
            total_D = 0
            total_F = 0
            item_counter = 0
            board = {}
            reversed_board = {}
            for (i, j) in itertools.product(range(n), range(n)):
                total_D += d[i][j]
                total_F += f[i][j]
                board[(i, j)] = item_counter
                reversed_board[item_counter] = (i, j)
                item_counter += 1
            return total_D, total_F, item_counter, board, reversed_board

        self.N = max(len(d), len(f))
        self.D = d
        self.F = f
        self.product_term, self.product_total = _calculate_product(
            self.N, self.D, self.F)
        self.total_D, self.total_F, self.item_counter, self.board, self.reversed_board = _create_board(
            self.N, self.D, self.F)
        self.delta_D, self.delta_F = QAP.perturb(self)
        self.perturbed_product_term, self.max_perturbed_product_term = QAP.perturb_product(
            self)
        self.perturbed = perturbed
        self.max_product = QAP.calculate_parameter(self)

    @staticmethod
    def perturb(qap):
        n_D, n_F = len(qap.D), len(qap.F)
        a_D = np.full((n_D**2, n_D**2),
                      -n_D**2) + (n_D**4 - n_D**2) * np.identity(n_D**2)

        a_F = np.full((n_F**2, n_F**2),
                      -n_F**2) + (n_F**4 - n_F**2) * np.identity(n_F**2)

        v_D = np.array([
            qap.product_total - n_D**2 * qap.total_D *
            qap.D[qap.reversed_board[i][0]][qap.reversed_board[i][1]]
            for i in range(n_D**2)
        ])

        v_F = np.array([
            qap.product_total - n_F**2 * qap.total_F *
            qap.F[qap.reversed_board[i][0]][qap.reversed_board[i][1]]
            for i in range(n_F**2)
        ])
        delta_D = np.linalg.lstsq(a_D, v_D, rcond=-1)[0]
        delta_F = np.linalg.lstsq(a_F, v_F, rcond=-1)[0]

        return delta_D, delta_F

    @staticmethod
    def perturb_product(qap):
        perturbed_product_term = {}
        max_perturbed_product_term = 0
        for (i, j, k, l) in itertools.product(range(qap.N), range(qap.N),
                                              range(qap.N), range(qap.N)):
            perturbed_product_term[(
                i, j, k,
                l)] = qap.product_term[(i, j, k, l)] + qap.delta_F[qap.board[
                    (i, j)]] + qap.delta_D[qap.board[(k, l)]]
            max_perturbed_product_term = max(
                max_perturbed_product_term,  # 0
                abs(perturbed_product_term[(i, j, k, l)]))

        for (i, j, k, l) in itertools.product(range(qap.N), range(qap.N),
                                              range(qap.N), range(qap.N)):
            perturbed_product_term[(i, j, k,
                                    l)] /= max_perturbed_product_term
        return perturbed_product_term, max_perturbed_product_term

    @staticmethod
    def calculate_parameter(qap):
        if not qap.perturbed:
            product_term = getattr(qap, 'product_term')
        else:
            product_term = getattr(qap, 'perturbed_product_term')
        n = int(pow(product_term.keys().__len__(), 1 / 4))
        rowF, colF = [0] * n, [0] * n
        rowD, colD = [0] * n, [0] * n

        for (i, j, k, l) in itertools.product(range(n), range(n), range(n),
                                              range(n)):
            rowF[i] += product_term[(i, j, k, l)]
            colF[i] += product_term[(j, i, k, l)]
            rowD[i] += product_term[(j, k, i, l)]
            colD[i] += product_term[(j, k, l, i)]

        return max([
            max(rowF[i] * colD[j], rowD[i] * colF[j]) for i in range(n)
            for j in range(n)
        ])