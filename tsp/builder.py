import itertools
import numpy as np

from bisect import bisect
from qpoly import *


def prepare_config(qubo):
    N_states = [1024, 2048, 4096, 8192]
    N_state = N_states[bisect(N_states, qubo._size)]
    # constant
    constant = qubo.constant
    # bias
    bias = np.repeat(-2**63 + 1, N_state).astype(int)
    bias[:qubo._size] = -np.diag(qubo.array).astype(int)
    # weight
    weight = np.zeros((N_state, N_state), dtype=int)
    np.fill_diagonal(qubo.array, 0)
    weight[:qubo._size, :qubo._size] = -(qubo.array + qubo.array.T).astype(int)
    s = np.zeros(weight.shape[0], dtype=int)
    local_field = np.dot(weight, s) + bias
    E = np.dot(np.dot(weight, s), s) / (-2) - np.dot(bias, s) + constant
    return N_state, bias, constant, weight, s, local_field, E


def build_column_rule(N):
    '''Slot constraint
    '''
    board = np.arange(N * N).reshape(N, N)
    qubo = QPoly(N * N)
    for col in range(N):
        tmp = QPoly(N * N)
        for i in board[:, col]:
            tmp.add_term(1, i)
        tmp.add_constant_term(-1)
        tmp.power(2)
        qubo.sum(tmp)
    return qubo


def build_row_rule(N):
    '''City constraint
    '''
    board = np.arange(N * N).reshape(N, N)
    qubo = QPoly(N * N)
    for row in range(N):
        tmp = QPoly(N * N)
        for i in board[row]:
            tmp.add_term(1, i)
        tmp.add_constant_term(-1)
        tmp.power(2)
        qubo.sum(tmp)
    return qubo


def build_edge_rule(G, N):
    '''Edge constraint
    '''
    board = np.arange(N * N).reshape(N, N)
    qubo = QPoly(N * N)
    for u in range(N):
        for v in range(N):
            if G[u, v] <= np.inf:
                for j in range(N):
                    qubo.add_term(G[u, v], board[u][j], board[v][(j + 1) % N])
    return qubo


def build_edge_rule_unperturbed(G, N):
    '''Unperturbed edge constraint
    '''
    board = np.arange(N * N).reshape(N, N)
    qubo = QPoly(N * N)
    for u in range(N):
        for v in range(N):
            if G[u,
                 v] <= np.inf:  # it is an edge, might want to change this condition
                for j in range(N):
                    qubo.add_term(G[u, v], board[u][j], board[v][(j + 1) % N])
    return qubo


def perturb(g):
    '''Data Perturbation
    '''
    n = g.shape[0]
    a = np.full((n, n), -1. / n) + np.identity(n)
    tmp_sum = np.sum(g) / float(n)
    v = np.array([np.sum(g[i, :] + g[:, i]) for i in range(n)])
    v = tmp_sum - v * 0.5
    v = v / (n - 2.0)
    delta = np.linalg.lstsq(a, v, rcond=-1)[0]
    new_g = np.copy(g)
    min_num = float('inf')
    for (i, j) in list(itertools.product(range(n), range(n))):
        if i != j:
            new_g[i, j] += delta[i] + delta[j]
            min_num = min(min_num, new_g[i, j])
    for (i, j) in list(itertools.product(range(n), range(n))):
        if i != j:
            new_g[i, j] -= min_num
    return new_g


def build_QUBO(N, A, rules):
    '''Get QUBO form
    '''
    cons = QPoly(N * N)
    cons.sum(rules["column"])
    if len(A) > 1:
        cons.multiply_by_factor(A[0] / A[1])
        cons.sum(rules["row"])
        cons.multiply_by_factor(A[1])
    else:
        cons.sum(rules["row"])
        cons.multiply_by_factor(A[0])
    obj = rules["edge"]
    qubo = QPoly(N * N)
    qubo.sum(cons)
    qubo.sum(obj)
#     qubo.array = qubo.array.astype(int)
    return qubo


def getEnergy(QPoly, conf):
    '''Gets energy of QUBO mapped configuration
    '''
    E = 0
    for rowIndx in conf.keys():
        if conf[rowIndx]:
            E += QPoly.array[int(rowIndx), int(rowIndx)]  #X**2
            for colIndx in range(int(rowIndx) + 1, conf.__len__()):
                if conf[str(colIndx)]:
                    E += QPoly.array[int(rowIndx), colIndx]  #XY
    return E + QPoly.constant


def compute_perm_makespan(N, conf, seq):
    '''Compute Path if feasible
    '''
    def check_valid_cycle(perm_vector):
        return (not (None in perm_vector)) and (len(set(perm_vector))
                                                == len(perm_vector))

    board = np.arange(N * N).reshape(N, N)
    FSP_perm = [None for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if conf[board[i][j]]:
                FSP_perm[j] = i

    # evaluate the perm cost
    if check_valid_cycle(FSP_perm):
        seq.reorder(FSP_perm)
        perm_makespan = seq._makespan

        # reorder the sequence
        idx = FSP_perm.index(0)
        FSP_perm_reordered = FSP_perm[idx + 1:] + FSP_perm[0:idx + 1]
        seq.reorder(FSP_perm_reordered)
        perm_makespan_reordered = seq._makespan

        if perm_makespan_reordered <= perm_makespan:
            return FSP_perm_reordered, perm_makespan_reordered
        else:
            return FSP_perm, perm_makespan
    return [], -1


def compute_path_distance(N, conf, G):
    '''Compute Path if feasible
    '''
    def check_valid_cycle(path_vector):
        return (not (None in path_vector)) and (len(set(path_vector))
                                                == len(path_vector))

    board = np.arange(N * N).reshape(N, N)
    TSP_path = [None for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if conf[board[i][j]]:
                TSP_path[j] = i
    # evaluate the path cost
    if check_valid_cycle(TSP_path):
        path_cost = sum(
            [G[TSP_path[i], TSP_path[(i + 1) % N]] for i in range(N)])
        return TSP_path, path_cost
    return [], -1


### QAP


def build_row_sum(qap):
    N = qap.N
    qubo = QPoly(N * N)
    for i in range(N):
        tmp = QPoly(N * N)
        for j in range(N):
            tmp.add_term(1, qap.board[(i, j)])
        tmp.add_constant_term(-1)
        tmp.power(2)
        qubo.sum(tmp)
    return qubo


def build_col_sum(qap):
    N = qap.N
    qubo = QPoly(N * N)
    for j in range(N):
        tmp = QPoly(N * N)
        for i in range(N):
            tmp.add_term(1, qap.board[(i, j)])
        tmp.add_constant_term(-1)
        tmp.power(2)
        qubo.sum(tmp)
    return qubo


def build_obj(qap):
    N = qap.N
    qubo = QPoly(N * N)
    for (i, j, k, l) in itertools.product(range(N), range(N), range(N),
                                          range(N)):
        if qap.perturbed:
            qubo.add_term(qap.perturbed_product_term[(i, j, k, l)],
                          qap.board[(i, k)], qap.board[(j, l)])
        else:
            qubo.add_term(qap.product_term[(i, j, k, l)], qap.board[(i, k)],
                          qap.board[(j, l)])
    return qubo


def build_QAP(qap, A, rules):
    '''Get QUBO form
    '''
    N = qap.N
    cons1 = QPoly(N * N)
    cons1.sum(rules["column"])
    cons1.multiply_by_factor(A[0])
    cons2 = QPoly(N * N)
    cons2.sum(rules["row"])
    if len(A) > 1:
        cons2.multiply_by_factor(A[1])
    else:
        cons2.multiply_by_factor(A[0])
    obj = rules["obj"]
    qubo = QPoly(N * N)
    qubo.sum(obj)
    qubo.sum(cons1)
    qubo.sum(cons2)
    return qubo


def compute_cost(qap, conf):
    '''Compute cost if feasible
    '''
    N = qap.N
    y = {}
    for i in conf:
        if conf[i]:
            y[qap.reversed_board[i]] = 1

    # infeasible
    for j in range(N):
        cost = 0
        for i in range(N):
            cost += y.get((i, j), 0)
        if cost != 1:
            return y, -1

    for i in range(N):
        cost = 0
        for j in range(N):
            cost += y.get((i, j), 0)
        if cost != 1:
            return y, -1

    # feasible
    cost = 0
    for (i, j, k, l) in itertools.product(range(N), range(N), range(N),
                                          range(N)):
        cost += qap.product_term[(i, j, k, l)] * y.get((i, k), 0) * y.get(
            (j, l), 0)

    return y, cost