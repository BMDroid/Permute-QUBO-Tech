import itertools
import numpy as np

from bisect import bisect
from itertools import combinations
from qpoly import *


def prepare_config(qubo, conf=None):  # conf is the state retrieved from DA
    N_states = [1024, 2048, 4096, 8192]
    N_state = N_states[bisect(N_states, qubo._size)]
    # constant
    constant = qubo.constant
    # bias
    bias = np.repeat(-(2 ** 63) + 1, N_state).astype(int)
    bias[: qubo._size] = -np.diag(qubo.array).astype(int)
    # weight
    weight = np.zeros((N_state, N_state), dtype=int)
    np.fill_diagonal(qubo.array, 0)
    weight[: qubo._size, : qubo._size] = -(qubo.array + qubo.array.T).astype(int)
    if conf is not None:
        s = conf.copy()
    else:
        # s = np.zeconsros(weight.shape[0], dtype=int)
        s = np.random.randint(2, size=weight.shape[0])
    local_field = np.dot(weight, s) + bias
    E = np.dot(np.dot(weight, s), s) / (-2) - np.dot(bias, s) + constant
    return N_state, bias, constant, weight, s, local_field, E

def prepare_config_rl(qubo, conf=None, seed=None):  # conf is the state retrieved from DA
    """ Created for RL to enable config initial state and define random seed
    """
    N_states = [1024, 2048, 4096, 8192]
    N_state = N_states[bisect(N_states, qubo._size)]
    # constant
    constant = qubo.constant
    # bias
    bias = np.repeat(-(2 ** 63) + 1, N_state).astype(int)
    bias[: qubo._size] = -np.diag(qubo.array).astype(int)
    # weight
    weight = np.zeros((N_state, N_state), dtype=int)
    np.fill_diagonal(qubo.array, 0)
    weight[: qubo._size, : qubo._size] = -(qubo.array + qubo.array.T).astype(int)
    if conf is not None:
        s = conf.copy()
    else:
#         s = np.zeros(weight.shape[0], dtype=int)
        if seed is not None:
            np.random.seed(seed)
        s = np.random.randint(2, size=weight.shape[0])
    local_field = np.dot(weight, s) + bias
    E = np.dot(np.dot(weight, s), s) / (-2) - np.dot(bias, s) + constant
    return N_state, bias, constant, weight, s, local_field, E

def prep_config_rnd(qubo, conf=None, seed=None):  # conf is the state retrieved from DA
    """ Created for RL to enable config initial state and start from random initial state
    """
    N_states = [1024, 2048, 4096, 8192]
    N_state = N_states[bisect(N_states, qubo._size)]
    # constant
    constant = qubo.constant
    # bias
    bias = np.repeat(-(2 ** 63) + 1, N_state).astype(int)
    bias[: qubo._size] = -np.diag(qubo.array).astype(int)
    # weight
    weight = np.zeros((N_state, N_state), dtype=int)
    np.fill_diagonal(qubo.array, 0)
    weight[: qubo._size, : qubo._size] = -(qubo.array + qubo.array.T).astype(int)
    if conf is not None:
        s = conf.copy()
    else:
        if seed is not None:
            np.random.seed(seed)
        s = np.random.randint(2, size=weight.shape[0])
    local_field = np.dot(weight, s) + bias
    E = np.dot(np.dot(weight, s), s) / (-2) - np.dot(bias, s) + constant
    return N_state, bias, constant, weight, s, local_field, E

def prep_config_zero(qubo, conf=None, seed=None):  # conf is the state retrieved from DA
    """ Created for RL to enable config initial state and start from all zero initial state
    """
    N_states = [1024, 2048, 4096, 8192]
    N_state = N_states[bisect(N_states, qubo._size)]
    # constant
    constant = qubo.constant
    # bias
    bias = np.repeat(-(2 ** 63) + 1, N_state).astype(int)
    bias[: qubo._size] = -np.diag(qubo.array).astype(int)
    # weight
    weight = np.zeros((N_state, N_state), dtype=int)
    np.fill_diagonal(qubo.array, 0)
    weight[: qubo._size, : qubo._size] = -(qubo.array + qubo.array.T).astype(int)
    if conf is not None:
        s = conf.copy()
    else:
        s = np.zeros(weight.shape[0], dtype=int)
    local_field = np.dot(weight, s) + bias
    E = np.dot(np.dot(weight, s), s) / (-2) - np.dot(bias, s) + constant
    return N_state, bias, constant, weight, s, local_field, E

### MAX CUT
def build_cut_weight(W, V):
    ''' qubo for the objective of the max cut
    Args: 
        W::np.array
            weight matrix of the graph
        V::int
            number of vertices in the graph
    Returns:
        qubo::Qpoly
            object
    '''
    qubo = QPoly(V)
    for i in range(V):
        for j in range(V):
            qubo.add_term(- W[i, j], i)
            qubo.add_term(- W[i, j], j)
            qubo.add_term(2 * W[i, j], i, j)
    return qubo

# same set rule
# $x_i = x_j$ ⇒ $P(x_i+x_j-2x_ix_j)$
def build_cut_same_rule(V, S):
    ''' create rule of a pair of vertices that are in the same set
    Args: 
        W::np.array
            weight matrix of the graph
        V::int
            number of vertices in the graph
        S::[[int, int]]
            list of tuple (index of the vertices)
    Returns:
        qubo::QPoly
            same set rule
    '''
    qubo = QPoly(V)
    for (i, j) in S:
        qubo.add_term(1, i)
        qubo.add_term(1, j)
        qubo.add_term(-2, i, j)
    return qubo

# different set rule
# $x_i + x_j = 1$ ⇒ $P(1-x_i-x_j+2x_ix_j)$
def build_cut_diff_rule(V, D):
    ''' create rule of a pair of vertices that are in the different sets
    Args: 
        W::np.array
            weight matrix of the graph
        V::int
            number of vertices in the graph
        D::[[int, int]]
            list of lists (index of the vertices)
    Returns:
        qubo::QPoly
            different set rule
    '''
    # for pair (i, j)
    # qubo = QPoly(V)
    # for (i, j) in D:
    #     qubo.add_constant_term(1)
    #     qubo.add_term(-1, i)
    #     qubo.add_term(-1, j)
    #     qubo.add_term(2, i, j)
    # return qubo

    # allow group of vertices (i, j, k)
    qubo = QPoly(V)
    for group in D:
        tmp = QPoly(V)
        tmp.add_constant_term(1)
        for i in group:
            tmp.add_term(-1, i)
        for (i, j) in combinations(group, 2):
            tmp.add_term(2, i, j)
        qubo.sum(tmp)
    return qubo

def build_cut_QUBO(para, V, rules):
    cons = QPoly(V)
    cons.sum(rules["same"])
    if len(para) > 1:
        cons.multiply_by_factor(para[0] / para[1])
        cons.sum(rules["diff"])
        cons.multiply_by_factor(para[1])
    else:
        cons.sum(rules["diff"])
        cons.multiply_by_factor(para[0])
    obj = rules["weight"]
    qubo = QPoly(V)
    qubo.sum(cons)
    qubo.sum(obj)
    return qubo

def compute_cut_cost(W, V, S, D, conf):
    def _check_feasible(dec, S, D):
        for (i, j) in S:
            if dec[i] != dec[j]:
                return False
        for group in D:
            if np.sum(dec[group]) != 1:
                return False
        return True 
            
    dec = conf[:V] # .reshape(1, V)
    if _check_feasible(dec, S, D):
        dec_concat = np.tile(dec, (V, 1))
        cost = 0.5 * np.sum(np.subtract(dec, dec_concat.T)**2 * W)
        return dec, cost
    return None, 0

### Assignment
def build_agent_rule(A, T):
    # each agent should and only get one task
    # \sum_j^T x_ij = 1, \forall i \in A
    board = np.arange(A * T).reshape(A, T)
    qubo = QPoly(A * T)
    for i in range(A):
        tmp = QPoly(A * T)
        for j in board[i]:
            tmp.add_term(1, j)
        tmp.add_constant_term(-1)
        tmp.power(2)
        qubo.sum(tmp)
    return qubo

def build_task_rule(A, T):
    # each task could be assigned to at most one agent
    # \sum_i^N x_ij <= 1, \forall j \in T
    board = np.arange(A * T).reshape(A, T)
    qubo = QPoly(A * T)
    for j in range(T):
        tmp = QPoly(A * T)
        for idx, i in enumerate(board[:, j]):
            for k in board[:, j][idx + 1:]:
                tmp.add_term(1, i, k)
        qubo.sum(tmp)     
    return qubo

def build_weights_rule(weights, A, T):
    # \sum_i \sum_j - w_ij * x_ij
    board = np.arange(A * T).reshape(A, T)
    qubo = QPoly(A * T)
    for i in range(A):
        for j in range(T):
            if weights[i, j] <= np.inf:
                qubo.add_term(- weights[i, j], board[i][j])
    return qubo

def build_extra_rule(extra, A, T):
    # \sum_p \sum_q - w_pq * x_ip * x_jq, p, q \in T
    board = np.arange(A * T).reshape(A, T)
    qubo = QPoly(A * T)
    for i in range(A):
        tmp = QPoly(A * T)
        for j in range(i + 1, A):
            for idx1, p in enumerate(board[i]):
                for idx2, q in enumerate(board[j]):
                    if idx1 != idx2:
                        qubo.add_term(- extra[idx1, idx2], p, q)
        qubo.sum(tmp)
    return qubo
    
def build_asgmt_QUBO(para, A, T, rules):
    cons = QPoly(A * T)
    cons.sum(rules["agent"])
    if len(para) > 1:
        cons.multiply_by_factor(para[0] / para[1])
        cons.sum(rules["task"])
        cons.multiply_by_factor(para[1])
    else:
        cons.sum(rules["task"])
        cons.multiply_by_factor(para[0])
    obj1 = rules["weight"]
    obj2 = rules["extra"]
    qubo = QPoly(A * T)
    qubo.sum(cons)
    qubo.sum(obj1)
    qubo.sum(obj2)
    return qubo

def compute_asgmt_cost(A, T, weights, extra, conf):
    def check_feasible(dec, A):
        return np.sum(dec) == A and np.array_equal(np.sum(dec, -1), np.ones(A))
            
#     board = np.arange(A * T).reshape(A, T)
#     dec = np.zeros((A, T))
#     for i in range(A):
#         for j in range(T):
#             if conf[board[i][j]]:
#                 dec[i, j] = 1
    dec = conf[:A * T].reshape(A, T)
    # evaluate the cost
    if check_feasible(dec, A):
        cost = - np.sum(dec * weights) - 0.5 * np.sum(dec, 0) @ extra @ np.sum(dec, 0).T
        return dec, cost
    return None, 0

# Additional constraints
def build_pairs_rule(pairs, A, T):
    # \sum_i^A x_{ip} + \sum_j^A x_{jq} \geq 1, \forall p, q \in T
    # QUBO => P(1 - \sum_i^A x_{ip} - \sum_i^A x_{ip} + \sum_i^A \sum_j^A x_{ip}x_{jq}), \forall p, q \in T
    board = np.arange(A * T).reshape(A, T)
    qubo = QPoly(A * T)
    for pair in pairs:
        tmp = QPoly(A * T)
        for j in pair:
            for i in range(A):
                tmp.add_term(-1, board[i][j])
        for (p, q) in combinations(pair, 2):
            for i, j in itertools.product(range(A), range(A)):
                tmp.add_term(1, board[i][p], board[j][q])
        tmp.add_constant_term(1)
        qubo.sum(tmp)
    return qubo

def build_cstr_asgmt_QUBO(para, A, T, rules):
    cons = QPoly(A * T)
    
    if len(para) > 1:
        cons1 = QPoly(A * T)
        cons1.sum(rules["agent"])
        cons1.multiply_by_factor(para[0])
        cons2 = QPoly(A * T)
        cons2.sum(rules["task"])
        cons2.multiply_by_factor(para[1])
        cons3 = QPoly(A * T)
        cons3.sum(rules["pair"])
        cons3.multiply_by_factor(para[2])
        cons.sum(cons1)
        cons.sum(cons2)
        cons.sum(cons3)
    else:
        cons.sum(rules["agent"])
        cons.sum(rules["task"])
        cons.sum(rules["pair"])
        cons.multiply_by_factor(para[0])
    obj1 = rules["weight"]
    obj2 = rules["extra"]
    qubo = QPoly(A * T)
    qubo.sum(cons)
    qubo.sum(obj1)
    qubo.sum(obj2)
    return qubo

def compute_cstr_asgmt_cost(A, T, weights, extra, pairs, conf):
    def check_feasible(dec, A, pairs):
        return np.sum(dec) == A and np.array_equal(np.sum(dec, axis=-1), np.ones(A)) and all([np.sum(np.sum(dec, axis=0)[pair]) >= 1 for pair in pairs])
            
    dec = conf[:A * T].reshape(A, T)
    # evaluate the cost
    if check_feasible(dec, A, pairs):
        cost = - np.sum(dec * weights) - 0.5 * np.sum(dec, 0) @ extra @ np.sum(dec, 0).T
        return dec, cost
    return None, 0
        
### TSP
def build_column_rule(N):
    """Slot constraint
    """
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
    """City constraint
    """
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
    """Edge constraint
    """
    board = np.arange(N * N).reshape(N, N)
    qubo = QPoly(N * N)
    for u in range(N):
        for v in range(N):
            if G[u, v] <= np.inf:
                for j in range(N):
                    qubo.add_term(G[u, v], board[u][j], board[v][(j + 1) % N])
    return qubo


def build_edge_rule_unperturbed(G, N):
    """Unperturbed edge constraint
    """
    board = np.arange(N * N).reshape(N, N)
    qubo = QPoly(N * N)
    for u in range(N):
        for v in range(N):
            if G[u, v] <= np.inf:  # it is an edge, might want to change this condition
                for j in range(N):
                    qubo.add_term(G[u, v], board[u][j], board[v][(j + 1) % N])
    return qubo

### TSP with first city inserted into the first Slot
def build_column_rule_one_less(N):
    """Slot constraint
    """
    board = np.arange((N - 1)**2).reshape(N - 1, N - 1)
    qubo = QPoly((N - 1)**2)
    for col in range(N - 1):
        tmp = QPoly((N - 1)**2)
        for i in board[:, col]:
            tmp.add_term(1, i)
        tmp.add_constant_term(-1)
        tmp.power(2)
        qubo.sum(tmp)
    return qubo

def build_row_rule_one_less(N):
    """City constraint
    """
    board = np.arange((N - 1)**2).reshape(N - 1, N - 1)
    qubo = QPoly((N - 1)**2)
    for row in range(N - 1):
        tmp = QPoly((N - 1)**2)
        for i in board[row]:
            tmp.add_term(1, i)
        tmp.add_constant_term(-1)
        tmp.power(2)
        qubo.sum(tmp)
    return qubo

def build_edge_rule_one_less(G, N):
    """Edge constraint while the start city and end city is the 0th city
    """
    qubo = QPoly((N - 1)**2)
    for t in range(N - 2):
        for u in range(1, N):
            for v in range(1, N):
                if t == 0 and u == v:
                    qubo.add_term(G[0, u], u - 1)
                if t == N - 3:
                    qubo.add_term(G[u, v] + G[v, 0], t * (N - 1) + u - 1, (t + 1) * (N - 1) + v - 1)
                else:
                    qubo.add_term(G[u, v], t * (N - 1) + u - 1, (t + 1) * (N - 1) + v - 1)
    return qubo

def perturb(g):
    """Data Perturbation
    """
    n = g.shape[0]
    a = np.full((n, n), -1.0 / n) + np.identity(n)
    tmp_sum = np.sum(g) / float(n)
    v = np.array([np.sum(g[i, :] + g[:, i]) for i in range(n)])
    v = tmp_sum - v * 0.5
    v = v / (n - 2.0)
    delta = np.linalg.lstsq(a, v, rcond=-1)[0]
    new_g = np.copy(g)
    min_num = float("inf")
    for (i, j) in list(itertools.product(range(n), range(n))):
        if i != j:
            new_g[i, j] += delta[i] + delta[j]
            min_num = min(min_num, new_g[i, j])
    for (i, j) in list(itertools.product(range(n), range(n))):
        if i != j:
            new_g[i, j] -= min_num
    return new_g


def build_QUBO(N, A, rules):
    """Get QUBO form
    """
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

# One Less City
def build_QUBO_one_less(N, A, rules):
    """Get QUBO form
    """
    cons = QPoly((N - 1)**2)
    cons.sum(rules["column_one_less"])
    if len(A) > 1:
        cons.multiply_by_factor(A[0] / A[1])
        cons.sum(rules["row_one_less"])
        cons.multiply_by_factor(A[1])
    else:
        cons.sum(rules["row_one_less"])
        cons.multiply_by_factor(A[0])
    obj = rules["edge_one_less"]
    qubo = QPoly((N - 1)**2)
    qubo.sum(cons)
    qubo.sum(obj)
    return qubo

# ALM
def build_QUBO_lambda(N, A, lambda_param, rules):
    """Get QUBO form
    """
    def _update_linear(qubo, N, lambda_param):
        board = np.arange(N * N).reshape(N, N)
        for u in range(N):
            for v in range(N):
                qubo.add_term(lambda_param[u] + lambda_param[v+N], board[u][v])
            qubo.add_constant_term(-sum(lambda_param))
        return qubo
    
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
    qubo = _update_linear(qubo, N, lambda_param)
    return qubo


def getEnergy(QPoly, conf):
    """Gets energy of QUBO mapped configuration
    """
    E = 0
    for rowIndx in range(conf.shape[0]):
        if conf[rowIndx]:
            E += QPoly.array[int(rowIndx), int(rowIndx)]  # X**2
            for colIndx in range(int(rowIndx) + 1, conf.shape[0]):
                if conf[colIndx]:
                    E += QPoly.array[int(rowIndx), colIndx]  # XY
    return E + QPoly.constant


def compute_perm_makespan(N, conf, seq):
    """Compute Path if feasible
    """

    def check_valid_cycle(perm_vector):
        return (not (None in perm_vector)) and (
            len(set(perm_vector)) == len(perm_vector)
        )

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
        FSP_perm_reordered = FSP_perm[idx + 1 :] + FSP_perm[0 : idx + 1]
        seq.reorder(FSP_perm_reordered)
        perm_makespan_reordered = seq._makespan

        if perm_makespan_reordered <= perm_makespan:
            return FSP_perm_reordered, perm_makespan_reordered
        else:
            return FSP_perm, perm_makespan
    return [], -1


def compute_path_distance(N, conf, G):
    """Compute Path if feasible
    """

    def check_valid_cycle(path_vector):
        return (not (None in path_vector)) and (
            len(set(path_vector)) == len(path_vector)
        )

    TSP_path = [None for _ in range(N)]
#     board = np.arange(N * N).reshape(N, N)
#     for i in range(N):
#         for j in range(N):
#             if conf[board[i][j]]:
#                 TSP_path[j] = i
    conf = conf[:N ** 2].reshape(N, N)
    idx = np.asarray(np.where(conf == 1))
    for col in range(idx.shape[1]):
        i, j = idx[:, col]
        TSP_path[j] = i
    # evaluate the path cost
    if check_valid_cycle(TSP_path):
        path_cost = sum([G[TSP_path[i], TSP_path[(i + 1) % N]] for i in range(N)])
        return TSP_path, path_cost
    return [], -1

# One Less City
def compute_path_distance_one_less(N, conf, G):
    """Compute Path if feasible
    """

    def check_valid_cycle(path_vector):
        return (not (None in path_vector)) and (
            len(set(path_vector)) == len(path_vector)
        )

    board = np.arange((N - 1)**2).reshape(N - 1, N - 1)
    TSP_path = [None for _ in range(N - 1)]
    for i in range(N - 1):
        for j in range(N - 1):
            if conf[board[i][j]]:
                TSP_path[j] = i + 1
    TSP_path.insert(0, 0)
    # evaluate the path cost
    if check_valid_cycle(TSP_path):
        path_cost = sum([G[TSP_path[i], TSP_path[(i + 1) % N]] for i in range(N)])
        return TSP_path, path_cost
    return [], -1


def extract_tsp_features(N, conf, G):
    # given the tsp solution (could be infeasible)
    # return (max_edge, min_edge, avg_edge, std_edge, path_tour)
    board = np.arange(N * N).reshape(N, N)
    TSP_path = [None for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if conf[board[i][j]]:
                TSP_path[j] = i
                
    TSP_path = list(filter(lambda x: x is not None, TSP_path))
    N = len(TSP_path)
    edges = [G[TSP_path[i], TSP_path[(i + 1) % N]] for i in range(N)]
    return np.max(edges), np.min(edges), np.mean(edges), np.std(edges), np.sum(edges)


### QAP
def build_qap_row_sum(qap):
    N = qap.N
    qubo = QPoly(N * N)
    for i in range(N):
        tmp = QPoly(N * N)
        for j in range(N):
            # tmp.add_term(1, qap.board[(i, j)])
            tmp.add_term(1, i*N+j)
        tmp.add_constant_term(-1)
        tmp.power(2)
        qubo.sum(tmp)
    return qubo

def build_qap_col_sum(qap):
    N = qap.N
    qubo = QPoly(N * N)
    for j in range(N):
        tmp = QPoly(N * N)
        for i in range(N):
            # tmp.add_term(1, qap.board[(i, j)])
            tmp.add_term(1, i*N+j)
        tmp.add_constant_term(-1)
        tmp.power(2)
        qubo.sum(tmp)
    return qubo

def build_qap_obj(qap):
    N = qap.N
    qubo = QPoly(N * N)
    for (i, j, k, l) in itertools.product(range(N), range(N), range(N), range(N)):
        if qap.perturbed:
            # qubo.add_term(
            #     qap.perturbed_product_term[(i, j, k, l)],
            #     qap.board[(i, k)],
            #     qap.board[(j, l)],
            # )
            qubo.add_term(qap.perturbed_product_term[i*N+j, k*N+l], i*N+k, j*N+l)
        else:
            # qubo.add_term(
            #     qap.product_term[(i, j, k, l)], qap.board[(i, k)], qap.board[(j, l)]
            # )
            qubo.add_term(qap.product_term[i*N+j, k*N+l], i*N+k, j*N+l)
    return qubo

def build_QAP(qap, A, rules):
    """Get QUBO form
    """
    N = qap.N
    cons1 = QPoly(N * N)
    if type(rules["column"]) == np.ndarray:
        cons1.array = rules["column"]
    else:
        cons1.sum(rules["column"])
    cons1.multiply_by_factor(A[0])
    cons2 = QPoly(N * N)
    if type(rules["row"]) == np.ndarray:
        cons2.array = rules["row"]
    else:
        cons2.sum(rules["row"])
    if len(A) > 1:
        cons2.multiply_by_factor(A[1])
    else:
        cons2.multiply_by_factor(A[0])
    qubo = QPoly(N * N)
    if type(rules["obj"]) == np.ndarray:
        qubo.array = rules["obj"]
    else:
        qubo.sum(rules["obj"])
    qubo.sum(cons1)
    qubo.sum(cons2)
    return qubo

def compute_qap_cost(qap, conf):
    """Compute cost if feasible
    """
    def _check_infeasible(N, conf):
        if isinstance(conf, np.ndarray):
            x = conf[:N**2].reshape(N, N)
        else:
            # for qbsolv
            x = np.array(list(conf.values())).reshape(N, N)
        return np.array_equal(np.sum(x, axis=0), np.ones(N)) and np.array_equal(np.sum(x, axis=1), np.ones(N))
    
    N = qap.N
    y = {}
    for i in range(N**2):
        if conf[i]:
            y[divmod(i, N)] = 1
                                                               
    if not _check_infeasible(N, conf):
        return None, -1                                                                
                                                                             
    # feasible
    cost = 0
    for (i, j, k, l) in itertools.product(range(N), range(N), range(N), range(N)):
        cost += qap.product_term[i*N+j, k*N+l] * y.get((i, k), 0) * y.get((j, l), 0)
    print('Cost: {}'.format(cost))
    return y, cost