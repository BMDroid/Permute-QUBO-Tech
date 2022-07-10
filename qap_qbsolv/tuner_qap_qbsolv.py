import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

import bisect
import joblib
import math
import networkx as nx
import numpy as np
import optuna
import pandas as pd
import queue
import random
import time

# import tensorflow as tf
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from bisect import bisect
from collections import defaultdict
from heapq import heappush, nsmallest
from hyperopt import fmin, tpe, hp, partial, STATUS_OK, space_eval, Trials
from numpy import linalg as LA
from pathlib import Path
from sklearn.cluster import KMeans, SpectralClustering
from threading import Thread
from tqdm import tqdm_notebook as tqdm
from solver_qap_qbsolv import *

def get_deg_sequence(n, topo):
    if topo == "Ring":
        return [2] * n
    if topo == "Square":
        return [4] * n
    return []  # All


def create_graph(n, deg_seq):
    assert len(deg_seq) == n
    flag = False
    while not flag:
        g = nx.configuration_model(deg_seq)
        g = nx.Graph(g)  # remove parallel edges
        g.remove_edges_from(nx.selfloop_edges(g))  # remove selfloop
        actual_degrees = [d for v, d in g.degree()]
        flag = (deg_seq == actual_degrees) and (nx.number_connected_components(g) == 1)
    return g


def get_neighbor(adj, U=True):
    neighbor = defaultdict(list)
    for i in range(adj.shape[0]):
        index = np.where(adj[i] == 1)[-1].tolist()
        if U:
            neighbor[i] = index
        else:
            index.append(i)
            neighbor[i] = index
    return neighbor


def get_min_pos(pos_arr, val_arr, neighbor):
    min_pos_ = pos_arr.copy()
    for idx, index in neighbor.items():
        pos_arr_idx = pos_arr[index]
        val_arr_idx = val_arr[index]
        min_pos_[idx] = pos_arr_idx[np.argmin(val_arr_idx)]
    return min_pos_


# operate after replace the inf
def get_min_conf(conf_arr, val_arr, neighbor):
    min_conf_ = conf_arr.copy()
    for idx, index in neighbor.items():
        conf_arr_idx = [conf_arr[i] for i in index]
        val_arr_idx = val_arr[index]
        min_conf_[idx] = conf_arr_idx[np.argmin(val_arr_idx)]
    return min_conf_


def get_min_conf_all(conf_arr, val_arr):
    min_conf = conf_arr[np.argmin(val_arr)]
    return [min_conf] * len(conf_arr)


def is_tabu(clustering, clustering_means, clustering_var, cluster_order, pos_arr):
    y_pred = clustering.fit_predict(pos_arr)
    tabu_pos_arr = (y_pred == any(cluster_order[1:])).astype(
        int
    )  # only the pos in best cluster will be evaluated
    return y_pred, tabu_pos_arr


def get_clustering(pos_arr_his, val_arr_his, n_clusters=6):
    clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(pos_arr_his)
    y_pred = clustering.fit_predict(pos_arr_his)
    val_arr_his_copy = np.nan_to_num(val_arr_his.copy())
    clustering_means = np.zeros(n_clusters)
    clustering_var = np.zeros(n_clusters)
    for c in range(n_clusters):
        temp_val_arr = val_arr_his_copy[np.where(y_pred == c)]
        temp_val_arr = temp_val_arr[temp_val_arr != 0]
        if len(temp_val_arr) > 0:
            clustering_means[c] = temp_val_arr.mean()
            clustering_var[c] = np.var(temp_val_arr)
        else:
            clustering_means[c] = np.inf
            clustering_var[c] = 0
    return clustering, clustering_means, clustering_var, clustering_means.argsort()


def get_k_nearest_neighbor(pos, pos_arr_his, val_arr_his, k=3):
    heap = []
    for i, his_pos in enumerate(pos_arr_his):
        dis = np.linalg.norm(pos - his_pos)
        heappush(heap, (dis, val_arr_his[i]))
    k_nearest_neighbors = nsmallest(k, heap)
    k_nearest_neighbors_dis = [item[1] for item in k_nearest_neighbors]
    if k_nearest_neighbors_dis.count(float("inf")) >= k / 2:
        return float("inf")
    return sum(list(filter(lambda a: a != float("inf"), k_nearest_neighbors_dis))) / (
        k - k_nearest_neighbors_dis.count(float("inf"))
    )


def select_inheritance_p(val_arr, p=0.5):
    # select particles from the current epoch to conduct inheritance
    # pr = softmax(x / math.floor(np.log10(np.mean(x))))
    val_arr_copy = val_arr.copy()
    val_arr_copy = val_arr_copy.reshape(val_arr_copy.shape[0],)
    pr = val_arr_copy / np.sum(val_arr_copy)
    pr_order = np.argsort(pr)
    sorted_pr = sorted(pr)
    cum_pr = np.cumsum(sorted_pr)
    # random selection probability
    # p = np.random.uniform(5/len(x), 1)
    #     idx = bisect.bisect(cum_pr, p)
    idx = bisect(cum_pr, p)
    # return the indices of selected particles for objective inheritance
    inheritance_indices = pr_order[idx:]
    return inheritance_indices


def select_inheritance(val_arr, p=0.5):
    val_arr_copy = val_arr.copy()
    val_arr_copy = val_arr_copy.reshape(val_arr.shape[0],)
    val_order = np.argsort(val_arr_copy)
    # return the indices of selected particles for objective inheritance
    inheritance_indices = val_order[int(p * val_arr_copy.shape[0]) :]
    return inheritance_indices


def neighbors_weight(pos, pos_history, k=5):
    # calculate the distance between the particle's current position with all history positions
    dis = LA.norm(pos - pos_history, axis=1)
    epsilon = 10 ** -7
    dis[dis == 0] = epsilon
    dis_order = np.argsort(dis)
    # calculate the weight for each historical positions given the distance calculated
    weight = np.zeros((len(pos_history), 1))
    # sum the weight for those selected positions
    dis_sum = np.sum(1 / dis[dis_order[:k]])
    for i in range(weight.shape[0]):
        if i in dis_order[:k]:
            weight[i] = (1 / dis[i]) / dis_sum
    return weight


def weighted_inheritance(pos, pos_his, val_his, k=5):
    weight = neighbors_weight(pos, pos_his, k)
    return np.sum(val_his.reshape(weight.shape) * weight)


def pso_inheritance_mod(fileName, inst, rules, bounds, parameters, solve, prep_config):
    ### evaluate the particles' positions using DA
    def _obj_thr(solve, prep_config, pos_arr):
        nonlocal fileName
        nonlocal inst
        nonlocal rules
        nonlocal bounds_copy
        nonlocal parameters
        nonlocal num_penalties
        nonlocal lbd
        nonlocal lbd_flag
        nonlocal lbd_coef

        temp_qubo = list(rules.values())[0]
        N_states = [1024, 2048, 4096, 8192]
        N_state = N_states[bisect(N_states, temp_qubo._size)]
        initial_conf_arr = [None] * len(pos_arr)
        #         initial_conf_arr = [np.zeros(N_state, dtype=int)] * len(pos_arr)
        #         random_conf = np.random.randint(2, size=N_state)
        #         initial_conf_arr = [random_conf] * len(pos_arr)
        thrs = []
        que = queue.Queue()
        for idx, pos in enumerate(pos_arr):
            thr = Thread(
                target=lambda q, arg: q.put((idx, *solve(*arg))),
                args=(
                    que,
                    (
                        inst,
                        rules,
                        pos[:num_penalties],
                        int(pos[-2]),
                        int(pos[-1]),
                        prep_config,
                        initial_conf_arr[idx],
                    ),
                ),
            )
            thrs.append(thr)
            thr.start()
        for thr in thrs:
            thr.join()
        val_arr = [None] * len(pos_arr)
        obj_arr = [None] * len(pos_arr)
        sol_arr = [None] * len(pos_arr)
        conf_arr = [None] * len(pos_arr)
        while not que.empty():
            idx, pos, min_sol, min_obj, avg_obj, min_conf, elapsed_time = que.get()
            conf_arr[idx] = min_conf
            sol_arr[idx] = min_sol
            if min_obj == float("inf") or avg_obj == float("inf"):
                val_arr[idx] = float("inf")
            else:
                if not lbd_flag:
                    lbd = min(
                        lbd,
                        -int(
                            math.log(
                                (1 - parameters["mixed"]) * min_obj
                                + parameters["mixed"] * avg_obj,
                                10,
                            )
                        ),
                    )
                    lbd_flag = 1
                val = (
                    (1 - parameters["mixed"]) * min_obj
                    + parameters["mixed"] * avg_obj
                    + parameters["reg"] * 10 ** lbd * LA.norm(lbd_coef * pos) ** 2
                )
                val_arr[idx] = val
            obj_arr[idx] = min_obj
            save_solution(
                fileName,
                -1,
                pos,
                min_sol,
                min_obj,
                val_arr[idx],
                elapsed_time,
                parameters,
                bounds_copy,
            )
        return (
            np.array(val_arr).reshape(len(pos_arr), 1).astype(float),
            np.array(obj_arr).reshape(len(pos_arr), 1),
            sol_arr,
            conf_arr,
        )

    ### evaluate using either the original objective function or objective inheritance
    def _obj_inheritance(epoch, solve, prep_config, pos_arr, val_arr, p=0.5, k=5, l=50):
        nonlocal fileName
        nonlocal parameters
        nonlocal inst
        nonlocal rules
        nonlocal bounds_copy
        nonlocal parameters
        nonlocal pos_his
        nonlocal val_his
        nonlocal obj_his
        nonlocal sol_his
        nonlocal num_penalties
        nonlocal lbd
        nonlocal lbd_flag
        nonlocal lbd_coef
        nonlocal min_conf_arr

        val_arr_temp = [None] * len(pos_arr)
        val_arr_temp = np.array(val_arr_temp).reshape(val_arr.shape).astype(float)
        obj_arr_temp = [None] * len(pos_arr)
        obj_arr_temp = np.array(obj_arr_temp).reshape(val_arr.shape).astype(float)
        conf_arr_temp = [None] * len(pos_arr)
        sol_arr = [None] * len(pos_arr)
        inheritance_idxes = np.array(select_inheritance(val_arr, p))
        thrs = []
        que = queue.Queue()
        for idx, pos in enumerate(pos_arr):
            if idx not in inheritance_idxes:
                thr = Thread(
                    target=lambda q, arg: q.put((idx, *solve(*arg))),
                    args=(
                        que,
                        (
                            inst,
                            rules,
                            pos[:num_penalties],
                            int(pos[-2]),
                            int(pos[-1]),
                            prep_config,
                            min_conf_arr[idx],
                        ),
                    ),
                )
                thrs.append(thr)
                thr.start()
            else:
                # val_arr_temp[idx] = weighted_inheritance(
                #     pos, pos_his[-l:], val_his[-l:], k)
                obj_arr_temp[idx] = weighted_inheritance(
                    pos, pos_his[-l:], obj_his[-l:], k
                )
                conf_arr_temp[idx] = min_conf_arr[idx]
        for thr in thrs:
            thr.join()
        while not que.empty():
            idx, pos, min_sol, min_obj, avg_obj, min_conf, elapsed_time = que.get()
            conf_arr_temp[idx] = min_conf
            sol_arr[idx] = min_sol
            if min_obj == float("inf") or avg_obj == float("inf"):
                val_arr_temp[idx] = float("inf")
            else:
                if not lbd_flag:
                    lbd = min(
                        lbd,
                        -int(
                            math.log(
                                (1 - parameters["mixed"]) * min_obj
                                + parameters["mixed"] * avg_obj,
                                10,
                            )
                        ),
                    )
                    lbd_flag = 1
                val = (
                    (1 - parameters["mixed"]) * min_obj
                    + parameters["mixed"] * avg_obj
                    + parameters["reg"] * 10 ** lbd * LA.norm(lbd_coef * pos) ** 2
                )
                val_arr_temp[idx] = val
            obj_arr_temp[idx] = min_obj
            save_solution(
                fileName,
                epoch,
                pos,
                min_sol,
                min_obj,
                val_arr_temp[idx][0],
                elapsed_time,
                parameters,
                bounds_copy,
            )

        for idx, pos in enumerate(pos_arr):
            if idx in inheritance_idxes:
                val_arr_temp[idx] = (
                    obj_arr_temp[idx]
                    + parameters["reg"] * 10 ** lbd * LA.norm(lbd_coef * pos) ** 2
                )
        if len(inheritance_idxes) != 0:
            pos_his = np.concatenate((pos_his, pos_arr[~inheritance_idxes]), axis=0)
            val_his = np.concatenate(
                (val_his, val_arr_temp[~inheritance_idxes]), axis=0
            )
            obj_his = np.concatenate(
                (obj_his, obj_arr_temp[~inheritance_idxes]), axis=0
            )
        else:
            pos_his = np.concatenate((pos_his, pos_arr), axis=0)
            val_his = np.concatenate((val_his, val_arr_temp), axis=0)
            obj_his = np.concatenate((obj_his, obj_arr_temp), axis=0)
        #         sol_his = sol_his + [sol for sol in sol_arr if sol]
        sol_his = sol_his + sol_arr
        return val_arr_temp, obj_arr_temp, conf_arr_temp

    # bounds contain the penalty terms and DA settings
    bounds_copy = bounds.copy()
    num_penalties = 0
    for k, v in sorted(bounds_copy.items()):
        if len(k) == 2 and k[-1].isalpha():
            bounds_copy[k] = (
                v[0] * getattr(inst, parameters["base"]),
                v[1] * getattr(inst, parameters["base"]),
            )
            num_penalties += 1
    # Start PSO
    print("\n{}".format(Path(fileName).stem))
    print("PSO Initialization\n")

    # create topology of the particles
    deg_sequence = get_deg_sequence(parameters["particles"], parameters["topo"])
    if len(deg_sequence) != 0:
        g = create_graph(parameters["particles"], deg_sequence)
        adj = nx.to_numpy_matrix(g)
    else:
        adj = np.ones((parameters["particles"], parameters["particles"]))
        np.fill_diagonal(adj, 0)
    neighbor = get_neighbor(adj, U=parameters["U"])
    # initialize the particles' positions and velocities
    if not parameters["ini"]:
        pos_arr = np.concatenate(
            [
                np.random.uniform(v[0], v[1], parameters["particles"],).reshape(
                    parameters["particles"], 1
                )
                for k, v in sorted(bounds_copy.items())
            ],
            axis=1,
        ).astype(float)
    else:
        pos_arr = np.concatenate(
            [
                np.linspace(v[0], v[1], parameters["particles"],).reshape(
                    parameters["particles"], 1
                )
                for k, v in sorted(bounds_copy.items())
            ],
            axis=1,
        ).astype(float)
    vel_arr = np.concatenate(
        [
            np.random.uniform(
                v[0] - v[1], v[1] - v[0], parameters["particles"],
            ).reshape(parameters["particles"], 1)
            for k, v in sorted(bounds_copy.items())
        ],
        axis=1,
    ).astype(float)
    # set up the regularization term
    lbd = -3
    lbd_flag = 0
    lbd_coef = parameters["coef"]
    # personal best
    personal_best = pos_arr.copy()
    initialization_start = time.time()
    # evaluate the fitness value and objective value (real solution) of the initialized particles
    val_arr, obj_arr, sol_arr, conf_arr = _obj_thr(solve, prep_config, pos_arr)
    #     val_arr[np.isinf(val_arr)] = 10 ** - (3 * lbd)
    #     obj_arr[np.isinf(obj_arr)] = 10 ** - (3 * lbd)
    val_arr[np.isinf(val_arr)] = 10 ** -(lbd - 1)
    obj_arr[np.isinf(obj_arr)] = 10 ** -(lbd - 1)
    # TODO: best conf
    min_conf_arr = conf_arr.copy()
    min_conf_arr = get_min_conf(conf_arr, val_arr, neighbor)
    #     min_conf_arr = get_min_conf_all(conf_arr, val_arr)
    # global best
    global_best = personal_best[np.argmin(val_arr)]
    # update the best fitness value so far
    cur_val_min = min(np.min(val_arr), float("inf"))
    # update the best objective vale so far
    cur_obj_min = min(np.min(obj_arr), float("inf"))
    # update the best solution so far
    cur_sol_min = sol_arr[np.argmin(cur_obj_min)]
    # record the particles position and fitness value
    pos_his = pos_arr.copy()
    val_his = val_arr.copy()
    obj_his = obj_arr.copy()
    sol_his = sol_arr.copy()  # list
    print(f"\nBest solution found: {cur_obj_min}")
    print(f"Best mixed solution found: {cur_val_min}")
    for i, k in enumerate(sorted(bounds_copy.keys())):
        print(f"Best {k[1:]} found: {global_best[i]}")
    print("\n")
    # record the total runtime
    total_elapsed = time.time() - initialization_start
    ### start the optimization
    for e in range(parameters["epoch"]):
        epoch_start = time.time()
        # calculate the velocities
        vel_arr = (
            round(
                parameters["w"]
                + (parameters["w0"] - parameters["w"])
                * (1 - (e + 1) / parameters["epoch"]),
                2,
            )
            * vel_arr
            + round(
                parameters["c1"]
                + (parameters["c10"] - parameters["c1"])
                * (1 - (e + 1) / parameters["epoch"]),
                2,
            )
            * np.random.uniform(0, 1, len(bounds_copy))
            * (personal_best - pos_arr)
            + round(
                parameters["c2"]
                + (parameters["c20"] - parameters["c2"])
                * (1 - (e + 1) / parameters["epoch"]),
                2,
            )
            * np.random.uniform(0, 1, len(bounds_copy))
            * (get_min_pos(personal_best, val_arr, neighbor) - pos_arr)
        )
        # update the particles' positions
        pos_arr = np.concatenate(
            [
                np.array(
                    [
                        min(
                            bounds_copy[k][1],
                            max(bounds_copy[k][0], pos_arr[i][j] + vel_arr[i][j],),
                        )
                        for i in range(parameters["particles"])
                    ]
                ).reshape(parameters["particles"], 1)
                for j, k in enumerate(sorted(bounds_copy.keys()))
            ],
            axis=1,
        ).astype(float)
        # calculate the fitness value of the particles
        val_arr_temp, obj_arr_temp, conf_arr_temp = _obj_inheritance(
            e, solve, prep_config, pos_arr, val_arr
        )
        val_arr_temp_ = val_arr_temp[~np.isnan(val_arr_temp)]  # pd.idnull()
        val_arr_temp_ = val_arr_temp_[~np.isinf(val_arr_temp_)]
        nan_to_max = np.max(val_arr_temp_)
        val_arr_temp = np.nan_to_num(val_arr_temp, copy=True, nan=nan_to_max)
        #         val_arr_temp[np.isinf(val_arr_temp)] = 10 ** - (3 * lbd)
        #         obj_arr_temp[np.isinf(obj_arr_temp)] = 10 ** - (3 * lbd)
        #         val_his[np.isinf(val_his)] = 10 ** - (3 * lbd)
        #         obj_his[np.isinf(obj_his)] = 10 ** - (3 * lbd)
        val_arr_temp[np.isinf(val_arr_temp)] = 10 ** -(lbd - 1)
        obj_arr_temp[np.isinf(obj_arr_temp)] = 10 ** -(lbd - 1)
        val_his[np.isinf(val_his)] = 10 ** -(lbd - 1)
        obj_his[np.isinf(obj_his)] = 10 ** -(lbd - 1)
        # create a mask
        mask = np.array(val_arr_temp < val_arr).reshape(parameters["particles"], 1)
        # update the value
        val_arr = val_arr_temp * mask + val_arr * (1 - mask)
        # update the personal_best
        personal_best = pos_arr * mask + personal_best * (1 - mask)
        # TODO: update particles' conf if the solution improved
        conf_arr = [
            conf_arr_temp[i] if mask[i][0] else conf_arr[i]
            for i in range(parameters["particles"])
        ]
        # TODO: best conf
        #         min_conf_arr = conf_arr.copy()
        min_conf_arr = get_min_conf(conf_arr, val_arr, neighbor)
        #         min_conf_arr = get_min_conf_all(conf_arr, val_arr)
        # update the global_best
        global_best = personal_best[np.argmin(val_arr)]
        # update the best fitness value
        cur_val_min = min(np.min(val_his), cur_val_min)
        # update the best objective value
        cur_obj_min = min(np.min(obj_his), cur_obj_min)
        # elapsed time of the epoch
        epoch_elapsed = time.time() - epoch_start
        # print the best solution found so far
        print(f"\nBest solution found: {cur_obj_min}")
        print(f"Best mixed solution found: {cur_val_min}")
        for i, k in enumerate(sorted(bounds_copy.keys())):
            print(f"Best {k[1:]} found: {global_best[i]}")
        print("\n")
    total_elapsed += epoch_elapsed
    return (
        sol_his[np.argmin(obj_his)],
        cur_obj_min,
        cur_val_min,
        pos_his[np.argmin(obj_his)],
        0,
        total_elapsed,
    )


def pso_inheritance(fileName, inst, rules, bounds, parameters, solve, prep_config):
    ### evaluate the particles' positions using DA
    def _obj_thr(solve, prep_config, pos_arr):
        nonlocal fileName
        nonlocal inst
        nonlocal rules
        nonlocal bounds_copy
        nonlocal parameters
        nonlocal num_penalties
        nonlocal lbd
        nonlocal lbd_flag
        nonlocal lbd_coef

        thrs = []
        que = queue.Queue()
        for idx, pos in enumerate(pos_arr):
            thr = Thread(
                target=lambda q, arg: q.put((idx, *solve(*arg))),
                args=(
                    que,
                    (inst, rules, pos[:num_penalties], int(pos[-2]), int(pos[-1]), prep_config),
                ),
            )
            thrs.append(thr)
            thr.start()
        for thr in thrs:
            thr.join()
        val_arr = [None] * len(pos_arr)
        obj_arr = [None] * len(pos_arr)
        sol_arr = [None] * len(pos_arr)
        conf_arr = [None] * len(pos_arr)
        while not que.empty():
            idx, pos, min_sol, min_obj, avg_obj, min_conf, elapsed_time = que.get()
            conf_arr[idx] = min_conf
            sol_arr[idx] = min_sol
            if min_obj == float("inf") or avg_obj == float("inf"):
                val_arr[idx] = float("inf")
            else:
                if not lbd_flag:
                    lbd = min(
                        lbd,
                        -int(
                            math.log(
                                (1 - parameters["mixed"]) * min_obj
                                + parameters["mixed"] * avg_obj,
                                10,
                            )
                        ),
                    )
                    lbd_flag = 1
                val = (
                    (1 - parameters["mixed"]) * min_obj
                    + parameters["mixed"] * avg_obj
                    + parameters["reg"] * 10 ** lbd * LA.norm(lbd_coef * pos) ** 2
                )
                val_arr[idx] = val
            obj_arr[idx] = min_obj
            save_solution(
                fileName,
                -1,
                pos,
                min_sol,
                min_obj,
                val_arr[idx],
                elapsed_time,
                parameters,
                bounds_copy,
            )
        return (
            np.array(val_arr).reshape(len(pos_arr), 1).astype(float),
            np.array(obj_arr).reshape(len(pos_arr), 1),
            sol_arr,
            conf_arr,
        )

    ### evaluate using either the original objective function or objective inheritance
    def _obj_inheritance(epoch, solve, prep_config, pos_arr, val_arr, p=0.5, k=5, l=50):
        nonlocal fileName
        nonlocal parameters
        nonlocal inst
        nonlocal rules
        nonlocal bounds_copy
        nonlocal parameters
        nonlocal pos_his
        nonlocal val_his
        nonlocal obj_his
        nonlocal sol_his
        nonlocal num_penalties
        nonlocal lbd
        nonlocal lbd_flag
        nonlocal lbd_coef
        nonlocal conf_arr

        val_arr_temp = [None] * len(pos_arr)
        val_arr_temp = np.array(val_arr_temp).reshape(val_arr.shape).astype(float)
        obj_arr_temp = [None] * len(pos_arr)
        obj_arr_temp = np.array(obj_arr_temp).reshape(val_arr.shape).astype(float)
        sol_arr = [None] * len(pos_arr)
        inheritance_idxes = np.array(select_inheritance(val_arr, p))
        thrs = []
        que = queue.Queue()
        for idx, pos in enumerate(pos_arr):
            if idx not in inheritance_idxes:
                thr = Thread(
                    target=lambda q, arg: q.put((idx, *solve(*arg))),
                    args=(
                        que,
                        (
                            inst,
                            rules,
                            pos[:num_penalties],
                            int(pos[-2]),
                            int(pos[-1]),
                            prep_config,
                            conf_arr[idx],
                        ),
                    ),
                )
                thrs.append(thr)
                thr.start()
            else:
                val_arr_temp[idx] = weighted_inheritance(
                    pos, pos_his[-l:], val_his[-l:], k
                )
                obj_arr_temp[idx] = weighted_inheritance(
                    pos, pos_his[-l:], obj_his[-l:], k
                )
        for thr in thrs:
            thr.join()
        while not que.empty():
            idx, pos, min_sol, min_obj, avg_obj, min_conf, elapsed_time = que.get()
            conf_arr[idx] = min_conf
            sol_arr[idx] = min_sol
            if min_obj == float("inf") or avg_obj == float("inf"):
                val_arr_temp[idx] = float("inf")
            else:
                if not lbd_flag:
                    lbd = min(
                        lbd,
                        -int(
                            math.log(
                                (1 - parameters["mixed"]) * min_obj
                                + parameters["mixed"] * avg_obj,
                                10,
                            )
                        ),
                    )
                    lbd_flag = 1
                val = (
                    (1 - parameters["mixed"]) * min_obj
                    + parameters["mixed"] * avg_obj
                    + parameters["reg"] * 10 ** lbd * LA.norm(lbd_coef * pos) ** 2
                )
                val_arr_temp[idx] = val
            obj_arr_temp[idx] = min_obj
            save_solution(
                fileName,
                epoch,
                pos,
                min_sol,
                min_obj,
                val_arr_temp[idx][0],
                elapsed_time,
                parameters,
                bounds_copy,
            )
        if len(inheritance_idxes) != 0:
            pos_his = np.concatenate((pos_his, pos_arr[~inheritance_idxes]), axis=0)
            val_his = np.concatenate(
                (val_his, val_arr_temp[~inheritance_idxes]), axis=0
            )
            obj_his = np.concatenate(
                (obj_his, obj_arr_temp[~inheritance_idxes]), axis=0
            )
        else:
            pos_his = np.concatenate((pos_his, pos_arr), axis=0)
            val_his = np.concatenate((val_his, val_arr_temp), axis=0)
            obj_his = np.concatenate((obj_his, obj_arr_temp), axis=0)
        #         sol_his = sol_his + [sol for sol in sol_arr if sol]
        sol_his = sol_his + sol_arr
        return val_arr_temp, obj_arr_temp

    # bounds contain the penalty terms and DA settings
    bounds_copy = bounds.copy()
    num_penalties = 0
    for k, v in sorted(bounds_copy.items()):
        if len(k) == 2 and k[-1].isalpha():
            bounds_copy[k] = (
                v[0] * getattr(inst, parameters["base"]),
                v[1] * getattr(inst, parameters["base"]),
            )
            num_penalties += 1
    # Start PSO
    print("\n{}".format(Path(fileName).stem))
    print("PSO Initialization\n")

    # create topology of the particles
    deg_sequence = get_deg_sequence(parameters["particles"], parameters["topo"])
    if len(deg_sequence) != 0:
        g = create_graph(parameters["particles"], deg_sequence)
        adj = nx.to_numpy_matrix(g)
    else:
        adj = np.ones((parameters["particles"], parameters["particles"]))
        np.fill_diagonal(adj, 0)
    neighbor = get_neighbor(adj, U=parameters["U"])
    # initialize the particles' positions and velocities
    if not parameters["ini"]:
        pos_arr = np.concatenate(
            [
                np.random.uniform(v[0], v[1], parameters["particles"],).reshape(
                    parameters["particles"], 1
                )
                for k, v in sorted(bounds_copy.items())
            ],
            axis=1,
        ).astype(float)
    else:
        pos_arr = np.concatenate(
            [
                np.linspace(v[0], v[1], parameters["particles"],).reshape(
                    parameters["particles"], 1
                )
                for k, v in sorted(bounds_copy.items())
            ],
            axis=1,
        ).astype(float)
    vel_arr = np.concatenate(
        [
            np.random.uniform(
                v[0] - v[1], v[1] - v[0], parameters["particles"],
            ).reshape(parameters["particles"], 1)
            for k, v in sorted(bounds_copy.items())
        ],
        axis=1,
    ).astype(float)
    # set up the regularization term
    lbd = -3
    lbd_flag = 0
    lbd_coef = parameters["coef"]
    # personal best
    personal_best = pos_arr.copy()
    initialization_start = time.time()
    # evaluate the fitness value and objective value (real solution) of the initialized particles
    val_arr, obj_arr, sol_arr, conf_arr = _obj_thr(solve, prep_config, pos_arr)
    val_arr[np.isinf(val_arr)] = 10 ** -(lbd - 1)
    obj_arr[np.isinf(obj_arr)] = 10 ** -(lbd - 1)
    # global best
    global_best = personal_best[np.argmin(val_arr)]
    # update the best fitness value so far
    cur_val_min = min(np.min(val_arr), float("inf"))
    # update the best objective vale so far
    cur_obj_min = min(np.min(obj_arr), float("inf"))
    # update the best solution so far
    cur_sol_min = sol_arr[np.argmin(cur_obj_min)]
    # record the particles position and fitness value
    pos_his = pos_arr.copy()
    val_his = val_arr.copy()
    obj_his = obj_arr.copy()
    sol_his = sol_arr.copy()  # list
    print(f"\nBest solution found: {cur_obj_min}")
    print(f"Best mixed solution found: {cur_val_min}")
    for i, k in enumerate(sorted(bounds_copy.keys())):
        print(f"Best {k[1:]} found: {global_best[i]}")
    print("\n")
    # record the total runtime
    total_elapsed = time.time() - initialization_start
    ### start the optimization
    for e in range(parameters["epoch"]):
        epoch_start = time.time()
        # calculate the velocities
        vel_arr = (
            round(
                parameters["w"]
                + (parameters["w0"] - parameters["w"])
                * (1 - (e + 1) / parameters["epoch"]),
                2,
            )
            * vel_arr
            + round(
                parameters["c1"]
                + (parameters["c10"] - parameters["c1"])
                * (1 - (e + 1) / parameters["epoch"]),
                2,
            )
            * np.random.uniform(0, 1, len(bounds_copy))
            * (personal_best - pos_arr)
            + round(
                parameters["c2"]
                + (parameters["c20"] - parameters["c2"])
                * (1 - (e + 1) / parameters["epoch"]),
                2,
            )
            * np.random.uniform(0, 1, len(bounds_copy))
            * (get_min_pos(personal_best, val_arr, neighbor) - pos_arr)
        )
        # update the particles' positions
        pos_arr = np.concatenate(
            [
                np.array(
                    [
                        min(
                            bounds_copy[k][1],
                            max(bounds_copy[k][0], pos_arr[i][j] + vel_arr[i][j],),
                        )
                        for i in range(parameters["particles"])
                    ]
                ).reshape(parameters["particles"], 1)
                for j, k in enumerate(sorted(bounds_copy.keys()))
            ],
            axis=1,
        ).astype(float)
        # calculate the fitness value of the particles
        #         p = parameters["p"]
        #         k = parameters["k"]
        #         val_arr_temp = _obj_inheritance(obj, pos_arr, val_arr, p, k)
        val_arr_temp, obj_arr_temp = _obj_inheritance(e, solve, prep_config, pos_arr, val_arr)
        val_arr_temp_ = val_arr_temp[~np.isnan(val_arr_temp)]  # pd.idnull()
        val_arr_temp_ = val_arr_temp_[~np.isinf(val_arr_temp_)]
        nan_to_max = np.max(val_arr_temp_)
        val_arr_temp = np.nan_to_num(val_arr_temp, copy=True, nan=nan_to_max)
        val_arr_temp[np.isinf(val_arr_temp)] = 10 ** -(lbd - 1)
        obj_arr_temp[np.isinf(obj_arr_temp)] = 10 ** -(lbd - 1)
        val_his[np.isinf(val_his)] = 10 ** -(lbd - 1)
        obj_his[np.isinf(obj_his)] = 10 ** -(lbd - 1)
        #         print(val_arr_temp)
        # create a mask
        mask = np.array(val_arr_temp < val_arr).reshape(parameters["particles"], 1)
        # update the value
        val_arr = val_arr_temp * mask + val_arr * (1 - mask)
        #         print(val_arr)
        # update the personal_best
        personal_best = pos_arr * mask + personal_best * (1 - mask)
        # update the global_best
        global_best = personal_best[np.argmin(val_arr)]
        # update the best fitness value
        cur_val_min = min(np.min(val_his), cur_val_min)
        # update the best objective value
        cur_obj_min = min(np.min(obj_his), cur_obj_min)
        # elapsed time of the epoch
        epoch_elapsed = time.time() - epoch_start
        # print the best solution found so far
        print(f"\nBest solution found: {cur_obj_min}")
        print(f"Best mixed solution found: {cur_val_min}")
        for i, k in enumerate(sorted(bounds_copy.keys())):
            print(f"Best {k[1:]} found: {global_best[i]}")
        print("\n")
    total_elapsed += epoch_elapsed
    return (
        sol_his[np.argmin(obj_his)],
        cur_obj_min,
        cur_val_min,
        pos_his[np.argmin(obj_his)],
        0,
        total_elapsed,
    )


def pso_clustering(fileName, inst, rules, bounds, parameters, solve, prep_config):

    print("\n{}".format(Path(fileName).stem))
    print("PSO initialization start ...\n")
    total_elapsed = 0

    # create topology
    deg_sequence = get_deg_sequence(parameters["particles"], parameters["topo"])
    if len(deg_sequence) != 0:
        g = create_graph(parameters["particles"], deg_sequence)
        adj = nx.to_numpy_matrix(g)
    else:
        adj = np.ones((parameters["particles"], parameters["particles"]))
        np.fill_diagonal(adj, 0)
    neighbor = get_neighbor(adj, U=parameters["U"])

    # initialization
    if not parameters["ini"]:
        pos_arr = np.concatenate(
            [
                np.random.uniform(
                    bounds["min"][i] * getattr(inst, parameters["base"]),
                    bounds["max"][i] * getattr(inst, parameters["base"]),
                    parameters["particles"],
                ).reshape(parameters["particles"], 1)
                for i in range(len(bounds["min"]))
            ],
            axis=1,
        ).astype(float)
    else:
        pos_arr = np.concatenate(
            [
                np.linspace(
                    bounds["min"][i] * getattr(inst, parameters["base"]),
                    bounds["max"][i] * getattr(inst, parameters["base"]),
                    num=parameters["particles"],
                ).reshape(parameters["particles"], 1)
                for i in range(len(bounds["min"]))
            ],
            axis=1,
        ).astype(float)

    best_pos_arr = pos_arr.copy()
    vel_arr = np.concatenate(
        [
            np.random.uniform(
                bounds["min"][i] * getattr(inst, parameters["base"])
                - bounds["max"][i] * getattr(inst, parameters["base"]),
                bounds["max"][i] * getattr(inst, parameters["base"])
                - bounds["min"][i] * getattr(inst, parameters["base"]),
                parameters["particles"],
            ).reshape(parameters["particles"], 1)
            for i in range(len(bounds["min"]))
        ],
        axis=1,
    ).astype(float)
    val_arr = np.zeros((parameters["particles"])).astype(float)
    sol_arr = np.zeros((parameters["particles"])).astype(
        float
    )  # array for storing the minimum solution
    cur_mixed = float("inf")
    cur_cost = float("inf")
    cur_perm = []
    min_pos = [None] * len(bounds["min"])
    best_epoch = -1
    lbd = -3  # lambda for the regularization term
    lbd_coef = [1, 1.5]
    lbd_flag = 0

    # get initial values
    thrs = []
    que = queue.Queue()
    for idx, pos in enumerate(pos_arr):
        thr = Thread(
            target=lambda q, arg: q.put((idx, *solve(*arg))),
            args=(
                que,
                (
                    inst,
                    rules,
                    pos,
                    parameters["num_run"],
                    parameters["num_iteration"][0],
                    prep_config,
                ),
            ),
        )
        thrs.append(thr)
        thr.start()
    for thr in thrs:
        thr.join()
    epoch_elapsed = 0
    while not que.empty():
        idx, pos, min_perm, min_cost, avg_cost, elapsed_time = que.get()
        epoch_elapsed = max(elapsed_time, epoch_elapsed)
        if min_cost == float("inf") or avg_cost == float("inf"):
            val = float("inf")
            sol = float("inf")
        else:
            if not lbd_flag:
                lbd = min(
                    lbd,
                    -int(
                        math.log(
                            (1 - parameters["mixed"]) * min_cost
                            + parameters["mixed"] * avg_cost,
                            10,
                        )
                    ),
                )
                lbd_flag = 1
                print("lambda: {}".format(lbd))
            val = (
                (1 - parameters["mixed"]) * min_cost
                + parameters["mixed"] * avg_cost
                + parameters["reg"] * 10 ** lbd * LA.norm(lbd_coef * pos) ** 2
            )
            sol = min_cost
        # cur_cost = min(sol, cur_cost)
        if sol < cur_cost:
            cur_cost = sol
            cur_perm = min_perm
        save_solution(
            fileName, -1, pos, min_perm, min_cost, val, elapsed_time, parameters, bounds
        )
        if val < cur_mixed:
            min_pos = pos
            cur_mixed = val
        val_arr[idx] = val
        sol_arr[idx] = sol
    total_elapsed += epoch_elapsed
    save_epoch(
        fileName,
        -1,
        cur_cost,
        cur_cost,
        cur_mixed,
        min_pos,
        epoch_elapsed,
        parameters,
        bounds,
    )
    print("\nInitialization finished")
    print(f"Best solution found after initialization: {cur_cost}")
    print(f"Best mixed solution found after initialization: {cur_mixed}")
    for i in range(len(bounds["min"])):
        print(f"Best {chr(65 + i)} found after initialization: {min_pos[i]}")

    # record all the particles and their values
    pos_arr_his = pos_arr.copy()
    val_arr_his = val_arr.copy()
    sol_arr_his = sol_arr.copy()
    tabu_pos_arr = np.zeros((parameters["particles"],)).astype(float)
    y_pred = np.zeros((parameters["particles"],))
    clustering, clustering_means, clustering_var, cluster_order = None, None, None, None

    # start optimization
    for e in range(parameters["epoch"]):
        print(f"\nEpoch {e + 1} start:")
        vel_arr = (
            round(
                parameters["w"]
                + (parameters["w0"] - parameters["w"])
                * (1 - (e + 1) / parameters["epoch"]),
                2,
            )
            * vel_arr
            + round(
                parameters["c1"]
                + (parameters["c10"] - parameters["c1"])
                * (1 - (e + 1) / parameters["epoch"]),
                2,
            )
            * np.random.uniform(0, 1, len(bounds["min"]))
            * (best_pos_arr - pos_arr)
            + round(
                parameters["c2"]
                + (parameters["c20"] - parameters["c2"])
                * (1 - (e + 1) / parameters["epoch"]),
                2,
            )
            * np.random.uniform(0, 1, len(bounds["min"]))
            * (get_min_pos(pos_arr, val_arr, neighbor) - pos_arr)
        )
        pos_arr = np.concatenate(
            [
                np.array(
                    [
                        min(
                            bounds["max"][j] * getattr(inst, parameters["base"]),
                            max(
                                bounds["min"][j] * getattr(inst, parameters["base"]),
                                pos_arr[i][j] + vel_arr[i][j],
                            ),
                        )
                        for i in range(parameters["particles"])
                    ]
                ).reshape(parameters["particles"], 1)
                for j in range(len(bounds["min"]))
            ],
            axis=1,
        ).astype(float)
        if e > 0:
            y_pred, tabu_pos_arr = is_tabu(
                clustering, clustering_means, clustering_var, cluster_order, pos_arr
            )

        epoch_min = float("inf")
        thrs = []
        que = queue.Queue()
        for idx, pos in enumerate(pos_arr):
            if not tabu_pos_arr[idx]:
                thr = Thread(
                    target=lambda q, arg: q.put((idx, *solve(*arg))),
                    args=(
                        que,
                        (
                            inst,
                            rules,
                            pos,
                            parameters["num_run"],
                            parameters["num_iteration"][e + 1],
                            prep_config
                        ),
                    ),
                )
                thrs.append(thr)
                thr.start()
            else:
                sol = get_k_nearest_neighbor(pos, pos_arr_his, sol_arr_his)
                val = get_k_nearest_neighbor(
                    pos, pos_arr_his, val_arr_his
                )  # average of k nearest neighbor
                sol_arr[idx] = sol
                # cur_cost = min(sol, cur_cost)
                if sol < cur_cost:
                    cur_cost = sol
                    cur_perm = min_perm
                epoch_min = min(sol, epoch_min)
                if val < val_arr[idx]:
                    best_pos_arr[idx] = pos
                    val_arr[idx] = val
                    ratio = max(cur_mixed / val_arr[idx], val_arr[idx] / cur_mixed)
                    if (
                        val_arr[idx] < cur_mixed and sol_arr[idx] <= ratio * cur_cost
                    ):  # some relaxation is allowed
                        #                         print("! val: {}, cur_mixed:{}, min_pos: {}".format(val, cur_mixed, str(min_pos)))
                        min_pos = best_pos_arr[idx].copy()
                        cur_mixed = val_arr[idx].copy()
                        #                         print("! val: {}, cur_mixed:{}, min_pos: {}".format(val, cur_mixed, str(min_pos)))
                        best_epoch = e
        for thr in thrs:
            thr.join()
        epoch_elapsed = 0
        while not que.empty():
            idx, pos, min_perm, min_cost, avg_cost, elapsed_time = que.get()
            epoch_elapsed = max(elapsed_time, epoch_elapsed)
            if min_cost == float("inf") or avg_cost == float("inf"):
                val = float("inf")
                sol = float("inf")
            else:
                val = (
                    (1 - parameters["mixed"]) * min_cost
                    + parameters["mixed"] * avg_cost
                    + parameters["reg"] * 10 ** lbd * LA.norm(lbd_coef * pos) ** 2
                )
                sol = min_cost
            sol_arr[idx] = sol
            # cur_cost = min(sol, cur_cost)
            if sol < cur_cost:
                cur_cost = sol
                cur_perm = min_perm
            epoch_min = min(sol, epoch_min)
            save_solution(
                fileName,
                e,
                pos,
                min_perm,
                min_cost,
                val,
                elapsed_time,
                parameters,
                bounds,
            )
            if val < val_arr[idx]:
                best_pos_arr[idx] = pos
                val_arr[idx] = val
                #                 print("min_pos: {}".format(str(min_pos)))
                ratio = max(cur_mixed / val_arr[idx], val_arr[idx] / cur_mixed)
                if (
                    val_arr[idx] < cur_mixed and sol_arr[idx] <= ratio * cur_cost
                ):  # some relaxation is allowed
                    #                     print("val: {}, cur_mixed:{}, min_pos: {}".format(val, cur_mixed, str(min_pos)))
                    min_pos = best_pos_arr[idx].copy()
                    cur_mixed = val_arr[idx].copy()
                    #                     print("val: {}, cur_mixed:{}, min_pos: {}".format(val, cur_mixed, str(min_pos)))
                    best_epoch = e
        # clustering on all the historical particles positions
        pos_arr_his = np.concatenate([pos_arr_his, pos_arr], axis=0).astype(float)
        val_arr_his = np.concatenate([val_arr_his, val_arr], axis=0).astype(float)
        sol_arr_his = np.concatenate([sol_arr_his, sol_arr], axis=0).astype(float)
        clustering, clustering_means, clustering_var, cluster_order = get_clustering(
            pos_arr_his, val_arr_his
        )

        # save info about every epoch
        save_epoch(
            fileName,
            e,
            cur_cost,
            epoch_min,
            cur_mixed,
            min_pos,
            epoch_elapsed,
            parameters,
            bounds,
        )
        total_elapsed += epoch_elapsed
        # print out the current best
        print("{}".format(Path(fileName).stem))
        print(f"Best solution found at {e + 1} Epoch: {epoch_min}")
        print(f"Best solution found after {e + 1} Epoch: {cur_cost}")
        print(f"Best mixed solution found after {e + 1} Epoch: {cur_mixed}")
        for i in range(len(bounds["min"])):
            print(f"Best {chr(65 + i)} found after {e + 1} Epoch: {min_pos[i]}")
    return cur_perm, cur_cost, cur_mixed, min_pos, best_epoch, total_elapsed


def pso(fileName, inst, rules, bounds, parameters, solve, prep_config):

    print("\n{}".format(Path(fileName).stem))
    print("PSO initialization start ...\n")

    total_elapsed = 0

    # create topology
    deg_sequence = get_deg_sequence(parameters["particles"], parameters["topo"])
    if len(deg_sequence) != 0:
        g = create_graph(parameters["particles"], deg_sequence)
        adj = nx.to_numpy_matrix(g)
    else:
        adj = np.ones((parameters["particles"], parameters["particles"]))
        np.fill_diagonal(adj, 0)
    neighbor = get_neighbor(adj, U=parameters["U"])

    # initialization
    if not parameters["ini"]:
        pos_arr = np.concatenate(
            [
                np.random.uniform(
                    bounds["min"][i] * getattr(inst, parameters["base"]),
                    bounds["max"][i] * getattr(inst, parameters["base"]),
                    parameters["particles"],
                ).reshape(parameters["particles"], 1)
                for i in range(len(bounds["min"]))
            ],
            axis=1,
        ).astype(float)
    else:
        pos_arr = np.concatenate(
            [
                np.linspace(
                    bounds["min"][i] * getattr(inst, parameters["base"]),
                    bounds["max"][i] * getattr(inst, parameters["base"]),
                    num=parameters["particles"],
                ).reshape(parameters["particles"], 1)
                for i in range(len(bounds["min"]))
            ],
            axis=1,
        ).astype(float)

    best_pos_arr = pos_arr.copy()
    vel_arr = np.concatenate(
        [
            np.random.uniform(
                bounds["min"][i] * getattr(inst, parameters["base"])
                - bounds["max"][i] * getattr(inst, parameters["base"]),
                bounds["max"][i] * getattr(inst, parameters["base"])
                - bounds["min"][i] * getattr(inst, parameters["base"]),
                parameters["particles"],
            ).reshape(parameters["particles"], 1)
            for i in range(len(bounds["min"]))
        ],
        axis=1,
    ).astype(float)
    val_arr = np.zeros((parameters["particles"])).astype(float)
    sol_arr = np.zeros((parameters["particles"])).astype(
        float
    )  # array for storing the minimum solution
    cur_mixed = float("inf")
    cur_cost = float("inf")
    cur_perm = []
    min_pos = [None] * len(bounds["min"])
    best_epoch = -1
    lbd = -3  # lambda for the regularization term
    lbd_coef = [1, 1.5]
    lbd_flag = 0
    # parameters['reg'] = 0.5 if enabled else 0

    # get initial values
    thrs = []
    que = queue.Queue()
    for idx, pos in enumerate(pos_arr):
        thr = Thread(
            target=lambda q, arg: q.put((idx, *solve(*arg))),
            args=(
                que,
                (
                    inst,
                    rules,
                    pos,
                    parameters["num_run"],
                    parameters["num_iteration"][0],
                    prep_config
                ),
            ),
        )
        thrs.append(thr)
        thr.start()
    for thr in thrs:
        thr.join()

    epoch_elapsed = 0
    while not que.empty():
        idx, pos, min_perm, min_cost, avg_cost, elapsed_time = que.get()
        epoch_elapsed = max(epoch_elapsed, elapsed_time)
        if min_cost == float("inf") or avg_cost == float("inf"):
            val = float("inf")
            sol = float("inf")
        else:
            if not lbd_flag:
                lbd = min(
                    lbd,
                    -int(
                        math.log(
                            (1 - parameters["mixed"]) * min_cost
                            + parameters["mixed"] * avg_cost,
                            10,
                        )
                    ),
                )
                lbd_flag = 1
                print("lambda: {}".format(lbd))
            val = (
                (1 - parameters["mixed"]) * min_cost
                + parameters["mixed"] * avg_cost
                + parameters["reg"] * 10 ** lbd * LA.norm(lbd_coef * pos) ** 2
            )
            sol = min_cost
        # cur_cost = min(sol, cur_cost)
        if sol < cur_cost:
            cur_cost = sol
            cur_perm = min_perm
        save_solution(
            fileName, -1, pos, min_perm, min_cost, val, elapsed_time, parameters, bounds
        )
        if val < cur_mixed:
            min_pos = pos
            cur_mixed = val
        val_arr[idx] = val
        sol_arr[idx] = sol

    save_epoch(
        fileName,
        -1,
        cur_cost,
        cur_cost,
        cur_mixed,
        min_pos,
        epoch_elapsed,
        parameters,
        bounds,
    )
    print("\nInitialization finished")
    print(f"Best solution found after initialization: {cur_cost}")
    print(f"Best mixed solution found after initialization: {cur_mixed}")
    for i in range(len(bounds["min"])):
        print(f"Best {chr(65 + i)} found initialization: {min_pos[i]}")

    # start optimization
    for e in range(parameters["epoch"]):
        print(f"\nEpoch {e + 1} start:")
        start = time.time()
        vel_arr = (
            round(
                parameters["w"]
                + (parameters["w0"] - parameters["w"])
                * (1 - (e + 1) / parameters["epoch"]),
                2,
            )
            * vel_arr
            + round(
                parameters["c1"]
                + (parameters["c10"] - parameters["c1"])
                * (1 - (e + 1) / parameters["epoch"]),
                2,
            )
            * np.random.uniform(0, 1, len(bounds["min"]))
            * (best_pos_arr - pos_arr)
            + round(
                parameters["c2"]
                + (parameters["c20"] - parameters["c2"])
                * (1 - (e + 1) / parameters["epoch"]),
                2,
            )
            * np.random.uniform(0, 1, len(bounds["min"]))
            * (get_min_pos(pos_arr, val_arr, neighbor) - pos_arr)
        )
        pos_arr = np.concatenate(
            [
                np.array(
                    [
                        min(
                            bounds["max"][j] * getattr(inst, parameters["base"]),
                            max(
                                bounds["min"][j] * getattr(inst, parameters["base"]),
                                pos_arr[i][j] + vel_arr[i][j],
                            ),
                        )
                        for i in range(parameters["particles"])
                    ]
                ).reshape(parameters["particles"], 1)
                for j in range(len(bounds["min"]))
            ],
            axis=1,
        ).astype(float)

        epoch_min = float("inf")
        thrs = []
        que = queue.Queue()
        for idx, pos in enumerate(pos_arr):
            thr = Thread(
                target=lambda q, arg: q.put((idx, *solve(*arg))),
                args=(
                    que,
                    (
                        inst,
                        rules,
                        pos,
                        parameters["num_run"],
                        parameters["num_iteration"][e + 1],
                        prep_config 
                    ),
                ),
            )
            thrs.append(thr)
            thr.start()
        for thr in thrs:
            thr.join()

        epoch_elapsed = 0
        while not que.empty():
            idx, pos, min_perm, min_cost, avg_cost, elapsed_time = que.get()
            epoch_elapsed = max(elapsed_time, epoch_elapsed)
            if min_cost == float("inf") or avg_cost == float("inf"):
                val = float("inf")
                sol = float("inf")
            else:
                val = (
                    (1 - parameters["mixed"]) * min_cost
                    + parameters["mixed"] * avg_cost
                    + parameters["reg"] * 10 ** lbd * LA.norm(lbd_coef * pos) ** 2
                )
                sol = min_cost
            sol_arr[idx] = sol
            # cur_cost = min(sol, cur_cost)
            if sol < cur_cost:
                cur_cost = sol
                cur_perm = min_perm
            epoch_min = min(sol, epoch_min)
            save_solution(
                fileName,
                e,
                pos,
                min_perm,
                min_cost,
                val,
                elapsed_time,
                parameters,
                bounds,
            )
            if val < val_arr[idx]:
                best_pos_arr[idx] = pos
                val_arr[idx] = val
                ratio = max(cur_mixed / val_arr[idx], val_arr[idx] / cur_mixed)
                if (
                    val_arr[idx] < cur_mixed and sol_arr[idx] <= ratio * cur_cost
                ):  # some relaxation is allowed
                    min_pos = best_pos_arr[idx].copy()
                    cur_mixed = val_arr[idx].copy()
                    best_epoch = e

        # save info about every epoch
        save_epoch(
            fileName,
            e,
            cur_cost,
            epoch_min,
            cur_mixed,
            min_pos,
            epoch_elapsed,
            parameters,
            bounds,
        )
        total_elapsed += epoch_elapsed
        # print out the current best
        print("{}".format(Path(fileName).stem))
        print(f"Best solution found at {e + 1} Epoch: {epoch_min}")
        print(f"Best solution found after {e + 1} Epoch: {cur_cost}")
        print(f"Best mixed solution found after {e + 1} Epoch: {cur_mixed}")
        for i in range(len(bounds["min"])):
            print(f"Best {chr(65 + i)} found after {e + 1} Epoch: {min_pos[i]}")
    return cur_perm, cur_cost, cur_mixed, min_pos, best_epoch, total_elapsed


def pso_ini_state(fileName, inst, rules, bounds, parameters, solve, prep_config): # pso_all_mod

    bounds_copy = bounds.copy()
    num_penalties = 0
    for k, v in sorted(bounds_copy.items()):
        if len(k) == 2 and k[-1].isalpha():
            bounds_copy[k] = (
                v[0] * getattr(inst, parameters["base"]),
                v[1] * getattr(inst, parameters["base"]),
            )
            num_penalties += 1

    print("\n{}".format(Path(fileName).stem))
    print("PSO initialization start ...\n")

    total_elapsed = 0

    # create topology
    deg_sequence = get_deg_sequence(parameters["particles"], parameters["topo"])
    if len(deg_sequence) != 0:
        g = create_graph(parameters["particles"], deg_sequence)
        adj = nx.to_numpy_matrix(g)
    else:
        adj = np.ones((parameters["particles"], parameters["particles"]))
        np.fill_diagonal(adj, 0)
    neighbor = get_neighbor(adj, U=parameters["U"])

    # initialization
    if not parameters["ini"]:
        pos_arr = np.concatenate(
            [
                np.random.uniform(v[0], v[1], parameters["particles"],).reshape(
                    parameters["particles"], 1
                )
                for k, v in sorted(bounds_copy.items())
            ],
            axis=1,
        ).astype(float)
    else:
        pos_arr = np.concatenate(
            [
                np.linspace(v[0], v[1], parameters["particles"],).reshape(
                    parameters["particles"], 1
                )
                for k, v in sorted(bounds_copy.items())
            ],
            axis=1,
        ).astype(float)

    best_pos_arr = pos_arr.copy()
    vel_arr = np.concatenate(
        [
            np.random.uniform(
                v[0] - v[1], v[1] - v[0], parameters["particles"],
            ).reshape(parameters["particles"], 1)
            for k, v in sorted(bounds_copy.items())
        ],
        axis=1,
    ).astype(float)

    val_arr = np.zeros((parameters["particles"])).astype(float)
    sol_arr = np.zeros((parameters["particles"])).astype(
        float
    )  # array for storing the minimum solution
    conf_arr = [None] * parameters["particles"]
    temp_qubo = list(rules.values())[0]
    N_states = [1024, 2048, 4096, 8192]
    N_state = N_states[bisect(N_states, temp_qubo._size)]
    initial_conf_arr = [None] * len(pos_arr)
    #     initial_conf_arr = [np.zeros(N_state, dtype=int)] * len(pos_arr)
    #     random_conf = np.random.randint(2, size=N_state)
    #     initial_conf_arr = [random_conf] * len(pos_arr)
    cur_mixed = float("inf")
    cur_cost = float("inf")
    cur_perm = []
    min_pos = [None] * len(bounds_copy)
    best_epoch = -1
    lbd = -3  # lambda for the regularization term
    lbd_coef = parameters["coef"]
    lbd_flag = 0

    # get initial values
    thrs = []
    que = queue.Queue()
    for idx, pos in enumerate(pos_arr):
        thr = Thread(
            target=lambda q, arg: q.put((idx, *solve(*arg))),
            args=(
                que,
                (
                    inst,
                    rules,
                    pos[:num_penalties],
                    int(pos[-2]),
                    int(pos[-1]),
                    prep_config,
                    initial_conf_arr[idx],
                ),
            ),
        )
        thrs.append(thr)
        thr.start()
    for thr in thrs:
        thr.join()

    epoch_elapsed = 0
    while not que.empty():
        idx, pos, min_perm, min_cost, avg_cost, min_conf, elapsed_time = que.get()
        conf_arr[idx] = min_conf
        epoch_elapsed = max(epoch_elapsed, elapsed_time)
        if min_cost == float("inf") or avg_cost == float("inf"):
            val = float("inf")
            sol = float("inf")
        else:
            if not lbd_flag:
                lbd = min(
                    lbd,
                    -int(
                        math.log(
                            (1 - parameters["mixed"]) * min_cost
                            + parameters["mixed"] * avg_cost,
                            10,
                        )
                    ),
                )
                lbd_flag = 1
                print("lambda: {}".format(lbd))
            val = (
                (1 - parameters["mixed"]) * min_cost
                + parameters["mixed"] * avg_cost
                + parameters["reg"] * 10 ** lbd * LA.norm(lbd_coef * pos) ** 2
            )
            sol = min_cost
        # cur_cost = min(sol, cur_cost)
        if sol < cur_cost:
            cur_cost = sol
            cur_perm = min_perm
        save_solution(
            fileName,
            -1,
            pos,
            min_perm,
            min_cost,
            val,
            elapsed_time,
            parameters,
            bounds_copy,
        )
        if val < cur_mixed:
            min_pos = pos
            cur_mixed = val
        val_arr[idx] = val
        sol_arr[idx] = sol
    #     val_arr[np.isinf(val_arr)] = 10 ** - (3 * lbd)
    #     sol_arr[np.isinf(sol_arr)] = 10 ** - (3 * lbd)
    val_arr[np.isinf(val_arr)] = 10 ** -(lbd - 1)
    sol_arr[np.isinf(sol_arr)] = 10 ** -(lbd - 1)

    # TODO: best conf
    #     min_conf_arr = conf_arr.copy()
    min_conf_arr = get_min_conf(conf_arr, val_arr, neighbor)
    #     min_conf_arr = get_min_conf_all(conf_arr, val_arr)

    save_epoch(
        fileName,
        -1,
        cur_cost,
        cur_cost,
        cur_mixed,
        min_pos,
        epoch_elapsed,
        parameters,
        bounds_copy,
    )
    print("\nInitialization finished")
    print(f"Best solution found after initialization: {cur_cost}")
    print(f"Best mixed solution found after initialization: {cur_mixed}")
    for i, k in enumerate(sorted(bounds_copy.keys())):
        print(f"Best {k[1:]} found initialization: {min_pos[i]}")

    # start optimization
    for e in range(parameters["epoch"]):
        print(f"\nEpoch {e + 1} start:")
        start = time.time()
        vel_arr = (
            round(
                parameters["w"]
                + (parameters["w0"] - parameters["w"])
                * (1 - (e + 1) / parameters["epoch"]),
                2,
            )
            * vel_arr
            + round(
                parameters["c1"]
                + (parameters["c10"] - parameters["c1"])
                * (1 - (e + 1) / parameters["epoch"]),
                2,
            )
            * np.random.uniform(0, 1, len(bounds_copy))
            * (best_pos_arr - pos_arr)
            + round(
                parameters["c2"]
                + (parameters["c20"] - parameters["c2"])
                * (1 - (e + 1) / parameters["epoch"]),
                2,
            )
            * np.random.uniform(0, 1, len(bounds_copy))
            * (get_min_pos(best_pos_arr, val_arr, neighbor) - pos_arr)
        )
        #                    (get_min_pos(pos_arr, val_arr, neighbor) - pos_arr))
        pos_arr = np.concatenate(
            [
                np.array(
                    [
                        min(
                            bounds_copy[k][1],
                            max(bounds_copy[k][0], pos_arr[i][j] + vel_arr[i][j],),
                        )
                        for i in range(parameters["particles"])
                    ]
                ).reshape(parameters["particles"], 1)
                for j, k in enumerate(sorted(bounds_copy.keys()))
            ],
            axis=1,
        ).astype(float)

        epoch_min = float("inf")
        thrs = []
        que = queue.Queue()
        for idx, pos in enumerate(pos_arr):
            thr = Thread(
                target=lambda q, arg: q.put((idx, *solve(*arg))),
                args=(
                    que,
                    (
                        inst,
                        rules,
                        pos[:num_penalties],
                        int(pos[-2]),
                        int(pos[-1]),
                        prep_config,
                        min_conf_arr[idx],
                    ),
                ),
            )
            thrs.append(thr)
            thr.start()
        for thr in thrs:
            thr.join()

        epoch_elapsed = 0
        while not que.empty():
            idx, pos, min_perm, min_cost, avg_cost, min_conf, elapsed_time = que.get()
            epoch_elapsed = max(elapsed_time, epoch_elapsed)
            if min_cost == float("inf") or avg_cost == float("inf"):
                val = float("inf")
                sol = float("inf")
            else:
                val = (
                    (1 - parameters["mixed"]) * min_cost
                    + parameters["mixed"] * avg_cost
                    + parameters["reg"] * 10 ** lbd * LA.norm(lbd_coef * pos) ** 2
                )
                sol = min_cost
            sol_arr[idx] = sol
            # cur_cost = min(sol, cur_cost)
            if sol < cur_cost:
                cur_cost = sol
                cur_perm = min_perm
            epoch_min = min(sol, epoch_min)
            save_solution(
                fileName,
                e,
                pos,
                min_perm,
                min_cost,
                val,
                elapsed_time,
                parameters,
                bounds_copy,
            )
            if val < val_arr[idx]:
                best_pos_arr[idx] = pos
                val_arr[idx] = val
                # TODO: only update best conf if solution improved
                conf_arr[idx] = min_conf
                ratio = max(cur_mixed / val_arr[idx], val_arr[idx] / cur_mixed)
                if (
                    val_arr[idx] < cur_mixed and sol_arr[idx] <= ratio * cur_cost
                ):  # some relaxation is allowed
                    min_pos = best_pos_arr[idx].copy()
                    cur_mixed = val_arr[idx].copy()
                    best_epoch = e
        #         val_arr[np.isinf(val_arr)] = 10 ** - (3 * lbd)
        #         sol_arr[np.isinf(sol_arr)] = 10 ** - (3 * lbd)
        val_arr[np.isinf(val_arr)] = 10 ** -(lbd - 1)
        sol_arr[np.isinf(sol_arr)] = 10 ** -(lbd - 1)
        # TODO: best conf
        #         min_conf_arr = conf_arr.copy()
        min_conf_arr = get_min_conf(conf_arr, val_arr, neighbor)
        #         min_conf_arr = get_min_conf_all(conf_arr, val_arr)

        # save info about every epoch
        save_epoch(
            fileName,
            e,
            cur_cost,
            epoch_min,
            cur_mixed,
            min_pos,
            epoch_elapsed,
            parameters,
            bounds_copy,
        )
        total_elapsed += epoch_elapsed
        # print out the current best
        print("{}".format(Path(fileName).stem))
        print(f"Best solution found at {e + 1} Epoch: {epoch_min}")
        print(f"Best solution found after {e + 1} Epoch: {cur_cost}")
        print(f"Best mixed solution found after {e + 1} Epoch: {cur_mixed}")
        for i, k in enumerate(sorted(bounds_copy.keys())):
            print(f"Best {k[1:]} found after {e + 1} Epoch: {min_pos[i]}")
    return cur_perm, cur_cost, cur_mixed, min_pos, best_epoch, total_elapsed


def pso_all(fileName, inst, rules, bounds, parameters, solve, prep_config):

    bounds_copy = bounds.copy()
    num_penalties = 0
    for k, v in sorted(bounds_copy.items()):
        if len(k) == 2 and k[-1].isalpha():
            bounds_copy[k] = (
                v[0] * getattr(inst, parameters["base"]),
                v[1] * getattr(inst, parameters["base"]),
            )
            num_penalties += 1

    print("\n{}".format(Path(fileName).stem))
    print("PSO initialization start ...\n")

    total_elapsed = 0

    # create topology
    deg_sequence = get_deg_sequence(parameters["particles"], parameters["topo"])
    if len(deg_sequence) != 0:
        g = create_graph(parameters["particles"], deg_sequence)
        adj = nx.to_numpy_matrix(g)
    else:
        adj = np.ones((parameters["particles"], parameters["particles"]))
        np.fill_diagonal(adj, 0)
    neighbor = get_neighbor(adj, U=parameters["U"])

    # initialization
    if not parameters["ini"]:
        pos_arr = np.concatenate(
            [
                np.random.uniform(v[0], v[1], parameters["particles"],).reshape(
                    parameters["particles"], 1
                )
                for k, v in sorted(bounds_copy.items())
            ],
            axis=1,
        ).astype(float)
    else:
        pos_arr = np.concatenate(
            [
                np.linspace(v[0], v[1], parameters["particles"],).reshape(
                    parameters["particles"], 1
                )
                for k, v in sorted(bounds_copy.items())
            ],
            axis=1,
        ).astype(float)

    best_pos_arr = pos_arr.copy()
    vel_arr = np.concatenate(
        [
            np.random.uniform(
                v[0] - v[1], v[1] - v[0], parameters["particles"],
            ).reshape(parameters["particles"], 1)
            for k, v in sorted(bounds_copy.items())
        ],
        axis=1,
    ).astype(float)

    val_arr = np.zeros((parameters["particles"])).astype(float)
    sol_arr = np.zeros((parameters["particles"])).astype(
        float
    )  # array for storing the minimum solution
    conf_arr = [None] * parameters["particles"]
    cur_mixed = float("inf")
    cur_cost = float("inf")
    cur_perm = []
    min_pos = [None] * len(bounds_copy)
    best_epoch = -1
    lbd = -3  # lambda for the regularization term
    lbd_coef = parameters["coef"]
    lbd_flag = 0

    # get initial values
    thrs = []
    que = queue.Queue()
    for idx, pos in enumerate(pos_arr):
        thr = Thread(
            target=lambda q, arg: q.put((idx, *solve(*arg))),
            args=(
                que,
                (inst, rules, pos[:num_penalties], int(pos[-2]), int(pos[-1]), prep_config,),
            ),
        )
        thrs.append(thr)
        thr.start()
    for thr in thrs:
        thr.join()

    epoch_elapsed = 0
    while not que.empty():
        idx, pos, min_perm, min_cost, avg_cost, min_conf, elapsed_time = que.get()
        conf_arr[idx] = min_conf
        epoch_elapsed = max(epoch_elapsed, elapsed_time)
        if min_cost == float("inf") or avg_cost == float("inf"):
            val = float("inf")
            sol = float("inf")
        else:
            if not lbd_flag:
                lbd = min(
                    lbd,
                    -int(
                        math.log(
                            (1 - parameters["mixed"]) * min_cost
                            + parameters["mixed"] * avg_cost,
                            10,
                        )
                    ),
                )
                lbd_flag = 1
                print("lambda: {}".format(lbd))
            val = (
                (1 - parameters["mixed"]) * min_cost
                + parameters["mixed"] * avg_cost
                + parameters["reg"] * 10 ** lbd * LA.norm(lbd_coef * pos) ** 2
            )
            sol = min_cost
        # cur_cost = min(sol, cur_cost)
        if sol < cur_cost:
            cur_cost = sol
            cur_perm = min_perm
        save_solution(
            fileName,
            -1,
            pos,
            min_perm,
            min_cost,
            val,
            elapsed_time,
            parameters,
            bounds_copy,
        )
        if val < cur_mixed:
            min_pos = pos
            cur_mixed = val
        val_arr[idx] = val
        sol_arr[idx] = sol
    val_arr[np.isinf(val_arr)] = 10 ** -(lbd - 1)
    sol_arr[np.isinf(sol_arr)] = 10 ** -(lbd - 1)

    save_epoch(
        fileName,
        -1,
        cur_cost,
        cur_cost,
        cur_mixed,
        min_pos,
        epoch_elapsed,
        parameters,
        bounds_copy,
    )
    print("\nInitialization finished")
    print(f"Best solution found after initialization: {cur_cost}")
    print(f"Best mixed solution found after initialization: {cur_mixed}")
    for i, k in enumerate(sorted(bounds_copy.keys())):
        print(f"Best {k[1:]} found initialization: {min_pos[i]}")

    # start optimization
    for e in range(parameters["epoch"]):
        print(f"\nEpoch {e + 1} start:")
        start = time.time()
        vel_arr = (
            round(
                parameters["w"]
                + (parameters["w0"] - parameters["w"])
                * (1 - (e + 1) / parameters["epoch"]),
                2,
            )
            * vel_arr
            + round(
                parameters["c1"]
                + (parameters["c10"] - parameters["c1"])
                * (1 - (e + 1) / parameters["epoch"]),
                2,
            )
            * np.random.uniform(0, 1, len(bounds_copy))
            * (best_pos_arr - pos_arr)
            + round(
                parameters["c2"]
                + (parameters["c20"] - parameters["c2"])
                * (1 - (e + 1) / parameters["epoch"]),
                2,
            )
            * np.random.uniform(0, 1, len(bounds_copy))
            * (get_min_pos(best_pos_arr, val_arr, neighbor) - pos_arr)
        )
        #                    (get_min_pos(pos_arr, val_arr, neighbor) - pos_arr))
        pos_arr = np.concatenate(
            [
                np.array(
                    [
                        min(
                            bounds_copy[k][1],
                            max(bounds_copy[k][0], pos_arr[i][j] + vel_arr[i][j],),
                        )
                        for i in range(parameters["particles"])
                    ]
                ).reshape(parameters["particles"], 1)
                for j, k in enumerate(sorted(bounds_copy.keys()))
            ],
            axis=1,
        ).astype(float)

        epoch_min = float("inf")
        thrs = []
        que = queue.Queue()
        for idx, pos in enumerate(pos_arr):
            thr = Thread(
                target=lambda q, arg: q.put((idx, *solve(*arg))),
                args=(
                    que,
                    (
                        inst,
                        rules,
                        pos[:num_penalties],
                        int(pos[-2]),
                        int(pos[-1]),
                        prep_config,
                        conf_arr[idx],
                    ),
                ),
            )
            thrs.append(thr)
            thr.start()
        for thr in thrs:
            thr.join()

        epoch_elapsed = 0
        while not que.empty():
            idx, pos, min_perm, min_cost, avg_cost, min_conf, elapsed_time = que.get()
            conf_arr[idx] = min_conf
            epoch_elapsed = max(elapsed_time, epoch_elapsed)
            if min_cost == float("inf") or avg_cost == float("inf"):
                val = float("inf")
                sol = float("inf")
            else:
                val = (
                    (1 - parameters["mixed"]) * min_cost
                    + parameters["mixed"] * avg_cost
                    + parameters["reg"] * 10 ** lbd * LA.norm(lbd_coef * pos) ** 2
                )
                sol = min_cost
            sol_arr[idx] = sol
            # cur_cost = min(sol, cur_cost)
            if sol < cur_cost:
                cur_cost = sol
                cur_perm = min_perm
            epoch_min = min(sol, epoch_min)
            save_solution(
                fileName,
                e,
                pos,
                min_perm,
                min_cost,
                val,
                elapsed_time,
                parameters,
                bounds_copy,
            )
            if val < val_arr[idx]:
                best_pos_arr[idx] = pos
                val_arr[idx] = val
                ratio = max(cur_mixed / val_arr[idx], val_arr[idx] / cur_mixed)
                if (
                    val_arr[idx] < cur_mixed and sol_arr[idx] <= ratio * cur_cost
                ):  # some relaxation is allowed
                    min_pos = best_pos_arr[idx].copy()
                    cur_mixed = val_arr[idx].copy()
                    best_epoch = e
        val_arr[np.isinf(val_arr)] = 10 ** -(lbd - 1)
        sol_arr[np.isinf(sol_arr)] = 10 ** -(lbd - 1)

        # save info about every epoch
        save_epoch(
            fileName,
            e,
            cur_cost,
            epoch_min,
            cur_mixed,
            min_pos,
            epoch_elapsed,
            parameters,
            bounds_copy,
        )
        total_elapsed += epoch_elapsed
        # print out the current best
        print("{}".format(Path(fileName).stem))
        print(f"Best solution found at {e + 1} Epoch: {epoch_min}")
        print(f"Best solution found after {e + 1} Epoch: {cur_cost}")
        print(f"Best mixed solution found after {e + 1} Epoch: {cur_mixed}")
        for i, k in enumerate(sorted(bounds_copy.keys())):
            print(f"Best {k[1:]} found after {e + 1} Epoch: {min_pos[i]}")
    return cur_perm, cur_cost, cur_mixed, min_pos, best_epoch, total_elapsed


def abc(fileName, inst, rules, bounds, parameters, solve, prep_config,):
    # parameters should include {'sn', 'limit', 'cycle'}

    def _random_initialize(bounds, size):
        return np.concatenate(
            [
                np.random.uniform(v[0], v[1], size,).reshape(size, 1)
                for k, v in sorted(bounds.items())
            ],
            axis=1,
        ).astype(float)

    def _fitness_thr(solve, prep_config, pos, epoch, tag):
        nonlocal fileName
        nonlocal inst
        nonlocal rules
        nonlocal parameters
        nonlocal bounds_copy
        nonlocal num_penalties
        nonlocal cur_mixed
        nonlocal cur_cost
        nonlocal cur_perm
        nonlocal min_pos
        nonlocal lbd
        nonlocal lbd_coef
        nonlocal best_epoch
        nonlocal epoch_min

        thrs = []
        que = queue.Queue()
        for idx, p in enumerate(pos):
            thr = Thread(
                target=lambda q, arg: q.put((idx, *solve(*arg))),
                args=(
                    que,
                    (
                        inst,
                        rules,
                        p[:num_penalties],
                        int(p[-2]),  # parameters['num_run']
                        int(p[-1]),  # parameters['num_iteration']
                        prep_config,
                    ),
                ),
            )
            thrs.append(thr)
            thr.start()
        for thr in thrs:
            thr.join()
        fitness = [None] * len(pos)
        while not que.empty():
            idx, p, min_perm, min_cost, avg_cost, elapsed_time = que.get()
            if min_cost != float("inf") or avg_cost != float("inf"):
                obj = (
                    (1 - parameters["mixed"]) * min_cost
                    + parameters["mixed"] * avg_cost
                    + parameters["reg"] * 10 ** lbd * LA.norm(lbd_coef * pos) ** 2
                )
                fitness[idx] = 1 / (1 + obj)
                if min_cost < cur_cost:
                    cur_cost = min_cost
                    cur_perm = min_perm
                    min_pos = p
                    best_epoch = epoch
                if obj < cur_mixed:
                    cur_mixed = obj
            else:
                obj = float("inf")
                fitness[idx] = 0
            epoch_min = min(epoch_min, min_cost)
            save_solution(
                fileName,
                str(epoch) + "_" + tag,
                p,
                min_perm,
                min_cost,
                obj,
                elapsed_time,
                parameters,
                bounds_copy,
            )
        return np.array(fitness).reshape(len(pos), 1)

    # update positions
    def _update(pos, food_pos):
        nonlocal bounds_copy
        return np.concatenate(
            [
                np.array(
                    [
                        min(
                            bounds_copy[k][1],
                            max(
                                bounds_copy[k][0],
                                pos[i][j]
                                + np.random.uniform(-1, 1)
                                * (pos[i][j] - food_pos[i][j]),
                            ),
                        )
                        for i in range(len(pos))
                    ]
                ).reshape(len(pos), 1)
                for j, k in enumerate(sorted(bounds_copy.keys()))
            ],
            axis=1,
        ).astype(float)

    bounds_copy = bounds.copy()
    num_penalties = 0
    for k, v in sorted(bounds_copy.items()):
        if len(k) == 2 and k[-1].isalpha():
            bounds_copy[k] = (
                v[0] * getattr(inst, parameters["base"]),
                v[1] * getattr(inst, parameters["base"]),
            )
            num_penalties += 1

    cur_mixed = float("inf")
    cur_cost = float("inf")
    cur_perm = []
    min_pos = [None] * len(bounds_copy)
    best_epoch = -1
    lbd = -3  # lambda for the regularization term
    lbd_coef = parameters["coef"]
    total_elapsed = 0
    best_epoch = 0

    print("\n{}".format(Path(fileName).stem))
    print("ABC start ...\n")

    # size of the population
    employ = parameters["sn"]
    onseeker = employ
    # random initialize the bees
    employed_pos = _random_initialize(bounds_copy, employ)
    onseeker_pos = _random_initialize(bounds_copy, onseeker)
    # random food source
    food_pos = _random_initialize(bounds_copy, employ)
    abandonment = np.zeros((employ, 1))
    scout = np.zeros((employ, 1))

    for epoch in range(parameters["cycle"]):
        epoch_min = float("inf")
        # scout start random search
        if np.sum(scout) >= 1:
            scout_pos = _random_initialize(bounds_copy, employ)
            employed_pos = scout_pos * scout + employed_pos * (1 - scout)
            scout = np.zeros((employ, 1))
        # evaluate employed
        epoch_start = time.time()
        employed_fitness = _fitness_thr(solve, prep_config, employed_pos, epoch, "employed")
        # update neighbors' positions for employed
        neighbor_pos = _update(employed_pos, food_pos)
        neighbor_fitness = _fitness_thr(solve, prep_config, neighbor_pos, epoch, "employed_neighbor")
        mask = np.array(employed_fitness - neighbor_fitness > 0).reshape(employ, 1)
        # abaddon the employed if no improvement has been made for $limit$ epochs
        abandonment += mask
        if epoch % parameters["limit"] == 0:
            scout = np.array(abandonment == parameters["limit"]).reshape(employ, 1)
            if np.sum(scout) >= 1:
                abandonment = np.zeros((employ, 1))
        employed_pos = employed_pos * mask + neighbor_pos * (1 - mask)
        employed_fitness = employed_fitness * mask + neighbor_fitness * (1 - mask)
        argsort_order = np.argsort(employed_fitness)
        sorted_pr = np.sort(employed_fitness) / np.sum(employed_fitness)
        # onlookers are searching
        selected_pos = [None] * onseeker
        selected_fitness = [None] * onseeker
        for idx, pos in enumerate(onseeker_pos):
            pr = np.random.uniform()
            employed_idx = argsort_order[bisect(sum(sorted_pr.tolist(), []), pr) - 1]
            selected_pos[idx] = employed_pos[employed_idx][0]
            selected_fitness[idx] = employed_fitness[employed_idx]
        selected_pos = np.array(selected_pos)
        selected_fitness = np.array(selected_fitness)
        # update neighbors' positions for onseeker
        neighbor_pos = _update(onseeker_pos, selected_pos)
        onseeker_fitness = _fitness_thr(solve, prep_config, onseeker_pos, epoch, "onseeker")
        neighbor_fitness = _fitness_thr(solve, prep_config, neighbor_pos, epoch, "onseeker_neighbor")
        mask = np.array(onseeker_fitness - neighbor_fitness > 0).reshape(onseeker, 1)
        onseeker_pos = onseeker_pos * mask + neighbor_pos * (1 - mask)
        onseeker_fitness = onseeker_fitness * mask + neighbor_fitness * (1 - mask)
        epoch_elapsed = time.time() - epoch_start
        save_epoch(
            fileName,
            epoch,
            cur_cost,
            epoch_min,
            cur_mixed,
            min_pos,
            epoch_elapsed,
            parameters,
            bounds_copy,
        )
    total_elapsed += epoch_elapsed
    return cur_perm, cur_cost, cur_mixed, min_pos, best_epoch, total_elapsed


def hyperopt(fileName, inst, rules, bounds, parameters, solve, prep_config):
    def _objective(x):
        nonlocal fileName
        nonlocal inst
        nonlocal rules
        nonlocal parameters
        nonlocal bounds
        nonlocal cur_cost
        nonlocal cur_mixed
        nonlocal cur_perm
        nonlocal epoch
        nonlocal best_epoch
        nonlocal total_elapsed
        nonlocal lbd
        nonlocal flag
        nonlocal cur_conf

        A, min_perm, min_cost, avg_cost, min_conf, elapsed_time, features = solve(
            inst,
            rules,
            [x[chr(65 + i)] for i in range(len(bounds["min"]))],
            parameters["num_run"],
            parameters["num_iteration"][epoch],
            prep_config,
            cur_conf
        )
        if min_conf is not None:
            cur_conf = min_conf
        # for aaai 2021 only
        cur_conf = None
        total_elapsed += elapsed_time
        if epoch <= 3 and min_cost != float("inf") and not flag:
            lbd = min(
                lbd,
                -int(
                    math.log(
                        max(1, np.abs((1 - parameters["mixed"]) * min_cost
                        + parameters["mixed"] * avg_cost)),
                        10,
                    )
                ),
            )
            flag = True
        mixed_cost = (
            (1 - parameters["mixed"]) * min_cost
            + parameters["mixed"] * avg_cost
            + parameters["reg"]
            * 10 ** lbd
            * LA.norm(parameters["reg_coef"] * np.array(A)) ** 2
        )
        cur_mixed = min(cur_mixed, mixed_cost)
        save_solution(
            fileName,
            epoch,
            A,
            min_perm,
            min_cost,
            mixed_cost,
            elapsed_time,
            parameters,
            bounds,
        )
        if min_perm is None or len(min_perm) == 0:
            epoch += 1
            return {"loss": float("inf"), "status": STATUS_OK}
        if min_cost < cur_cost:
            cur_cost = min_cost
            cur_perm = min_perm
            best_epoch = epoch
        epoch += 1
        return {"loss": mixed_cost, "status": STATUS_OK}

    # instance
    print("\n{}".format(Path(fileName).stem))

    # search space
    space = {}
    if not parameters["fixed"]:
        for i in range(len(bounds["min"])):
            space[chr(i + 65)] = hp.uniform(
                chr(i + 65),
                bounds["min"][i] * getattr(inst, parameters["base"]),
                bounds["max"][i] * getattr(inst, parameters["base"]),
            )
    else:
        for i in range(len(bounds["min"])):
            space[chr(i + 65)] = hp.uniform(
                chr(i + 65),
                bounds["min"][i] * getattr(inst, parameters["base"]),
                bounds["max"][i] * getattr(inst, parameters["base"])
                if i + 1 != len(bounds["min"])
                else bounds["min"][i] * getattr(inst, parameters["base"]),
            )

    # makespan
    cur_cost = float("inf")
    cur_mixed = float("inf")

    # order
    cur_perm = []

    # eval
    epoch = 0
    best_epoch = 0

    # time
    total_elapsed = 0

    # regularization term
    flag = False
    lbd = -3
    
    # initial state
    cur_conf = None

    # find best A
    trials = Trials()
    best = fmin(
        lambda x: _objective(x),
        space=space,
        algo=partial(tpe.suggest, n_startup_jobs=min(parameters["max_evals"] // 2, 20)),
        max_evals=parameters["max_evals"],
        trials=trials,
        rstate=None,
        # points_to_evaluate=[{'A':0.75 * getattr(inst, parameters['base']), 'B': 0.75 * getattr(inst, parameters['base'])}]
    )
    
    return (
        cur_perm,
        cur_cost,
        cur_mixed,
        [best[chr(65 + i)] for i in range(len(bounds["min"]))],
        best_epoch,
        total_elapsed,
    )


def optune(fileName, inst, rules, bounds, parameters, solve, prep_config):
    def _objective(trial):
        nonlocal fileName
        nonlocal inst
        nonlocal rules
        nonlocal bounds
        nonlocal parameters
        nonlocal cur_cost
        nonlocal cur_mixed
        nonlocal cur_perm
        nonlocal epoch
        nonlocal best_epoch
        nonlocal total_elapsed
        nonlocal lbd
        nonlocal flag
        nonlocal cur_conf

        space = {}
        if not parameters["fixed"]:
            for i in range(len(bounds["min"])):
                space[chr(i + 65)] = trial.suggest_uniform(
                    chr(i + 65),
                    bounds["min"][i] * getattr(inst, parameters["base"]),
                    bounds["max"][i] * getattr(inst, parameters["base"]),
                )
        else:
            for i in range(len(bounds["min"])):
                space[chr(i + 65)] = trial.suggest_uniform(
                    chr(i + 65),
                    bounds["min"][i] * getattr(inst, parameters["base"]),
                    bounds["max"][i] * getattr(inst, parameters["base"])
                    if i + 1 != len(bounds["min"])
                    else bounds["min"][i] * getattr(inst, parameters["base"]),
                )

        A_, min_perm, min_cost, avg_cost, min_conf, elapsed_time, features = solve(
            inst,
            rules,
            [space[chr(65 + i)] for i in range(len(bounds["min"]))],
            parameters["num_run"],
            parameters["num_iteration"][epoch],
            prep_config,
            cur_conf
        )
        if min_conf is not None:
            cur_conf = min_conf
        # for aaai 2021 only
        cur_conf = None
        total_elapsed += elapsed_time
        # mixed_cost = (1 - parameters["mixed"]) * min_cost + parameters[
        #     "mixed"
        # ] * avg_cost
        if epoch <= 3 and min_cost != float("inf") and not flag:
            lbd = min(
                lbd,
                -int(
                    math.log(
                        max(1, np.abs((1 - parameters["mixed"]) * min_cost
                        + parameters["mixed"] * avg_cost)),
                        10,
                    )
                ),
            )
            flag = True
        mixed_cost = (
            (1 - parameters["mixed"]) * min_cost
            + parameters["mixed"] * avg_cost
            + parameters["reg"]
            * 10 ** lbd
            * LA.norm(parameters["reg_coef"] * np.array(A_)) ** 2
        )
        cur_mixed = min(cur_mixed, mixed_cost)
        save_solution(
            fileName,
            epoch,
            A_,
            min_perm,
            min_cost,
            mixed_cost,
            elapsed_time,
            parameters,
            bounds,
        )
        if min_perm is None or len(min_perm) == 0:
            epoch += 1
            return float("inf")
        if min_cost < cur_cost:
            cur_cost = min_cost
            cur_perm = min_perm
            best_epoch = epoch
        epoch += 1
        return mixed_cost

    # instance
    print("\n{}".format(Path(fileName).stem))

    # makespan
    cur_cost = float("inf")
    cur_mixed = float("inf")

    # order
    cur_perm = []

    # eval
    epoch = 0
    best_epoch = 0

    # time
    total_elapsed = 0

    # regularization term
    lbd = -3
    flag = False
    
    # initial state
    cur_conf = None # start from random initial state

    # find best A
    study = optuna.create_study()
    optuna.logging.disable_default_handler()
    study.optimize(_objective, n_trials=parameters["max_evals"])
    return (
        cur_perm,
        cur_cost,
        cur_mixed,
        [study.best_params[chr(65 + i)] for i in range(len(bounds["min"]))],
        best_epoch,
        total_elapsed,
    )


def fixer(fileName, inst, rules, bounds, parameters, solve, prep_config, ratio=0.75938):
    # instance
    print("\n{}".format(Path(fileName).stem))

    # time
    total_elapsed = 0

    # parameter
    space = {}
    if not parameters["fixed"]:
        for i in range(len(bounds["min"])):
            space[chr(i + 65)] = ratio * getattr(inst, parameters["base"])
    else:
        for i in range(len(bounds["min"])):
            space[chr(i + 65)] = (
                ratio * getattr(inst, parameters["base"])
                if i + 1 != len(bounds["min"])
                else bounds["min"][i] * getattr(inst, parameters["base"])
            )

    A, min_perm, min_cost, avg_cost, elapsed_time = solve(
        inst,
        rules,
        [space[chr(65 + i)] for i in range(len(bounds["min"]))],
        parameters["num_run"],
        parameters["num_iteration"][0],
        prep_config,
    )
    total_elapsed += elapsed_time
    mixed_cost = (1 - parameters["mixed"]) * min_cost + parameters["mixed"] * avg_cost
    save_solution(
        fileName, 0, A, min_perm, min_cost, mixed_cost, elapsed_time, parameters, bounds
    )
    return (
        min_perm,
        min_cost,
        mixed_cost,
        [space[chr(65 + i)] for i in range(len(bounds["min"]))],
        0,
        total_elapsed,
    )


def randomer_seq(fileName, inst, rules, bounds, parameters, solve, prep_config, dist="uniform"):

    # instance
    print("\n{}".format(Path(fileName).stem))

    # makespan
    cur_cost = float("inf")
    cur_mixed = float("inf")

    # order
    cur_perm = []

    # eval
    best_epoch = 0
    best_params = {}

    # time
    total_elapsed = 0
    
    # initial state
    cur_conf = None
    
    space = {}
    if dist == "uniform":
        if not parameters["fixed"]:
            for i in range(len(bounds["min"])):
                space[chr(i + 65)] = np.random.uniform(
                    bounds["min"][i], bounds["max"][i], parameters["max_evals"]
                ) * getattr(inst, parameters["base"])
        else:
            for i in range(len(bounds["min"])):
                space[chr(i + 65)] = (
                    np.random.uniform(bounds["min"][i], bounds["max"][i], parameters["max_evals"])
                    * getattr(inst, parameters["base"])
                    if i + 1 != len(bounds["min"])
                    else bounds["min"][i] * getattr(inst, parameters["base"])
                )
    elif parameters["dist"] == "normal":
        mu = 0.7593874498315781
        sigma_2 = 0.0141
        if not parameters["fixed"]:
            for i in range(len(bounds["min"])):
                space[chr(i + 65)] = np.random.normal(
                    mu, math.sqrt(sigma_2), parameters["max_evals"]
                ) * getattr(inst, parameters["base"])
        else:
            for i in range(len(bounds["min"])):
                space[chr(i + 65)] = (
                    np.random.normal(mu, math.sqrt(sigma_2), parameters["max_evals"])
                    * getattr(inst, parameters["base"])
                    if i + 1 != len(bounds["min"])
                    else bounds["min"][i] * getattr(inst, parameters["base"])
                )

    for epoch in range(max(1, parameters["max_evals"])):

        A_, min_perm, min_cost, avg_cost, min_conf, elapsed_time, features = solve(
            inst,
            rules,
            [space[chr(65 + i)][epoch] for i in range(len(bounds["min"]))],
            parameters["num_run"],
            parameters["num_iteration"][0],
            prep_config,
            cur_conf
        )
        if min_conf is not None:
            cur_conf = min_conf
        total_elapsed += elapsed_time
        mixed_cost = (1 - parameters["mixed"]) * min_cost + parameters[
            "mixed"
        ] * avg_cost
        cur_mixed = min(cur_mixed, mixed_cost)
        save_solution(
            fileName,
            epoch,
            A_,
            min_perm,
            min_cost,
            mixed_cost,
            elapsed_time,
            parameters,
            bounds,
        )
        if min_cost < cur_cost:
            cur_cost = min_cost
            cur_perm = min_perm
            best_epoch = epoch
            best_params = A_
    
    return (
        cur_perm,
        cur_cost,
        cur_mixed,
        # [best_params[chr(65 + i)] for i in range(len(bounds["min"]))],
        best_params,
        best_epoch,
        total_elapsed,
    )


def randomer_thr(fileName, inst, rules, bounds, parameters, solve, prep_config, dist="uniform"):

    # instance
    print("\n{}".format(Path(fileName).stem))

    # makespan
    cur_cost = float("inf")
    cur_mixed = float("inf")

    # order
    cur_perm = []

    # eval
    best_params = {}
    best_trial = 0

    # time
    total_elapsed = 0

    # space_list
    space_lst = []
    for _ in range(max(1, parameters["max_evals"])):

        space = {}
        if dist == "uniform":
            if not parameters["fixed"]:
                for i in range(len(bounds["min"])):
                    space[chr(i + 65)] = random.uniform(
                        bounds["min"][i], bounds["max"][i]
                    ) * getattr(inst, parameters["base"])
            else:
                for i in range(len(bounds["min"])):
                    space[chr(i + 65)] = (
                        random.uniform(bounds["min"][i], bounds["max"][i])
                        * getattr(inst, parameters["base"])
                        if i + 1 != len(bounds["min"])
                        else bounds["min"][i] * getattr(inst, parameters["base"])
                    )
        elif parameters["dist"] == "normal":
            mu = 0.7593874498315781
            sigma_2 = 0.0141
            if not parameters["fixed"]:
                for i in range(len(bounds["min"])):
                    space[chr(i + 65)] = np.random.normal(
                        mu, math.sqrt(sigma_2)
                    ) * getattr(inst, parameters["base"])
            else:
                for i in range(len(bounds["min"])):
                    space[chr(i + 65)] = (
                        np.random.normal(mu, math.sqrt(sigma_2))
                        * getattr(inst, parameters["base"])
                        if i + 1 != len(bounds["min"])
                        else bounds["min"][i] * getattr(inst, parameters["base"])
                    )
        space_lst.append(space)

    start = time.time()
    thrs = []
    que = queue.Queue()
    for idx, space in enumerate(space_lst):
        thr = Thread(
            target=lambda q, arg: q.put((idx, *solve(*arg))),
            args=(
                que,
                (
                    inst,
                    rules,
                    [space[chr(65 + i)] for i in range(len(bounds["min"]))],
                    parameters["num_run"],
                    parameters["num_iteration"][0],
                    prep_config,
                ),
            ),
        )
        thrs.append(thr)
        thr.start()
    for thr in thrs:
        thr.join()

    while not que.empty():
        idx, A, min_perm, min_cost, avg_cost, min_conf, elapsed_time = que.get()
        mixed_cost = (1 - parameters["mixed"]) * min_cost + parameters[
            "mixed"
        ] * avg_cost
        cur_mixed = min(cur_mixed, mixed_cost)
        save_solution(
            fileName,
            idx,
            A,
            min_perm,
            min_cost,
            mixed_cost,
            elapsed_time,
            parameters,
            bounds,
        )
        if min_cost < cur_cost:
            cur_cost = min_cost
            cur_perm = min_perm
            best_trial = idx
            best_params = A
    total_elapsed = time.time() - start
    return cur_perm, cur_cost, cur_mixed, best_params, best_trial, total_elapsed


def ratio_predictor(fileName, inst, rules, bounds, parameters, solve, prep_config):
    # instance
    print("\n{}".format(Path(fileName).stem))

    # time
    total_elapsed = 0

    # predict the ratio
    model = joblib.load("./mlp/ratio_nn")
    scaler = joblib.load("./mlp/min_max_scaler")

    features = inst.calculate_features()
    features = scaler.transform(np.array([[8] + list(features)]))
    ratio = model.predict(features)[0][0]

    predicted_parameter = ratio * getattr(
        inst, parameters["base"]
    )  # base is the "_max_edge"

    A, min_perm, min_cost, avg_cost, elapsed_time = solve(
        inst,
        rules,
        [predicted_parameter],
        parameters["num_run"],
        parameters["num_iteration"][0],
        prep_config,
    )
    total_elapsed += elapsed_time
    mixed_cost = (1 - parameters["mixed"]) * min_cost + parameters["mixed"] * avg_cost
    save_solution(
        fileName, 0, A, min_perm, min_cost, mixed_cost, elapsed_time, parameters, bounds
    )
    return (min_perm, min_cost, mixed_cost, [predicted_parameter], 0, total_elapsed)


def ratio_predictor_thr(fileName, inst, rules, bounds, parameters, solve, prep_config):
    # instance
    print("\n{}".format(Path(fileName).stem))

    # makespan
    cur_cost = float("inf")
    cur_mixed = float("inf")

    # order
    cur_perm = []

    # eval
    best_trial = 0

    # time
    total_elapsed = 0

    # predict the ratio
    model = joblib.load("./mlp/ratio_nn")
    scaler = joblib.load("./mlp/min_max_scaler")

    features = inst.calculate_features()
    features = scaler.transform(np.array([[8] + list(features)]))
    ratio = model.predict(features)[0][0]

    predicted_parameter = ratio * getattr(
        inst, parameters["base"]
    )  # base is the "_max_edge"

    # list of identical predicted parameters
    parameter_lst = [predicted_parameter] * parameters["max_eval"]

    start = time.time()
    thrs = []
    que = queue.Queue()
    for idx, predicted_parameter in enumerate(parameter_lst):
        thr = Thread(
            target=lambda q, arg: q.put((idx, *solve(*arg))),
            args=(
                que,
                (
                    inst,
                    rules,
                    [predicted_parameter],
                    parameters["num_run"],
                    parameters["num_iteration"][0],
                    prep_config,
                ),
            ),
        )
        thrs.append(thr)
        thr.start()
    for thr in thrs:
        thr.join()

    while not que.empty():
        idx, A, min_perm, min_cost, avg_cost, elapsed_time = que.get()
        mixed_cost = (1 - parameters["mixed"]) * min_cost + parameters[
            "mixed"
        ] * avg_cost
        cur_mixed = min(cur_mixed, mixed_cost)
        save_solution(
            fileName,
            idx,
            A,
            min_perm,
            min_cost,
            mixed_cost,
            elapsed_time,
            parameters,
            bounds,
        )
        if min_cost < cur_cost:
            cur_cost = min_cost
            cur_perm = min_perm
            best_trial = idx
    total_elapsed = time.time() - start
    return (
        cur_perm,
        cur_cost,
        cur_mixed,
        [predicted_parameter],
        best_trial,
        total_elapsed,
    )
    
### ALM
def alm(fileName, inst, rules, bounds, parameters, solve_lambda, prep_config):
    
    def _random_action(lower_bounds, upper_bounds):
        return [round(np.random.uniform(lower_bounds[i], upper_bounds[i]), 2) for i in range(len(lower_bounds))]
    
    def _update_lambda(N, conf, A, lambda_param, update_coefficient=1.1):
        board = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if conf[i*N + j]:
                    board[i, j] = 1
        # evaluate c
        row_sum = np.sum(board, axis=1) - 1
        col_sum = np.sum(board, axis=0) - 1
        # update lambda
        for i in range(N):
            lambda_param[i] -= A[1] * row_sum[i] if A.shape[0] > 1 else A * row_sum[i]
            lambda_param[N+i] -= A[0] * col_sum[i] if A.shape[0] > 1 else A * col_sum[i]
        # update A
        A *= update_coefficient
        return A, lambda_param
    
    # instance
    print("\n{}".format(Path(fileName).stem))

    # makespan
    cur_cost = float("inf")
    cur_mixed = float("inf")

    # order
    cur_perm = []

    # eval
    best_epoch = 0
    best_params = None

    # time
    total_elapsed = 0
    
    # initial state
    cur_conf = None
    
    # store min conf
    confs = []
    
    lower_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['min']]
    upper_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['max']]
    
    # start from random parameters
    cur_dis = float('inf')
    while cur_dis == float('inf'):
        A = _random_action(lower_bounds, upper_bounds)
        lambda_param = [0.5 for i in range(2 * inst._no_nodes)]
        A, lambda_param, min_perm, cur_dis, avg_dis, min_conf, elapsed_time, features = solve_lambda(inst,
                    rules,
                    A,
                    lambda_param,
                    parameters["num_run"],
                    parameters["num_iteration"][0],
                    prep_config)
    mixed_cost = (1 - parameters["mixed"]) * cur_dis + parameters["mixed"] * avg_dis
    save_solution(fileName, -1, A, 
                  min_perm, cur_dis, mixed_cost, 
                  elapsed_time, parameters, bounds,)
    
    # after random initialization
    cur_cost = cur_dis
    cur_mixed = avg_dis
    cur_perm = min_perm
    cur_conf = min_conf
    best_params = A
    
        
    for epoch in range(max(1, parameters["max_evals"])):
        A, lambda_param = _update_lambda(inst._no_nodes, cur_conf, A, lambda_param, parameters["update_coefficient"])
        A, lambda_param, min_perm, min_cost, avg_cost, min_conf, elapsed_time, features = solve_lambda(
                                                                                    inst,
                                                                                    rules,
                                                                                    A,
                                                                                    lambda_param,
                                                                                    parameters["num_run"],
                                                                                    parameters["num_iteration"][0],
                                                                                    prep_config,
                                                                                    cur_conf
                                                                                )
        
        if min_cost != float('inf') and min_conf is not None:
            cur_conf = min_conf
        
        confs.append(min_conf)
        total_elapsed += elapsed_time
        mixed_cost = (1 - parameters["mixed"]) * min_cost + parameters["mixed"] * avg_cost
        cur_mixed = min(cur_mixed, mixed_cost)
        save_solution(fileName, epoch, A, 
                      min_perm, min_cost, mixed_cost, 
                      elapsed_time, parameters, bounds,)
        
        if min_cost < cur_cost:
            cur_cost = min_cost
            cur_perm = min_perm
            best_epoch = epoch
            best_params = A
    
    return (
        cur_perm,
        cur_cost,
        cur_mixed,
        best_params,
        best_epoch,
        total_elapsed,
    )
    
def alm_mod(fileName, inst, rules, bounds, parameters, solve_lambda, prep_config):
    
    def _random_action(lower_bounds, upper_bounds):
        return [round(np.random.uniform(lower_bounds[i], upper_bounds[i]), 2) for i in range(len(lower_bounds))]
    
    def _update_lambda(N, conf, A, lambda_param, update_coefficient=1.1):
        board = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if conf[i*N + j]:
                    board[i, j] = 1
        # evaluate c
        row_sum = np.sum(board, axis=1) - 1
        col_sum = np.sum(board, axis=0) - 1
        # update lambda
        for i in range(N):
            lambda_param[i] -= A[1] * row_sum[i] if A.shape[0] > 1 else A * row_sum[i]
            lambda_param[N+i] -= A[0] * col_sum[i] if A.shape[0] > 1 else A * col_sum[i]
        # update A
        A *= update_coefficient
        return A, lambda_param
    
    # instance
    print("\n{}".format(Path(fileName).stem))

    # makespan
    cur_cost = float("inf")
    cur_mixed = float("inf")

    # order
    cur_perm = []

    # eval
    best_epoch = 0
    best_params = None

    # time
    total_elapsed = 0
    
    # initial state
    cur_conf = None
    
    # store min conf
    confs = []
    
    lower_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['min']]
    upper_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['max']]
    
    # start from random parameters
    A = _random_action(lower_bounds, upper_bounds)
    lambda_param = [0.5 for i in range(2 * inst._no_nodes)]
    A, lambda_param, min_perm, cur_dis, avg_dis, min_conf, elapsed_time, features = solve_lambda(inst,
                rules,
                A,
                lambda_param,
                parameters["num_run"],
                parameters["num_iteration"][0],
                prep_config)
    mixed_cost = (1 - parameters["mixed"]) * cur_dis + parameters["mixed"] * avg_dis
    save_solution(fileName, -1, A, 
                  min_perm, cur_dis, mixed_cost, 
                  elapsed_time, parameters, bounds,)
    
    # after random initialization
    cur_cost = cur_dis
    cur_mixed = avg_dis
    cur_perm = min_perm
    cur_conf = min_conf
    best_params = A
    
        
    for epoch in range(max(1, parameters["max_evals"])):
        A, lambda_param = _update_lambda(inst._no_nodes, cur_conf, A, lambda_param, parameters["update_coefficient"])
        A, lambda_param, min_perm, min_cost, avg_cost, min_conf, elapsed_time, features = solve_lambda(
                                                                                    inst,
                                                                                    rules,
                                                                                    A,
                                                                                    lambda_param,
                                                                                    parameters["num_run"],
                                                                                    parameters["num_iteration"][0],
                                                                                    prep_config,
                                                                                    cur_conf
                                                                                )
        
        if min_conf is not None:
            cur_conf = min_conf
        
        confs.append(min_conf)
        total_elapsed += elapsed_time
        mixed_cost = (1 - parameters["mixed"]) * min_cost + parameters["mixed"] * avg_cost
        cur_mixed = min(cur_mixed, mixed_cost)
        save_solution(fileName, epoch, A, 
                      min_perm, min_cost, mixed_cost, 
                      elapsed_time, parameters, bounds,)
        
        if min_cost < cur_cost:
            cur_cost = min_cost
            cur_perm = min_perm
            best_epoch = epoch
            best_params = A
    
    return (
        cur_perm,
        cur_cost,
        cur_mixed,
        best_params,
        best_epoch,
        total_elapsed,
    )

# learned from the RL agent
def decreaser(fileName, inst, rules, bounds, parameters, solve, prep_config):

    def _random_action(lower_bounds, upper_bounds):
        return [round(np.random.uniform(lower_bounds[i], upper_bounds[i]), 2) for i in range(len(lower_bounds))]
    
    # instance
    print("\n{}".format(Path(fileName).stem))

    # makespan
    cur_cost = float("inf")
    cur_mixed = float("inf")

    # order
    cur_perm = []

    # eval
    best_epoch = 0
    best_params = []

    # time
    total_elapsed = 0
    
    # initial state
    cur_conf = None
    
    # store min conf
    confs = []
    
    lower_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['min']]
    upper_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['max']]
    
    # start from random parameters
    cur_dis = float('inf')
    while cur_dis == float('inf'):
        A = _random_action(lower_bounds, upper_bounds)
        A, min_perm, cur_dis, avg_dis, min_conf, elapsed_time, features = solve(inst,
                                                              rules,
                                                              A,
                                                              parameters["num_run"],
                                                              parameters["num_iteration"][0],
                                                              prep_config,)
    mixed_cost = (1 - parameters["mixed"]) * cur_dis + parameters["mixed"] * avg_dis
    save_solution(fileName, -1, A, 
                  min_perm, cur_dis, mixed_cost, 
                  elapsed_time, parameters, bounds,)
    
    # after random initialization
    cur_cost = cur_dis
    cur_mixed = avg_dis
    cur_perm = min_perm
    cur_conf = min_conf
    best_params = A
    
    # preparing the decreasing steps
    if len(lower_bounds) > 1:
        mu = [(x - lower_bounds[i]) / parameters["max_evals"] for i, x in enumerate(A)]
        cov = mu * np.eye(len(lower_bounds)) / 2 # [[1, 0], [0, 1]]
        steps = np.random.multivariate_normal(mu, cov, parameters["max_evals"])
    else:
        mu = (A[0] - lower_bounds[0]) / parameters["max_evals"]
        sigma = mu / 2 # 1
        steps = np.random.normal(mu, sigma, parameters["max_evals"])
        
    for epoch in range(max(1, parameters["max_evals"])):
        A = np.array(A) - steps[epoch - 1]
        A_, min_perm, min_cost, avg_cost, min_conf, elapsed_time, features = solve(
                                                                                    inst,
                                                                                    rules,
                                                                                    A,
                                                                                    parameters["num_run"],
                                                                                    parameters["num_iteration"][0],
                                                                                    prep_config,
                                                                                    cur_conf
                                                                                )
        if min_conf is not None:
            cur_conf = min_conf
            
        if min_cost == float('inf'):
            break
        
        confs.append(min_conf)
        total_elapsed += elapsed_time
        mixed_cost = (1 - parameters["mixed"]) * min_cost + parameters["mixed"] * avg_cost
        cur_mixed = min(cur_mixed, mixed_cost)
        save_solution(fileName, epoch, A, 
                      min_perm, min_cost, mixed_cost, 
                      elapsed_time, parameters, bounds,)
        
        if min_cost < cur_cost:
            cur_cost = min_cost
            cur_perm = min_perm
            best_epoch = epoch
            best_params = A_
        
    # test around the best params found
    cur_perm, cur_cost, cur_mixed, best_params, best_trial, total_elapsed = randomer(fileName, inst, rules, bounds, parameters, solve, best_params, steps)
    
    # test around the best params found with good initial state
#     cur_perm, cur_cost, cur_mixed, best_params, best_trial, total_elapsed = randomer_conf(fileName, inst, rules, bounds, parameters, solve, prep_config, best_params, steps, best_epoch, confs)
    
    if len(lower_bounds) == 1:
        best_params = best_params.tolist()
        best_params = best_params * 2
    
    return (
        cur_perm,
        cur_cost,
        cur_mixed,
        best_params,
        best_epoch,
        total_elapsed,
    )
    
def pop_decreaser(fileName, inst, rules, bounds, parameters, solve, prep_config):

    def _random_action(lower_bounds, upper_bounds):
        # return random parameters
        return [round(np.random.uniform(lower_bounds[i], upper_bounds[i]), 2) for i in range(len(lower_bounds))]
    
    def _start_ini(inst, rules, parameters, lower_bounds, upper_bounds, solve, prep_config):
        # feasible solution given random parameters
        
        cur_dis = float('inf')
        while cur_dis == float('inf'):
            A = _random_action(lower_bounds, upper_bounds)
            A, min_perm, cur_dis, avg_dis, min_conf, elapsed_time, features = solve(inst,
                                                                                    rules,
                                                                                    A,
                                                                                    parameters["num_run"],
                                                                                    parameters["num_iteration"][0],
                                                                                    prep_config,)
        return A, min_perm, cur_dis, avg_dis, min_conf, elapsed_time, features
    
    # instance
    print("\n{}".format(Path(fileName).stem))

    # makespan
    cur_costs = [float("inf")] * parameters["population_size"]

    # order
    cur_perms = [None] * parameters["population_size"]

    # eval
    best_iter = 0 # [0] * parameters["population_size"]
    best_perm = None
    min_cost = float('inf')
    best_params = [None] * parameters["population_size"]
    
    # best conf
    best_conf = None

    # time
    total_elapsed = 0
    
    # store confs
    cur_confs = [None] * parameters["population_size"]
    
    # store params
    cur_params = [None] * parameters["population_size"]
    
    # parameters bounds
    lower_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['min']]
    upper_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['max']]
    
    # start from random parameters
    thrs = []
    que = queue.Queue()
    for idx in range(parameters["population_size"]):
        thr = Thread(
            target=lambda q, arg: q.put((idx, *_start_ini(*arg))),
            args=(
                que,
                (inst, rules, parameters, lower_bounds, upper_bounds, solve, prep_config),
            ),
        )
        thrs.append(thr)
        thr.start()
    for thr in thrs:
        thr.join()

    while not que.empty():
        idx, A, min_perm, cur_dis, avg_dis, min_conf, elapsed_time, features = que.get()
        if cur_dis < min_cost:
            min_cost = cur_dis
            best_param = A
            best_conf = min_conf
            best_perm = min_perm
            
        cur_costs[idx] = cur_dis
        cur_perms[idx] = min_perm
        cur_confs[idx] = min_conf
        cur_params[idx] = A
        best_params[idx] = A
        
        save_pop_solution(
            fileName,
            -1,
            idx,
            A,
            min_perm,
            cur_dis,
            avg_dis,
            elapsed_time,
            parameters,
            bounds,
        )
    
    # preparing the decreasing steps for the population
    if len(lower_bounds) > 1:
        mu = [[(x - lower_bounds[i]) / parameters["max_evals"] for i, x in enumerate(A)] for A in cur_params]
        cov = [mu[i] * np.eye(len(lower_bounds)) / 2 for i in range(parameters["population_size"])] # [[1, 0], [0, 1]]
        steps = [np.random.multivariate_normal(mu[i], cov[i], parameters["max_evals"]) for i in range(parameters["population_size"])]
    else:
        mu = [(A[0] - lower_bounds[0]) / parameters["max_evals"] for A in cur_params]
        sigma = [mu[i] / 2 for i in range(parameters["population_size"])] # 1
        steps = [np.random.normal(mu[i], sigma[i], parameters["max_evals"]) for i in range(parameters["population_size"])]
        
    for epoch in range(max(1, parameters["max_evals"])):
        print("\nEpoch {}".format(epoch + 1))
        thrs = []
        que = queue.Queue()
        for idx in range(parameters["population_size"]):
            # speed up the tuning
            if cur_costs[idx] == float('inf'):
                continue
                
            thr = Thread(
                target=lambda q, arg: q.put((idx, *solve(*arg))),
                args=(
                    que,
                    (inst,
                    rules,
                    np.array(cur_params[idx]) - steps[idx][epoch],
                    parameters["num_run"],
                    parameters["num_iteration"][0],
                    prep_config,
                    cur_confs[idx] if (epoch + 1) % parameters["update_interval"] != 0 else best_conf,)
                ),
            )
            thrs.append(thr)
            thr.start()
        for thr in thrs:
            thr.join()

        while not que.empty():
            idx, A, min_perm, cur_dis, avg_dis, min_conf, elapsed_time, features = que.get()
            if cur_dis < min_cost:
                min_cost = cur_dis
                best_param = A
                best_conf = min_conf
                best_perm = min_perm
                best_iter = epoch
                
            if cur_dis < cur_costs[idx]:
                best_params[idx] = A
            
            cur_costs[idx] = cur_dis
            cur_perms[idx] = min_perm
            cur_confs[idx] = min_conf
            cur_params[idx] = A

            save_pop_solution(
                fileName,
                epoch,
                idx,
                A,
                min_perm,
                cur_dis,
                avg_dis,
                elapsed_time,
                parameters,
                bounds,
            )
            
        # stop criterion
        if len(set(cur_costs)) == 1 and cur_costs[0] == float('inf'):
            break
        
        total_elapsed += elapsed_time
    
    return (
        best_perm,
        min_cost,
        min_cost,
        best_param,
        best_iter,
        total_elapsed,
    )

def pop_ring_decreaser(fileName, inst, rules, bounds, parameters, solve, prep_config):

    def _random_action(lower_bounds, upper_bounds):
        # return random parameters
        return [round(np.random.uniform(lower_bounds[i], upper_bounds[i]), 2) for i in range(len(lower_bounds))]
    
    def _start_ini(inst, rules, parameters, lower_bounds, upper_bounds, solve, prep_config):
        # feasible solution given random parameters
        cur_dis = float('inf')
        while cur_dis == float('inf'):
            A = _random_action(lower_bounds, upper_bounds)
            A, min_perm, cur_dis, avg_dis, min_conf, elapsed_time, features = solve(inst,
                                                                                    rules,
                                                                                    A,
                                                                                    parameters["num_run"],
                                                                                    parameters["num_iteration"][0],
                                                                                    prep_config,)
        return A, min_perm, cur_dis, avg_dis, min_conf, elapsed_time, features
    
    def _get_neighors_conf(cur_costs, cur_confs):
        best_neighbor_confs = [None] * len(cur_costs)
        for i in range(len(cur_costs)):
            # two connected neighbors and self
            min_index = np.argmin([cur_costs[i - 1], cur_costs[i], cur_costs[(i + 1) % len(cur_costs)]]) 
            best_neighbor_confs[i] = [cur_confs[i - 1], cur_confs[i], cur_confs[(i + 1) % len(cur_confs)]][min_index]
        return best_neighbor_confs
    
    # instance
    print("\n{}".format(Path(fileName).stem))

    # makespan
    cur_costs = [float("inf")] * parameters["population_size"]

    # order
    cur_perms = [None] * parameters["population_size"]

    # eval
    best_iter = 0 # [0] * parameters["population_size"]
    best_perm = None
    best_params = [None] * parameters["population_size"]
    min_cost = float('inf')
    
    # best conf
    best_conf = None
    best_neighbor_confs = []

    # time
    total_elapsed = 0
    
    # store confs
    cur_confs = [None] * parameters["population_size"]
    
    # store params
    cur_params = [None] * parameters["population_size"]
    
    # parameters bounds
    lower_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['min']]
    upper_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['max']]
    
    # start from random parameters
    thrs = []
    que = queue.Queue()
    for idx in range(parameters["population_size"]):
        thr = Thread(
            target=lambda q, arg: q.put((idx, *_start_ini(*arg))),
            args=(
                que,
                (inst, rules, parameters, lower_bounds, upper_bounds, solve, prep_config),
            ),
        )
        thrs.append(thr)
        thr.start()
    for thr in thrs:
        thr.join()

    while not que.empty():
        idx, A, min_perm, cur_dis, avg_dis, min_conf, elapsed_time, features = que.get()
        if cur_dis < min_cost:
            min_cost = cur_dis
            best_param = A
            best_conf = min_conf
            best_perm = min_perm
        
        cur_costs[idx] = cur_dis
        cur_perms[idx] = min_perm
        cur_confs[idx] = min_conf
        cur_params[idx] = A
        best_params[idx] = A
        
        save_pop_solution(
            fileName,
            -1,
            idx,
            A,
            min_perm,
            cur_dis,
            avg_dis,
            elapsed_time,
            parameters,
            bounds,
        )
    print("\nInitialization finished")
    print(f"Best solution found after initialization: {min_cost}")
    
    # update the best neighbors configuration    
    best_neighbor_confs = _get_neighors_conf(cur_costs, cur_confs)
    
    # preparing the decreasing steps for the population
    if len(lower_bounds) > 1:
        mu = [[(x - lower_bounds[i]) / parameters["max_evals"] for i, x in enumerate(A)] for A in cur_params]
        cov = [mu[i] * np.eye(len(lower_bounds)) / 2 for i in range(parameters["population_size"])] # [[1, 0], [0, 1]]
        steps = [np.random.multivariate_normal(mu[i], cov[i], parameters["max_evals"]) for i in range(parameters["population_size"])]
    else:
        mu = [(A[0] - lower_bounds[0]) / parameters["max_evals"] for A in cur_params]
        sigma = [mu[i] / 2 for i in range(parameters["population_size"])] # 1
        steps = [np.random.normal(mu[i], sigma[i], parameters["max_evals"]) for i in range(parameters["population_size"])]
        
    for epoch in range(max(1, parameters["max_evals"])):
        print("\nEpoch {}".format(epoch + 1))
        thrs = []
        que = queue.Queue()
        for idx in range(parameters["population_size"]):
            # speed up the tuning
            if cur_costs[idx] == float('inf'):
                continue
                
            thr = Thread(
                target=lambda q, arg: q.put((idx, *solve(*arg))),
                args=(
                    que,
                    (inst,
                    rules,
                    np.clip(np.array(cur_params[idx]) - steps[idx][epoch], lower_bounds, upper_bounds), # clip between lower and upper bounds
                    parameters["num_run"],
                    parameters["num_iteration"][0],
                    prep_config,
                    cur_confs[idx] if (epoch + 1) % parameters["update_interval"] != 0 else best_neighbor_confs[idx],)
                ),
            )
            thrs.append(thr)
            thr.start()
        for thr in thrs:
            thr.join()

        while not que.empty():
            idx, A, min_perm, cur_dis, avg_dis, min_conf, elapsed_time, features = que.get()
            if cur_dis < min_cost:
                min_cost = cur_dis
                best_param = A
                best_conf = min_conf
                best_perm = min_perm
                best_iter = epoch
                
            if cur_dis < cur_costs[idx]:
                best_params[idx] = A
            
            cur_perms[idx] = min_perm
            cur_costs[idx] = cur_dis
            cur_confs[idx] = min_conf
            cur_params[idx] = A

            save_pop_solution(
                fileName,
                epoch,
                idx,
                A,
                min_perm,
                cur_dis,
                avg_dis,
                elapsed_time,
                parameters,
                bounds,
            )
        print(f"Best solution found after Iteration {epoch + 1}: {min_cost}")    
        # stop criterion
        if len(set(cur_costs)) == 1 and cur_costs[0] == float('inf'):
            break
        
        # update the best neighbors configuration
        best_neighbor_confs = _get_neighors_conf(cur_costs, cur_confs)
        
        total_elapsed += elapsed_time
    
    # if len(lower_bounds) == 1:
    #     best_param = best_param.tolist()
    #     best_param = best_param * 2
    
    return (
        best_perm,
        min_cost,
        min_cost,
        best_param,
        best_iter,
        total_elapsed,
    )


# pop_ring_mod
# only allow initial state of the smaller parameters to be changed to larger parameters
# only allow initial state of the larger parameters to be changed to smaller parameters
def pop_ring_decreaser_filter(fileName, inst, rules, bounds, parameters, solve, prep_config):

    def _random_action(lower_bounds, upper_bounds):
        # return random parameters
        return [round(np.random.uniform(lower_bounds[i], upper_bounds[i]), 2) for i in range(len(lower_bounds))]
    
    def _start_ini(inst, rules, parameters, lower_bounds, upper_bounds, solve, prep_config):
        # feasible solution given random parameters
        cur_dis = float('inf')
        while cur_dis == float('inf'):
            A = _random_action(lower_bounds, upper_bounds)
            A, min_perm, cur_dis, avg_dis, min_conf, elapsed_time, features = solve(inst,
                                                                                    rules,
                                                                                    A,
                                                                                    parameters["num_run"],
                                                                                    parameters["num_iteration"][0],
                                                                                    prep_config,)
        return A, min_perm, cur_dis, avg_dis, min_conf, elapsed_time, features
    
    def _get_neighors_conf(cur_costs, cur_confs, cur_params):
        best_neighbor_confs = [None] * len(cur_costs)
        for i in range(len(cur_costs)):
            # calculate the distance of the parameters to the origin
            params_dist = [LA.norm(x) for x in [cur_params[i - 1], cur_params[i], cur_params[(i + 1) % len(cur_params)]]]
            # filter out those parameters who are smaller than itself
#             dist_filter = (np.array(params_dist) <= params_dist[1]).astype(int)
            # filter out those parameters who are larger than itself
            dist_filter = (np.array(params_dist) >= params_dist[1]).astype(int)
            costs = [cur_costs[i - 1], cur_costs[i], cur_costs[(i + 1) % len(cur_costs)]]
            # find the index of the minimum postive cost
            min_index = np.argmin(np.where(dist_filter * costs > 0, dist_filter * costs, np.inf))
            best_neighbor_confs[i] = [cur_confs[i - 1], cur_confs[i], cur_confs[(i + 1) % len(cur_confs)]][min_index]
        return best_neighbor_confs
    
    # instance
    print("\n{}".format(Path(fileName).stem))

    # makespan
    cur_costs = [float("inf")] * parameters["population_size"]

    # order
    cur_perms = [None] * parameters["population_size"]

    # eval
    best_iter = 0 # [0] * parameters["population_size"]
    best_perm = None
    best_params = [None] * parameters["population_size"]
    min_cost = float('inf')
    
    # best conf
    best_conf = None
    best_neighbor_confs = []

    # time
    total_elapsed = 0
    
    # store confs
    cur_confs = [None] * parameters["population_size"]
    
    # store params
    cur_params = [None] * parameters["population_size"]
    
    # parameters bounds
    lower_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['min']]
    upper_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['max']]
    
    # start from random parameters
    thrs = []
    que = queue.Queue()
    for idx in range(parameters["population_size"]):
        thr = Thread(
            target=lambda q, arg: q.put((idx, *_start_ini(*arg))),
            args=(
                que,
                (inst, rules, parameters, lower_bounds, upper_bounds, solve, prep_config),
            ),
        )
        thrs.append(thr)
        thr.start()
    for thr in thrs:
        thr.join()

    while not que.empty():
        idx, A, min_perm, cur_dis, avg_dis, min_conf, elapsed_time, features = que.get()
        if cur_dis < min_cost:
            min_cost = cur_dis
            best_param = A
            best_conf = min_conf
            best_perm = min_perm
        
        cur_costs[idx] = cur_dis
        cur_perms[idx] = min_perm
        cur_confs[idx] = min_conf
        cur_params[idx] = A
        best_params[idx] = A
        
        save_pop_solution(
            fileName,
            -1,
            idx,
            A,
            min_perm,
            cur_dis,
            avg_dis,
            elapsed_time,
            parameters,
            bounds,
        )
    print("\nInitialization finished")
    print(f"Best solution found after initialization: {min_cost}")
    
    # update the best neighbors configuration    
    best_neighbor_confs = _get_neighors_conf(cur_costs, cur_confs, cur_params)
    
    # preparing the decreasing steps for the population
    if len(lower_bounds) > 1:
        mu = [[(x - lower_bounds[i]) / parameters["max_evals"] for i, x in enumerate(A)] for A in cur_params]
        cov = [mu[i] * np.eye(len(lower_bounds)) / 2 for i in range(parameters["population_size"])] # [[1, 0], [0, 1]]
        steps = [np.random.multivariate_normal(mu[i], cov[i], parameters["max_evals"]) for i in range(parameters["population_size"])]
    else:
        mu = [(A[0] - lower_bounds[0]) / parameters["max_evals"] for A in cur_params]
        sigma = [mu[i] / 2 for i in range(parameters["population_size"])] # 1
        steps = [np.random.normal(mu[i], sigma[i], parameters["max_evals"]) for i in range(parameters["population_size"])]
        
    for epoch in range(max(1, parameters["max_evals"])):
        print("\nEpoch {}".format(epoch + 1))
        thrs = []
        que = queue.Queue()
        for idx in range(parameters["population_size"]):
            # speed up the tuning
            if cur_costs[idx] == float('inf'):
                continue
                
            thr = Thread(
                target=lambda q, arg: q.put((idx, *solve(*arg))),
                args=(
                    que,
                    (inst,
                    rules,
                    np.clip(np.array(cur_params[idx]) - steps[idx][epoch], lower_bounds, upper_bounds), # clip between lower and upper bounds
                    parameters["num_run"],
                    parameters["num_iteration"][0],
                    prep_config,
                    cur_confs[idx] if (epoch + 1) % parameters["update_interval"] != 0 else best_neighbor_confs[idx],)
                ),
            )
            thrs.append(thr)
            thr.start()
        for thr in thrs:
            thr.join()

        while not que.empty():
            idx, A, min_perm, cur_dis, avg_dis, min_conf, elapsed_time, features = que.get()
            if cur_dis < min_cost:
                min_cost = cur_dis
                best_param = A
                best_conf = min_conf
                best_perm = min_perm
                best_iter = epoch
                
            if cur_dis < cur_costs[idx]:
                best_params[idx] = A
            
            cur_perms[idx] = min_perm
            cur_costs[idx] = cur_dis
            cur_confs[idx] = min_conf
            cur_params[idx] = A

            save_pop_solution(
                fileName,
                epoch,
                idx,
                A,
                min_perm,
                cur_dis,
                avg_dis,
                elapsed_time,
                parameters,
                bounds,
            )
        print(f"Best solution found after Iteration {epoch + 1}: {min_cost}")    
        # stop criterion
        if len(set(cur_costs)) == 1 and cur_costs[0] == float('inf'):
            break
        
        # update the best neighbors configuration
        best_neighbor_confs = _get_neighors_conf(cur_costs, cur_confs, cur_params)
        
        total_elapsed += elapsed_time
    
    # if len(lower_bounds) == 1:
    #     best_param = best_param.tolist()
    #     best_param = best_param * 2
    
    return (
        best_perm,
        min_cost,
        min_cost,
        best_param,
        best_iter,
        total_elapsed,
    )

def decrease_faster(fileName, inst, rules, bounds, parameters, solve, prep_config, k=8):

    def _random_action(lower_bounds, upper_bounds):
        return [round(np.random.uniform(lower_bounds[i], upper_bounds[i]), 2) for i in range(len(lower_bounds))]
    
    # instance
    print("\n{}".format(Path(fileName).stem))

    # makespan
    cur_cost = float("inf")
    cur_mixed = float("inf")

    # order
    cur_perm = []

    # eval
    best_epoch = 0
    best_params = {}

    # time
    total_elapsed = 0
    
    # initial state
    cur_conf = None
    
    lower_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['min']]
    upper_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['max']]
    
    # start from random parameters
    cur_dis = float('inf')
    while cur_dis == float('inf'):
        A = _random_action(lower_bounds, upper_bounds)
        A, min_perm, cur_dis, avg_dis, min_conf, elapsed_time, features = solve(inst,
                                                              rules,
                                                              A,
                                                              parameters["num_run"],
                                                              parameters["num_iteration"][0],
                                                              prep_config,)
    mixed_cost = (1 - parameters["mixed"]) * cur_dis + parameters["mixed"] * avg_dis
    save_solution(fileName, -1, A, 
                  min_perm, cur_dis, mixed_cost, 
                  elapsed_time, parameters, bounds,)
    
    # after random initialization
    cur_cost = cur_dis
    cur_mixed = avg_dis
    cur_perm = min_perm
    cur_conf = min_conf
    best_params = A
    
    for epoch in range(max(1, parameters["max_evals"])):
        delta = np.array(A) - np.array(lower_bounds)
        A = np.array(A) - delta / k
        A_, min_perm, min_cost, avg_cost, min_conf, elapsed_time, features = solve(
                                                                                    inst,
                                                                                    rules,
                                                                                    A,
                                                                                    parameters["num_run"],
                                                                                    parameters["num_iteration"][0],
                                                                                    prep_config,
                                                                                    cur_conf
                                                                                )
        if min_conf is not None:
            cur_conf = min_conf

        if min_cost == float('inf'):
            break
            
        total_elapsed += elapsed_time
        mixed_cost = (1 - parameters["mixed"]) * min_cost + parameters["mixed"] * avg_cost
        cur_mixed = min(cur_mixed, mixed_cost)
        save_solution(fileName, epoch, A, 
                      min_perm, min_cost, mixed_cost, 
                      elapsed_time, parameters, bounds,)
        
        if min_cost < cur_cost:
            cur_cost = min_cost
            cur_perm = min_perm
            best_epoch = epoch
            best_params = A_

    if len(lower_bounds) == 1:
        best_params = best_params.tolist()
        best_params = best_params * 2
    
    return (
        cur_perm,
        cur_cost,
        cur_mixed,
        best_params,
        best_epoch,
        total_elapsed,
    )
    
def randomer(fileName, inst, rules, bounds, parameters, solve, prep_config, best_params, steps):

    # instance
    print("Randomer")
    
    # random parameters
    rnd_para = []
    mean = best_params
    if steps is None:
        steps = best_params * (parameters["update_coefficient"] - 1)
    sigma = np.mean(steps) / 2
    if len(mean) > 1:
        cov = sigma * np.eye(len(mean))
        rnd_para = np.random.multivariate_normal(mean, cov, parameters["max_evals"])
    else:
        rnd_para = np.random.normal(mean, sigma, parameters["max_evals"])
    rnd_para = rnd_para.reshape(parameters["max_evals"], len(mean)).tolist()

    # makespan
    cur_cost = float("inf")
    cur_mixed = float("inf")

    # order
    cur_perm = []

    # eval
    best_params = []
    best_trial = 0

    # time
    total_elapsed = 0
    
    # start
    start = time.time()
    thrs = []
    que = queue.Queue()
    for idx, para in enumerate(rnd_para):
        thr = Thread(
            target=lambda q, arg: q.put((idx, *solve(*arg))),
            args=(
                que,
                (
                    inst,
                    rules,
                    para,
                    parameters["num_run"],
                    parameters["num_iteration"][0] * 10,
                    prep_config,
                ),
            ),
        )
        thrs.append(thr)
        thr.start()
    for thr in thrs:
        thr.join()

    while not que.empty():
        idx, A, min_perm, min_cost, avg_cost, min_conf, elapsed_time, features = que.get()
        mixed_cost = (1 - parameters["mixed"]) * min_cost + parameters[
            "mixed"
        ] * avg_cost
        cur_mixed = min(cur_mixed, mixed_cost)
        save_solution(
            fileName,
            idx,
            A,
            min_perm,
            min_cost,
            mixed_cost,
            elapsed_time,
            parameters,
            bounds,
        )
        if min_cost < cur_cost:
            cur_cost = min_cost
            cur_perm = min_perm
            best_trial = idx
            best_params = A
    total_elapsed = time.time() - start
    return cur_perm, cur_cost, cur_mixed, best_params, best_trial, total_elapsed

def randomer_conf(fileName, inst, rules, bounds, parameters, solve, prep_config, best_params, steps, best_epoch, confs):

    # instance
    print("Randomer")
    
    # random parameters
    rnd_para = []
    mean = best_params
    if steps is None:
        steps = best_params * (parameters["update_coefficient"] - 1)
    sigma = np.mean(steps) / 2
    if len(mean) > 1:
        cov = sigma * np.eye(len(mean))
        rnd_para = np.random.multivariate_normal(mean, cov, parameters["max_evals"])
    else:
        rnd_para = np.random.normal(mean, sigma, parameters["max_evals"])
    rnd_para = rnd_para.reshape(parameters["max_evals"], len(mean)).tolist()

    # makespan
    cur_cost = float("inf")
    cur_mixed = float("inf")

    # order
    cur_perm = []

    # eval
    best_params = []
    best_trial = 0

    # time
    total_elapsed = 0
    
    # start
    start = time.time()
    thrs = []
    que = queue.Queue()
    for idx, para in enumerate(rnd_para):
        thr = Thread(
            target=lambda q, arg: q.put((idx, *solve(*arg))),
            args=(
                que,
                (
                    inst,
                    rules,
                    para,
                    parameters["num_run"],
                    parameters["num_iteration"][0] * 10,
                    prep_config,
                    confs[best_epoch] # confs[best_epoch - 1]
                ),
            ),
        )
        thrs.append(thr)
        thr.start()
    for thr in thrs:
        thr.join()

    while not que.empty():
        idx, A, min_perm, min_cost, avg_cost, min_conf, elapsed_time, features = que.get()
        mixed_cost = (1 - parameters["mixed"]) * min_cost + parameters[
            "mixed"
        ] * avg_cost
        cur_mixed = min(cur_mixed, mixed_cost)
        save_solution(
            fileName,
            idx,
            A,
            min_perm,
            min_cost,
            mixed_cost,
            elapsed_time,
            parameters,
            bounds,
        )
        if min_cost < cur_cost:
            cur_cost = min_cost
            cur_perm = min_perm
            best_trial = idx
            best_params = A
    total_elapsed = time.time() - start
    return cur_perm, cur_cost, cur_mixed, best_params, best_trial, total_elapsed

### increaser
# increase the parameter from infeasible then sampling at the parameter which achieve the fist feasible solution
def increase_then_sampling(fileName, inst, rules, bounds, parameters, solve, prep_config):

    def _random_action(lower_bounds, upper_bounds):
        return [round(np.random.uniform(lower_bounds[i], upper_bounds[i]), 2) for i in range(len(lower_bounds))]
    
    # instance
    print("\n{}".format(Path(fileName).stem))

    # makespan
    cur_cost = float("inf")
    cur_mixed = float("inf")

    # order
    cur_perm = []

    # eval
    best_epoch = 0
    best_param = None

    # time
    total_elapsed = 0
    
    # initial state
    cur_conf = None
    
    # store min conf
    confs = []
    
    lower_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['min']]
    upper_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['max']]
    
    # initialization
    A = _random_action(lower_bounds, upper_bounds)
    cur_cost = float('inf')
    cur_mixed = float('inf')
    cur_perm = None
    cur_conf = None
    best_param = A
    
    for epoch in range(max(1, parameters["max_evals"])):
        if epoch == 0:
            A = np.array(A)
        else:
            A = parameters["update_coefficient"] * np.array(A)
        A, min_perm, min_cost, avg_cost, min_conf, elapsed_time, features = solve(inst,
                                                                                  rules,
                                                                                  A,
                                                                                  parameters["num_run"],
                                                                                  parameters["num_iteration"][0],
                                                                                  prep_config)     
        
        confs.append(min_conf)
        total_elapsed += elapsed_time
        mixed_cost = (1 - parameters["mixed"]) * min_cost + parameters["mixed"] * avg_cost
        cur_mixed = min(cur_mixed, mixed_cost)
        save_solution(fileName, epoch, A, 
                      min_perm, min_cost, mixed_cost, 
                      elapsed_time, parameters, bounds,)
        
        if min_cost < cur_cost:
            cur_conf = min_conf
            cur_cost = min_cost
            cur_perm = min_perm
            best_epoch = epoch
            best_param = A
            # break the loop when there is a feasible solution found
            break 
        
    # test around the best params found
    cur_perm_, cur_cost_, cur_mixed_, best_param_, best_trial, total_elapsed = randomer(fileName, inst, rules, bounds, parameters, solve, prep_config, best_param, None)
    
    # test around the best params and conf found
    # cur_perm_, cur_cost_, cur_mixed_, best_param_, best_trial, total_elapsed = randomer_conf(ileName, inst, rules, bounds, parameters, solve, prep_config, best_param, None, best_epoch, confs)
    
    if cur_cost_ == float('inf'):
        return (cur_perm,
                cur_cost,
                cur_mixed,
                best_param,
                best_epoch,
                total_elapsed)
        
    return (cur_perm_,
            cur_cost_,
            cur_mixed_,
            best_param_,
            best_epoch,
            total_elapsed)

def increase_then_sampling_mod(fileName, inst, rules, bounds, parameters, solve, prep_config):

    def _random_action(lower_bounds, upper_bounds):
        return [round(np.random.uniform(lower_bounds[i], upper_bounds[i]), 2) for i in range(len(lower_bounds))]
    
    # instance
    print("\n{}".format(Path(fileName).stem))

    # makespan
    best_cost = float("inf")
    best_mixed = float("inf")

    # order
    best_perm = []

    # eval
    best_epoch = 0
    best_param = None

    # time
    total_elapsed = 0
    
    # best initial state
    best_conf = None
    
    lower_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['min']]
    upper_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['max']]
    
    # initialization
    k = 10
    A = _random_action(lower_bounds, upper_bounds)
    costs = [float('inf')] * k
    perms = [None] * k
    confs = [None] * k
    params = [None] * k
    best_param = A
    flag = False
    
    # start
    while not flag:
        start = time.time()
        thrs = []
        que = queue.Queue()
        for idx in range(k):
            thr = Thread(
                target=lambda q, arg: q.put((idx, *solve(*arg))),
                args=(
                    que,
                    (
                        inst,
                        rules,
                        np.array(A) * (parameters["update_coefficient"]**idx),
                        parameters["num_run"],
                        parameters["num_iteration"][0] * 10,
                        prep_config,
                    ),
                ),
            )
            thrs.append(thr)
            thr.start()
        for thr in thrs:
            thr.join()

        while not que.empty():
            idx, A_, min_perm, min_cost, avg_cost, min_conf, elapsed_time, features = que.get()
            mixed_cost = (1 - parameters["mixed"]) * min_cost + parameters[
                "mixed"
            ] * avg_cost
            best_mixed = min(best_mixed, mixed_cost)
            save_solution(fileName, idx, A_,
                        min_perm, min_cost, mixed_cost, 
                        elapsed_time, parameters, bounds)
            # indicate feasible parameter found
            if min_cost != float('inf'):
                flag = True
            
            params[idx] = A_
            costs[idx] = min_cost
            perms[idx] = min_perm
            confs[idx] = min_conf
            
        if flag:
            for idx in range(k):
                if costs[idx] != float('inf'):
                    best_conf = confs[idx]
                    best_cost = costs[idx]
                    best_perm = perms[idx]
                    best_epoch = idx
                    best_param = params[idx] # np.array(A) * (parameters["update_coefficient"]**idx),
                    break
        else:
            A = np.array(A) * (parameters["update_coefficient"] ** k)
            costs = [float('inf')] * k
            perms = [None] * k
            confs = [None] * k
    
        
    # test around the best params found
    # smp_perm, smp_cost, smp_mixed, smp_param, best_trial, total_elapsed = randomer(fileName, inst, rules, bounds, parameters, solve, prep_config, best_param, None)
    
    # test around the best params and conf found
    smp_perm, smp_cost, smp_mixed, smp_param, best_trial, total_elapsed = randomer_conf(fileName, inst, rules, bounds, parameters, solve, prep_config, best_param, None, best_epoch, confs)
    
    if smp_cost == float('inf'):
        return (best_perm,
                best_cost,
                best_mixed,
                best_param,
                best_epoch,
                total_elapsed)
        
    return (smp_perm,
            smp_cost,
            smp_mixed,
            smp_param,
            best_epoch,
            total_elapsed)

# increase from smaller parameter
def increaser(fileName, inst, rules, bounds, parameters, solve, prep_config):

    def _random_action(lower_bounds, upper_bounds,):
        return [round(np.random.uniform(lower_bounds[i], upper_bounds[i]), 2) for i in range(len(lower_bounds))]
    
    # instance
    print("\n{}".format(Path(fileName).stem))

    # makespan
    cur_cost = float("inf")
    cur_mixed = float("inf")

    # order
    cur_perm = []

    # eval
    best_epoch = 0
    best_param = None

    # time
    total_elapsed = 0
    
    # initial state
    cur_conf = None
    
    # store min conf
    confs = []
    
    lower_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['min']]
    upper_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['max']]
    
    # start from random parameters
    cur_dis = float('inf')
    # while cur_dis == float('inf'):
    A = _random_action(lower_bounds, upper_bounds)
    A, min_perm, cur_dis, avg_dis, min_conf, elapsed_time, features = solve(inst,
                                                            rules,
                                                            A,
                                                            parameters["num_run"],
                                                            parameters["num_iteration"][0],
                                                            prep_config,)
    mixed_cost = (1 - parameters["mixed"]) * cur_dis + parameters["mixed"] * avg_dis
    save_solution(fileName, -1, A, 
                  min_perm, cur_dis, mixed_cost, 
                  elapsed_time, parameters, bounds,)
    
    # after random initialization
    cur_cost = cur_dis
    cur_mixed = avg_dis
    cur_perm = min_perm
    cur_conf = min_conf
    best_param = A
        
    for epoch in range(max(1, parameters["max_evals"])):
        A = parameters["update_coefficient"] * np.array(A) # increase the parameter
        A, min_perm, min_cost, avg_cost, min_conf, elapsed_time, features = solve(inst,
                                                                                  rules,
                                                                                  A,
                                                                                  parameters["num_run"],
                                                                                  parameters["num_iteration"][0],
                                                                                  prep_config,
                                                                                  cur_conf)
        
        cur_conf = min_conf
        confs.append(min_conf)
        total_elapsed += elapsed_time
        mixed_cost = (1 - parameters["mixed"]) * min_cost + parameters["mixed"] * avg_cost
        cur_mixed = min(cur_mixed, mixed_cost)
        save_solution(fileName, epoch, A, 
                      min_perm, min_cost, mixed_cost, 
                      elapsed_time, parameters, bounds,)
        
        if min_cost < cur_cost:
            cur_cost = min_cost
            cur_perm = min_perm
            best_epoch = epoch
            best_param = A
    
    return (cur_perm,
            cur_cost,
            cur_mixed,
            best_param,
            best_epoch,
            total_elapsed)
    
def pop_ring_increaser(fileName, inst, rules, bounds, parameters, solve, prep_config):

    def _random_action(lower_bounds, upper_bounds):
        # return random parameters
        return [round(np.random.uniform(lower_bounds[i], upper_bounds[i]), 2) for i in range(len(lower_bounds))]
    
    def _start_ini(inst, rules, parameters, lower_bounds, upper_bounds, solve, prep_config):
        # feasible solution given random parameters
#         cur_dis = float('inf')
#         while cur_dis == float('inf'):
        A = _random_action(lower_bounds, upper_bounds)
        A, min_perm, cur_dis, avg_dis, min_conf, elapsed_time, features = solve(inst,
                                                                                rules,
                                                                                A,
                                                                                parameters["num_run"],
                                                                                parameters["num_iteration"][0],
                                                                                prep_config,)
        return A, min_perm, cur_dis, avg_dis, min_conf, elapsed_time, features
    
    def _get_neighors_conf(cur_costs, cur_confs):
        best_neighbor_confs = [None] * len(cur_costs)
        for i in range(len(cur_costs)):
            # two connected neighbors and self
            min_index = np.argmin([cur_costs[i - 1], cur_costs[i], cur_costs[(i + 1) % len(cur_costs)]]) 
            best_neighbor_confs[i] = [cur_confs[i - 1], cur_confs[i], cur_confs[(i + 1) % len(cur_confs)]][min_index]
        return best_neighbor_confs
    
    # instance
    print("\n{}".format(Path(fileName).stem))

    # makespan
    cur_costs = [float("inf")] * parameters["population_size"]

    # order
    cur_perms = [None] * parameters["population_size"]

    # eval
    best_iter = 0 # [0] * parameters["population_size"]
    best_perm = None
    best_params = [None] * parameters["population_size"]
    min_cost = float('inf')
    
    # best conf
    best_conf = None
    best_neighbor_confs = []

    # time
    total_elapsed = 0
    
    # store confs
    cur_confs = [None] * parameters["population_size"]
    
    # store params
    cur_params = [None] * parameters["population_size"]
    
    # parameters bounds
    lower_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['min']]
    upper_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['max']]
    
    # start from random parameters
    thrs = []
    que = queue.Queue()
    for idx in range(parameters["population_size"]):
        thr = Thread(
            target=lambda q, arg: q.put((idx, *_start_ini(*arg))),
            args=(
                que,
                (inst, rules, parameters, lower_bounds, upper_bounds, solve, prep_config),
            ),
        )
        thrs.append(thr)
        thr.start()
    for thr in thrs:
        thr.join()

    while not que.empty():
        idx, A, min_perm, cur_dis, avg_dis, min_conf, elapsed_time, features = que.get()
        if cur_dis < min_cost:
            min_cost = cur_dis
            best_param = A
            best_conf = min_conf
            best_perm = min_perm
        
        cur_costs[idx] = cur_dis
        cur_perms[idx] = min_perm
        cur_confs[idx] = min_conf
        cur_params[idx] = A
        best_params[idx] = A
        
        save_pop_solution(
            fileName,
            -1,
            idx,
            A,
            min_perm,
            cur_dis,
            avg_dis,
            elapsed_time,
            parameters,
            bounds,
        )
    print("\nInitialization finished")
    print(f"Best solution found after initialization: {min_cost}")
    
    # update the best neighbors configuration    
    best_neighbor_confs = _get_neighors_conf(cur_costs, cur_confs)
        
    for epoch in range(max(1, parameters["max_evals"])):
        print("\nEpoch {}".format(epoch + 1))
        thrs = []
        que = queue.Queue()
        for idx in range(parameters["population_size"]):
            # speed up the tuning
#             if cur_costs[idx] == float('inf'):
#                 continue
                
            thr = Thread(
                target=lambda q, arg: q.put((idx, *solve(*arg))),
                args=(
                    que,
                    (inst,
                    rules,
                    parameters["update_coefficient"] * np.array(cur_params[idx]),
                    # np.clip(np.array(cur_params[idx]) - steps[idx][epoch], lower_bounds, upper_bounds), # clip between lower and upper bounds
                    parameters["num_run"],
                    parameters["num_iteration"][0],
                    prep_config,
                    cur_confs[idx] if (epoch + 1) % parameters["update_interval"] != 0 else best_neighbor_confs[idx],)
                ),
            )
            thrs.append(thr)
            thr.start()
        for thr in thrs:
            thr.join()

        while not que.empty():
            idx, A, min_perm, cur_dis, avg_dis, min_conf, elapsed_time, features = que.get()
            if cur_dis < min_cost:
                min_cost = cur_dis
                best_param = A
                best_conf = min_conf
                best_perm = min_perm
                best_iter = epoch
                
            if cur_dis < cur_costs[idx]:
                best_params[idx] = A
            
            cur_perms[idx] = min_perm
            cur_costs[idx] = cur_dis
            cur_confs[idx] = min_conf
            cur_params[idx] = A

            save_pop_solution(
                fileName,
                epoch,
                idx,
                A,
                min_perm,
                cur_dis,
                avg_dis,
                elapsed_time,
                parameters,
                bounds,
            )
        print(f"Best solution found after Iteration {epoch + 1}: {min_cost}")    
        # stop criterion
#         if len(set(cur_costs)) == 1 and cur_costs[0] == float('inf'):
#             break
        
        # update the best neighbors configuration
        best_neighbor_confs = _get_neighors_conf(cur_costs, cur_confs)
        
        total_elapsed += elapsed_time
    
    return (best_perm,
            min_cost,
            min_cost,
            best_param,
            best_iter,
            total_elapsed,)
    
### Pop with random
def pop_ring_randomer(fileName, inst, rules, bounds, parameters, solve, prep_config):

    def _random_action(lower_bounds, upper_bounds):
        # return random parameters
        return [round(np.random.uniform(lower_bounds[i], upper_bounds[i]), 2) for i in range(len(lower_bounds))]
    
    def _start_ini(inst, rules, parameters, lower_bounds, upper_bounds, solve, prep_config):
        # feasible solution given random parameters
#         cur_dis = float('inf')
#         while cur_dis == float('inf'):
        A = _random_action(lower_bounds, upper_bounds)
        A, min_perm, cur_dis, avg_dis, min_conf, elapsed_time, features = solve(inst,
                                                                                rules,
                                                                                A,
                                                                                parameters["num_run"],
                                                                                parameters["num_iteration"][0],
                                                                                prep_config,)
        return A, min_perm, cur_dis, avg_dis, min_conf, elapsed_time, features
    
    def _get_neighors_conf(cur_costs, cur_confs):
        best_neighbor_confs = [None] * len(cur_costs)
        for i in range(len(cur_costs)):
            # two connected neighbors and self
            min_index = np.argmin([cur_costs[i - 1], cur_costs[i], cur_costs[(i + 1) % len(cur_costs)]]) 
            best_neighbor_confs[i] = [cur_confs[i - 1], cur_confs[i], cur_confs[(i + 1) % len(cur_confs)]][min_index]
        return best_neighbor_confs
    
    # instance
    print("\n{}".format(Path(fileName).stem))

    # makespan
    cur_costs = [float("inf")] * parameters["population_size"]

    # order
    cur_perms = [None] * parameters["population_size"]

    # eval
    best_iter = 0 # [0] * parameters["population_size"]
    best_perm = None
    best_params = [None] * parameters["population_size"]
    min_cost = float('inf')
    
    # best conf
    best_conf = None
    best_neighbor_confs = []

    # time
    total_elapsed = 0
    
    # store confs
    cur_confs = [None] * parameters["population_size"]
    
    # store params
    cur_params = [None] * parameters["population_size"]
    
    # parameters bounds
    lower_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['min']]
    upper_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['max']]
    
    # start from random parameters
    thrs = []
    que = queue.Queue()
    for idx in range(parameters["population_size"]):
        thr = Thread(
            target=lambda q, arg: q.put((idx, *_start_ini(*arg))),
            args=(
                que,
                (inst, rules, parameters, lower_bounds, upper_bounds, solve, prep_config),
            ),
        )
        thrs.append(thr)
        thr.start()
    for thr in thrs:
        thr.join()

    while not que.empty():
        idx, A, min_perm, cur_dis, avg_dis, min_conf, elapsed_time, features = que.get()
        if cur_dis < min_cost:
            min_cost = cur_dis
            best_param = A
            best_conf = min_conf
            best_perm = min_perm
        
        cur_costs[idx] = cur_dis
        cur_perms[idx] = min_perm
        cur_confs[idx] = min_conf
        cur_params[idx] = A
        best_params[idx] = A
        
        save_pop_solution(
            fileName,
            -1,
            idx,
            A,
            min_perm,
            cur_dis,
            avg_dis,
            elapsed_time,
            parameters,
            bounds,
        )
    print("\nInitialization finished")
    print(f"Best solution found after initialization: {min_cost}")
    
    # update the best neighbors configuration    
    best_neighbor_confs = _get_neighors_conf(cur_costs, cur_confs)
        
    for epoch in range(max(1, parameters["max_evals"])):
        print("\nEpoch {}".format(epoch + 1))
        thrs = []
        que = queue.Queue()
        for idx in range(parameters["population_size"]):
            # speed up the tuning
#             if cur_costs[idx] == float('inf'):
#                 continue
                
            thr = Thread(
                target=lambda q, arg: q.put((idx, *solve(*arg))),
                args=(
                    que,
                    (inst,
                    rules,
                    np.random.uniform(parameters["random_coefficients"][0] if cur_costs[idx] != float('inf') else 1, parameters["random_coefficients"][1]) * np.array(cur_params[idx]),
                    # np.clip(np.array(cur_params[idx]) - steps[idx][epoch], lower_bounds, upper_bounds), # clip between lower and upper bounds
                    parameters["num_run"],
                    parameters["num_iteration"][0],
                    prep_config,
                    cur_confs[idx] if (epoch + 1) % parameters["update_interval"] != 0 else best_neighbor_confs[idx],)
                ),
            )
            thrs.append(thr)
            thr.start()
        for thr in thrs:
            thr.join()

        while not que.empty():
            idx, A, min_perm, cur_dis, avg_dis, min_conf, elapsed_time, features = que.get()
            if cur_dis < min_cost:
                min_cost = cur_dis
                best_param = A
                best_conf = min_conf
                best_perm = min_perm
                best_iter = epoch
                
            if cur_dis < cur_costs[idx]:
                best_params[idx] = A
            
            cur_perms[idx] = min_perm
            cur_costs[idx] = cur_dis
            cur_confs[idx] = min_conf
            cur_params[idx] = A

            save_pop_solution(
                fileName,
                epoch,
                idx,
                A,
                min_perm,
                cur_dis,
                avg_dis,
                elapsed_time,
                parameters,
                bounds,
            )
        print(f"Best solution found after Iteration {epoch + 1}: {min_cost}")    
        # stop criterion
#         if len(set(cur_costs)) == 1 and cur_costs[0] == float('inf'):
#             break
        
        # update the best neighbors configuration
        best_neighbor_confs = _get_neighors_conf(cur_costs, cur_confs)
        
        total_elapsed += elapsed_time
    
    return (best_perm,
            min_cost,
            min_cost,
            best_param,
            best_iter,
            total_elapsed,)
    
def single_randomer(fileName, inst, rules, bounds, parameters, solve, prep_config):

    def _random_action(lower_bounds, upper_bounds):
        # return random parameters
        return [round(np.random.uniform(lower_bounds[i], upper_bounds[i]), 2) for i in range(len(lower_bounds))]
    
    def _start_ini(inst, rules, parameters, lower_bounds, upper_bounds, solve, prep_config):
        # infeasible solution given random parameters
        A = _random_action(lower_bounds, upper_bounds)
        A, min_perm, cur_dis, avg_dis, min_conf, elapsed_time, features = solve(inst,
                                                                                rules,
                                                                                A,
                                                                                parameters["num_run"],
                                                                                parameters["num_iteration"][0],
                                                                                prep_config,)
        return A, min_perm, cur_dis, avg_dis, min_conf, elapsed_time, features
    
    # instance
    print("\n{}".format(Path(fileName).stem))

    # eval
    cur_conf = None
    cur_cost = float("inf")
    best_iter = 0
    best_perm = None
    best_param = None

    # time
    total_elapsed = 0
    
    # store conf
    confs = []
    
    # parameters bounds
    lower_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['min']]
    upper_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['max']]
    
    # start from random parameters
    A = _random_action(lower_bounds, upper_bounds)
    A, min_perm, min_cost, avg_cost, min_conf, elapsed_time, features = solve(inst,
                                                            rules,
                                                            A,
                                                            parameters["num_run"],
                                                            parameters["num_iteration"][0],
                                                            prep_config,)
    mixed_cost = (1 - parameters["mixed"]) * min_cost + parameters["mixed"] * avg_cost
    
    # after random initialization
    cur_cost = min_cost
    best_cost = min_cost
    if min_cost == float("inf"):
        best_mixed = min_cost
    else:
        best_mixed = mixed_cost
    best_perm = min_perm
    cur_conf = min_conf
    best_param = A
        
    save_solution(
        fileName,
        -1,
        A,
        best_perm,
        best_cost,
        best_mixed,
        elapsed_time,
        parameters,
        bounds,
    )
    
    # start randomer
    for epoch in range(max(1, parameters["max_evals"])):
        A = np.random.uniform(parameters["random_coefficients"][0] if cur_cost != float('inf') else 1, parameters["random_coefficients"][1]) * np.array(A)
        A, min_perm, min_cost, avg_cost, min_conf, elapsed_time, features = solve(inst,
                                                                                  rules,
                                                                                  A,
                                                                                  parameters["num_run"],
                                                                                  parameters["num_iteration"][0],
                                                                                  prep_config,
                                                                                  cur_conf)
        
        cur_cost = min_cost
        cur_conf = min_conf
        confs.append(min_conf)
        total_elapsed += elapsed_time
        mixed_cost = (1 - parameters["mixed"]) * min_cost + parameters["mixed"] * avg_cost
        best_mixed = min(best_mixed, mixed_cost)
        save_solution(fileName, epoch, A, 
                      min_perm, min_cost, mixed_cost, 
                      elapsed_time, parameters, bounds,)
        
        if min_cost < best_cost:
            best_cost = min_cost
            best_perm = min_perm
            best_iter = epoch
            best_param = A
    
    return (best_perm,
            best_cost,
            best_mixed,
            best_param,
            best_iter,
            total_elapsed)
    
### proportional tuner
def pop_prop(fileName, inst, rules, bounds, parameters, solve, prep_config):

    def _random_para(lower_bounds, upper_bounds):
        # return random parameters
        return [round(np.random.uniform(lower_bounds[i], upper_bounds[i]), 2) for i in range(len(lower_bounds))]
    
    def _start_ini(inst, rules, parameters, lower_bounds, upper_bounds, solve, prep_config):
        if parameters["type"] == "decrease":
            # start from a feasible solution
            cost = float('inf')
            while cost == float('inf'):
                para = _random_para(lower_bounds, upper_bounds)
                para, dec, cost, avg_cost, conf, elapsed_time, features = solve(inst,
                                                                            rules,
                                                                            para,
                                                                            parameters["num_run"],
                                                                            parameters["num_iteration"][0],
                                                                            prep_config,)
        else:
            para = _random_para(lower_bounds, upper_bounds)
            para, dec, cost, avg_cost, conf, elapsed_time, features = solve(inst,
                                                                        rules,
                                                                        para,
                                                                        parameters["num_run"],
                                                                        parameters["num_iteration"][0],
                                                                        prep_config,)
        return para, dec, cost, avg_cost, conf, elapsed_time, features
    
    def _update_para(para, cost, parameters, lower_bounds, upper_bounds):
        if parameters["type"] == "decrease":
            return np.clip(np.array(para) * np.random.uniform(parameters["update_coefficients"][0], 
                           parameters["update_coefficients"][1]),
                           lower_bounds, 
                           upper_bounds)
        if parameters["type"] == "increase":
            return np.array(para) * np.random.uniform(parameters["update_coefficients"][0], 
                           parameters["update_coefficients"][1])
        # random
        if parameters["clip"]:
            if np.array_equal(np.array(para), lower_bounds):
                return np.clip(np.array(para) * np.random.uniform(1, parameters["update_coefficients"][1], np.array(para).shape[0]),
                       lower_bounds,
                       upper_bounds)
            elif np.array_equal(np.array(para), upper_bounds):
                return np.clip(np.array(para) * np.random.uniform(parameters["update_coefficients"][0], 1, np.array(para).shape[0]),
                       lower_bounds,
                       upper_bounds)
            return np.clip(np.array(para) * np.random.uniform(parameters["update_coefficients"][0] if cost != float('inf') else 1, parameters["update_coefficients"][1]),
                        lower_bounds,
                        upper_bounds)
        return np.array(para) * np.random.uniform(parameters["update_coefficients"][0] if cost != float('inf') else 1, parameters["update_coefficients"][1])
    
    def _get_neighors_conf(costs, confs, parameters):
        if parameters["topo"] == "ring" and parameters["population_size"] > 1:
            neighbor_confs = [None] * len(costs)
            for i in range(len(costs)):
                # two connected neighbors and self
                min_index = np.argmin([costs[i - 1], costs[i], costs[(i + 1) % len(costs)]]) 
                neighbor_confs[i] = [confs[i - 1], confs[i], confs[(i + 1) % len(confs)]][min_index]
            return neighbor_confs
        # parameters["topo"] == "complete"
        return confs
    
    # instance
    print("\n{}".format(Path(fileName).stem))

    # makespan
    costs = [float("inf")] * parameters["population_size"]
    # mixed with min and avg
    mixed_costs = [float("inf")] * parameters["population_size"]
    # decisions
    decs = [None] * parameters["population_size"]
    # confs
    confs = [None] * parameters["population_size"]
    # neighbor_confs
    neighbor_confs = []
    # paras
    paras = [None] * parameters["population_size"]

    # eval
    best_iter = 0
    best_dec = None
    best_conf = None
    best_para = None
    min_cost = float('inf')
    min_mixed = float('inf')
    total_elapsed = 0
    
    # parameters bounds
    lower_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['min']]
    upper_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['max']]
    
    # start from random parameters
    thrs = []
    que = queue.Queue()
    for idx in range(parameters["population_size"]):
        thr = Thread(
            target=lambda q, arg: q.put((idx, *_start_ini(*arg))),
            args=(
                que,
                (inst, rules, parameters, lower_bounds, upper_bounds, solve, prep_config),
            ),
        )
        thrs.append(thr)
        thr.start()
    for thr in thrs:
        thr.join()

    while not que.empty():
        idx, para, dec, cost, avg_cost, conf, elapsed_time, features = que.get()
        mixed_cost = (1 - parameters["mixed"]) * cost + parameters["mixed"] * avg_cost
        min_mixed = min(min_mixed, mixed_cost)
        if cost < min_cost:
            min_cost = cost
            best_para = para
            best_conf = conf
            best_dec = dec
        
        paras[idx] = para
        decs[idx] = dec
        costs[idx] = cost
        mixed_costs = mixed_cost
        confs[idx] = conf
        
        save_pop_solution(
            fileName,
            -1,
            idx,
            para,
            dec,
            cost,
            mixed_cost,
            elapsed_time,
            parameters,
            bounds,
        )
    print("\nInitialization finished")
    print(f"Best solution found after initialization: {min_cost}")
    
    # update the best neighbors configuration    
    best_neighbor_confs = _get_neighors_conf(costs, confs, parameters)
        
    for noi in range(max(1, parameters["max_evals"])):
        print("\nIteration {}".format(noi + 1))
        thrs = []
        que = queue.Queue()
        for idx in range(parameters["population_size"]):
            # speed up the tuning
            if parameters["type"] == "decrease" and costs[idx] == float('inf'):
                continue
                
            thr = Thread(
                target=lambda q, arg: q.put((idx, *solve(*arg))),
                args=(
                    que,
                    (inst,
                    rules,
                    _update_para(paras[idx], costs[idx], parameters, lower_bounds, upper_bounds),
                    parameters["num_run"],
                    parameters["num_iteration"][noi + 1],
                    prep_config,
                    confs[idx] if (noi + 1) % parameters["update_interval"] != 0 else neighbor_confs[idx])
                ),
            )
            thrs.append(thr)
            thr.start()
        for thr in thrs:
            thr.join()

        while not que.empty():
            idx, para, dec, cost, avg_cost, conf, elapsed_time, features = que.get()
            mixed_cost = (1 - parameters["mixed"]) * cost + parameters["mixed"] * avg_cost
            min_mixed = min(min_mixed, mixed_cost)
            if cost < min_cost:
                min_cost = cost
                best_para = para
                best_conf = conf
                best_dec = dec
                best_iter = noi
            
            paras[idx] = para
            decs[idx] = dec
            costs[idx] = cost
            mixed_costs = mixed_cost
            # only feasible solution can be the initial state
            if parameters.get("feasible") is None or not parameters["feasible"]:
                confs[idx] = conf
            if parameters["feasible"] and cost != float('inf'):
                confs[idx] = conf
            save_pop_solution(
                fileName,
                noi,
                idx,
                para,
                dec,
                cost,
                mixed_cost,
                elapsed_time,
                parameters,
                bounds,
            )
        print(f"Best solution found after Iteration {noi + 1}: {min_cost}")   
         
        # stop criterion
        if parameters["type"] == "decrease" and len(set(costs)) == 1 and costs[0] == float('inf'):
            break
        
        # update the best neighbors configuration
        neighbor_confs = _get_neighors_conf(costs, confs, parameters)
        
        total_elapsed += elapsed_time
    
    return (best_dec,
            min_cost,
            min_mixed,
            best_para,
            best_iter,
            total_elapsed)
    
### pso
def pso_ini(fileName, inst, rules, bounds, parameters, solve, prep_config): # modified from pso_ini_state

    print("\n{}".format(Path(fileName).stem))
    
    lower_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['min']]
    upper_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['max']]

    total_elapsed = 0

    # create topology
    deg_sequence = get_deg_sequence(parameters["population_size"], parameters["topo"])
    if len(deg_sequence) != 0:
        g = create_graph(parameters["population_size"], deg_sequence)
        adj = nx.to_numpy_matrix(g)
    else:
        adj = np.ones((parameters["population_size"], parameters["population_size"]))
        np.fill_diagonal(adj, 0)
    neighbor = get_neighbor(adj, U=parameters["U"])

    # initialization
    if not parameters["ini"]:
        pos_arr = np.concatenate([np.random.uniform(lower_bounds[i], upper_bounds[i], parameters["population_size"]).reshape(parameters["population_size"], 1) for i in range(len(bounds["min"]))], axis=1).astype(float)
    else:
        pos_arr = np.concatenate([np.linspace(lower_bounds[i], upper_bounds[i], parameters["population_size"]).reshape(parameters["population_size"], 1) for i in range(len(bounds["min"]))], axis=1).astype(float)
    best_pos_arr = pos_arr.copy()
    vel_arr = np.concatenate([np.random.uniform(np.array(lower_bounds)[i] - np.array(upper_bounds)[i], np.array(upper_bounds)[i] - np.array(lower_bounds)[i], parameters["population_size"]).reshape(parameters["population_size"], 1) for i in range(len(bounds["min"]))], axis=1).astype(float)

    mixed_arr = np.zeros((parameters["population_size"])).astype(float)
    cost_arr = np.zeros((parameters["population_size"])).astype(float)  # array for storing the minimum cost
    conf_arr = [None] * parameters["population_size"]
    min_mixed = float("inf")
    min_cost = float("inf")
    min_dec = None
    min_pos = None
    best_iter = -1
    lbd = -3  # lambda for the regularization term
    lbd_coef = parameters["reg_coef"]
    lbd_flag = 0

    # get initial values
    thrs = []
    que = queue.Queue()
    for idx, pos in enumerate(pos_arr):
        thr = Thread(
            target=lambda q, arg: q.put((idx, *solve(*arg))),
            args=(
                que,
                (
                    inst,
                    rules,
                    pos,
                    parameters['num_run'],
                    parameters['num_iteration'][0],
                    prep_config
                ),
            ),
        )
        thrs.append(thr)
        thr.start()
    for thr in thrs:
        thr.join()

    iter_elapsed = 0
    while not que.empty():
        idx, pos, dec, cost, avg_cost, conf, elapsed_time, features = que.get()
        conf_arr[idx] = conf
        iter_elapsed = max(iter_elapsed, elapsed_time)
        if cost == float("inf") or avg_cost == float("inf"):
            mixed = float("inf")
        else:
            if not lbd_flag:
                lbd = min(
                    lbd,
                    -int(
                        math.log(
                            max(1, np.abs((1 - parameters["mixed"]) * cost
                            + parameters["mixed"] * avg_cost)),
                            10,
                        )
                    ),
                )
                lbd_flag = 1
                print("lambda: {}".format(lbd))
            mixed = (
                (1 - parameters["mixed"]) * cost
                + parameters["mixed"] * avg_cost
                + parameters["reg"] * 10 ** lbd * LA.norm(lbd_coef * pos) ** 2
            )
        if cost < min_cost:
            min_cost = cost
            min_dec = dec

        save_pop_solution(
            fileName,
            -1,
            idx,
            pos,
            dec,
            cost,
            mixed,
            elapsed_time,
            parameters,
            bounds,
            )
        if mixed < min_mixed:
            min_pos = pos
            min_mixed = mixed
        mixed_arr[idx] = mixed
        cost_arr[idx] = cost

    mixed_arr[np.isinf(mixed_arr)] = 10 ** -(lbd - 1)
    cost_arr[np.isinf(cost_arr)] = 10 ** -(lbd - 1)

    # conf_arr = conf_arr.copy()
    # conf_arr = get_min_conf_all(conf_arr, mixed_arr)
    conf_arr = get_min_conf(conf_arr, mixed_arr, neighbor)
    # for aaai 2021 only
    conf_arr = [None] * parameters["population_size"]

    save_epoch(
        fileName,
        -1,
        min_cost,
        min_cost, # minimum of the current iteration
        min_mixed,
        min_pos,
        iter_elapsed,
        parameters,
        bounds,
    )
    print("\nInitialization finished")
    print(f"Best solution found after initialization: {min_cost}")
    print(f"Best mixed solution found after initialization: {min_mixed}")
    for i in range(len(bounds['min'])):
        print(f"Best {chr(65 + i)} found: {min_pos[i]}")
        
    # start optimization
    for noi in range(parameters["max_evals"]):
        print(f"\nIteration {noi + 1} start:")
        start = time.time()
        vel_arr = (
            round(
                parameters["w"]
                + (parameters["w0"] - parameters["w"])
                * (1 - (noi + 1) / parameters["max_evals"]),
                2,
            )
            * vel_arr
            + round(
                parameters["c1"]
                + (parameters["c10"] - parameters["c1"])
                * (1 - (noi + 1) / parameters["max_evals"]),
                2,
            )
            * np.random.uniform(0, 1, len(bounds['min']))
            * (best_pos_arr - pos_arr)
            + round(
                parameters["c2"]
                + (parameters["c20"] - parameters["c2"])
                * (1 - (noi + 1) / parameters["max_evals"]),
                2,
            )
            * np.random.uniform(0, 1, len(bounds['min']))
            * (get_min_pos(best_pos_arr, mixed_arr, neighbor) - pos_arr)
        )

        pos_arr = np.clip(pos_arr + vel_arr, lower_bounds, upper_bounds).astype(float)
        
        iter_min = float("inf")
        thrs = []
        que = queue.Queue()
        for idx, pos in enumerate(pos_arr):
            thr = Thread(
                target=lambda q, arg: q.put((idx, *solve(*arg))),
                args=(
                    que,
                    (inst,
                    rules,
                    pos,
                    parameters['num_run'],
                    parameters['num_iteration'][noi + 1],
                    prep_config,
                    conf_arr[idx],
                    ),
                ),
            )
            thrs.append(thr)
            thr.start()
        for thr in thrs:
            thr.join()

        iter_elapsed = 0
        while not que.empty():
            idx, pos, dec, cost, avg_cost, conf, elapsed_time, features = que.get()
            iter_elapsed = max(elapsed_time, iter_elapsed)
            if cost == float("inf") or avg_cost == float("inf"):
                mixed = float("inf")
            else:
                mixed = (
                    (1 - parameters["mixed"]) * cost
                    + parameters["mixed"] * avg_cost
                    + parameters["reg"] * 10 ** lbd * LA.norm(lbd_coef * pos) ** 2
                )
            cost_arr[idx] = cost
            if cost < min_cost:
                min_cost = cost
                min_dec = dec
            iter_min = min(cost, iter_min)
            save_pop_solution(
                fileName,
                noi,
                idx,
                pos,
                dec,
                cost,
                mixed,
                elapsed_time,
                parameters,
                bounds,
            )
            if mixed < mixed_arr[idx]:
                best_pos_arr[idx] = pos
                mixed_arr[idx] = mixed
                # TODO: only update best conf if solution improved
                conf_arr[idx] = conf
                ratio = max(min_mixed / mixed_arr[idx], mixed_arr[idx] / min_mixed) if min_mixed > 0 else min(min_mixed / mixed_arr[idx], mixed_arr[idx] / min_mixed)
                # print(ratio)
                # print(mixed_arr[idx] < min_mixed)
                # print(cost_arr[idx] <= ratio * min_cost)
                if (mixed_arr[idx] < min_mixed and cost_arr[idx] <= ratio * min_cost):  # some relaxation is allowed
                    min_pos = best_pos_arr[idx].copy()
                    min_mixed = mixed_arr[idx].copy()
                    best_iter = noi
                    
        # best conf
        mixed_arr[np.isinf(mixed_arr)] = 10 ** -(lbd - 1)
        cost_arr[np.isinf(cost_arr)] = 10 ** -(lbd - 1)

        # conf_arr = conf_arr.copy()
        # conf_arr = get_min_conf_all(conf_arr, mixed_arr)
        conf_arr = get_min_conf(conf_arr, mixed_arr, neighbor)
        # for aaai 2021 only
        conf_arr = [None] * parameters["population_size"]

        # save info about every epoch
        save_epoch(
            fileName,
            noi,
            min_cost,
            iter_min,
            min_mixed,
            min_pos,
            iter_elapsed,
            parameters,
            bounds,
        )
        total_elapsed += iter_elapsed
        # print out the current best
        print("{}".format(Path(fileName).stem))
        print(f"Best solution found at Iteration {noi + 1}: {iter_min}")
        print(f"Best solution found after Iteration {noi + 1}: {min_cost}")
        print(f"Best mixed solution found after Iteration {noi + 1}: {min_mixed}")
        for i in range(len(bounds['min'])):
            print(f"Best {chr(65 + i)} found: {min_pos[i]}")
    return min_dec, min_cost, min_mixed, min_pos, best_iter, total_elapsed

### proportional tuner modified
# update partial of the parameters
# keep the minimum value unchanged
def pop_prop_par(fileName, inst, rules, bounds, parameters, solve, prep_config):

    def _random_para(lower_bounds, upper_bounds):
        # return random parameters
        return [round(np.random.uniform(lower_bounds[i], upper_bounds[i]), 2) for i in range(len(lower_bounds))]
    
    def _start_ini(inst, rules, parameters, lower_bounds, upper_bounds, solve, prep_config):
        if parameters["type"] == "decrease":
            # start from a feasible solution
            cost = float('inf')
            while cost == float('inf'):
                para = _random_para(lower_bounds, upper_bounds)
                para, dec, cost, avg_cost, conf, elapsed_time, features = solve(inst,
                                                                            rules,
                                                                            para,
                                                                            parameters["num_run"],
                                                                            parameters["num_iteration"][0],
                                                                            prep_config,)
        else:
            para = _random_para(lower_bounds, upper_bounds)
            para, dec, cost, avg_cost, conf, elapsed_time, features = solve(inst,
                                                                        rules,
                                                                        para,
                                                                        parameters["num_run"],
                                                                        parameters["num_iteration"][0],
                                                                        prep_config,)
        return para, dec, cost, avg_cost, conf, elapsed_time, features
    
    def _update_para(para, cost, parameters, lower_bounds, upper_bounds):
        min_idx = np.argmin(para)
        para_num = len(lower_bounds) # para.size
        if parameters["type"] == "decrease":
            coeff = np.random.uniform(parameters["update_coefficients"][0], 
                           parameters["update_coefficients"][1])
            coeff = [coeff] * para_num
            coeff[min_idx] = 1
            return np.clip(np.array(para) * coeff, lower_bounds, upper_bounds)
        if parameters["type"] == "increase":
            coeff = np.random.uniform(parameters["update_coefficients"][0], 
                           parameters["update_coefficients"][1])
            coeff = [coeff] * para_num
            coeff[min_idx] = 1
            return np.array(para) * coeff
        # random
        if parameters["clip"]:
            if np.array_equal(np.array(para), lower_bounds):
                coeff = np.random.uniform(1, parameters["update_coefficients"][1], np.array(para).shape[0])
                coeff = [coeff] * para_num
                coeff[min_idx] = 1
                return np.clip(np.array(para) * coeff, lower_bounds, upper_bounds)
            elif np.array_equal(np.array(para), upper_bounds):
                coeff = np.random.uniform(parameters["update_coefficients"][0], 1, np.array(para).shape[0])
                coeff = [coeff] * para_num
                coeff[min_idx] = 1
                return np.clip(np.array(para) * coeff, lower_bounds, upper_bounds)
            coeff = np.random.uniform(parameters["update_coefficients"][0] if cost != float('inf') else 1, parameters["update_coefficients"][1])
            coeff = [coeff] * para_num
            coeff[min_idx] = 1
            return np.clip(np.array(para) * coeff, lower_bounds, upper_bounds)
        coeff = np.random.uniform(parameters["update_coefficients"][0] if cost != float('inf') else 1, parameters["update_coefficients"][1])
        coeff = [coeff] * para_num
        coeff[min_idx] = 1
        return np.array(para) * coeff
    
    def _get_neighors_conf(costs, confs, parameters):
        if parameters["topo"] == "ring" and parameters["population_size"] > 1:
            neighbor_confs = [None] * len(costs)
            for i in range(len(costs)):
                # two connected neighbors and self
                min_index = np.argmin([costs[i - 1], costs[i], costs[(i + 1) % len(costs)]]) 
                neighbor_confs[i] = [confs[i - 1], confs[i], confs[(i + 1) % len(confs)]][min_index]
            return neighbor_confs
        # parameters["topo"] == "complete"
        return confs
    
    # instance
    print("\n{}".format(Path(fileName).stem))

    # makespan
    costs = [float("inf")] * parameters["population_size"]
    # mixed with min and avg
    mixed_costs = [float("inf")] * parameters["population_size"]
    # decisions
    decs = [None] * parameters["population_size"]
    # confs
    confs = [None] * parameters["population_size"]
    # neighbor_confs
    neighbor_confs = []
    # paras
    paras = [None] * parameters["population_size"]

    # eval
    best_iter = 0
    best_dec = None
    best_conf = None
    best_para = None
    min_cost = float('inf')
    min_mixed = float('inf')
    total_elapsed = 0
    
    # parameters bounds
    lower_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['min']]
    upper_bounds = [x * getattr(inst, parameters["base"]) for x in bounds['max']]
    
    # start from random parameters
    thrs = []
    que = queue.Queue()
    for idx in range(parameters["population_size"]):
        thr = Thread(
            target=lambda q, arg: q.put((idx, *_start_ini(*arg))),
            args=(
                que,
                (inst, rules, parameters, lower_bounds, upper_bounds, solve, prep_config),
            ),
        )
        thrs.append(thr)
        thr.start()
    for thr in thrs:
        thr.join()

    while not que.empty():
        idx, para, dec, cost, avg_cost, conf, elapsed_time, features = que.get()
        mixed_cost = (1 - parameters["mixed"]) * cost + parameters["mixed"] * avg_cost
        min_mixed = min(min_mixed, mixed_cost)
        if cost < min_cost:
            min_cost = cost
            best_para = para
            best_conf = conf
            best_dec = dec
        
        paras[idx] = para
        decs[idx] = dec
        costs[idx] = cost
        mixed_costs = mixed_cost
        confs[idx] = conf
        
        save_pop_solution(
            fileName,
            -1,
            idx,
            para,
            dec,
            cost,
            mixed_cost,
            elapsed_time,
            parameters,
            bounds,
        )
    print("\nInitialization finished")
    print(f"Best solution found after initialization: {min_cost}")
    
    # update the best neighbors configuration    
    best_neighbor_confs = _get_neighors_conf(costs, confs, parameters)
        
    for noi in range(max(1, parameters["max_evals"])):
        print("\nIteration {}".format(noi + 1))
        thrs = []
        que = queue.Queue()
        for idx in range(parameters["population_size"]):
            # speed up the tuning
            if parameters["type"] == "decrease" and costs[idx] == float('inf'):
                continue
                
            thr = Thread(
                target=lambda q, arg: q.put((idx, *solve(*arg))),
                args=(
                    que,
                    (inst,
                    rules,
                    _update_para(paras[idx], costs[idx], parameters, lower_bounds, upper_bounds),
                    parameters["num_run"],
                    parameters["num_iteration"][noi + 1],
                    prep_config,
                    confs[idx] if (noi + 1) % parameters["update_interval"] != 0 else neighbor_confs[idx])
                ),
            )
            thrs.append(thr)
            thr.start()
        for thr in thrs:
            thr.join()

        while not que.empty():
            idx, para, dec, cost, avg_cost, conf, elapsed_time, features = que.get()
            mixed_cost = (1 - parameters["mixed"]) * cost + parameters["mixed"] * avg_cost
            min_mixed = min(min_mixed, mixed_cost)
            if cost < min_cost:
                min_cost = cost
                best_para = para
                best_conf = conf
                best_dec = dec
                best_iter = noi
            
            paras[idx] = para
            decs[idx] = dec
            costs[idx] = cost
            mixed_costs = mixed_cost
            # only feasible solution can be the initial state
            if parameters.get("feasible") is None or not parameters["feasible"]:
                confs[idx] = conf
            if parameters["feasible"] and cost != float('inf'):
                confs[idx] = conf
            save_pop_solution(
                fileName,
                noi,
                idx,
                para,
                dec,
                cost,
                mixed_cost,
                elapsed_time,
                parameters,
                bounds,
            )
        print(f"Best solution found after Iteration {noi + 1}: {min_cost}")   
         
        # stop criterion
        if parameters["type"] == "decrease" and len(set(costs)) == 1 and costs[0] == float('inf'):
            break
        
        # update the best neighbors configuration
        neighbor_confs = _get_neighors_conf(costs, confs, parameters)
        
        total_elapsed += elapsed_time
    
    return (best_dec,
            min_cost,
            min_mixed,
            best_para,
            best_iter,
            total_elapsed)