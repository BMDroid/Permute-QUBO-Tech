import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import joblib
import math
import networkx as nx
import numpy as np
import optuna
import queue
import random
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from collections import defaultdict
from heapq import heappush, nsmallest
from hyperopt import fmin, tpe, hp, partial, STATUS_OK, space_eval, Trials
from numpy import linalg as LA
from pathlib import Path
from sklearn.cluster import KMeans, SpectralClustering
from threading import Thread
from tqdm import tqdm_notebook as tqdm
from solver import *


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
        flag = (deg_seq
                == actual_degrees) and (nx.number_connected_components(g) == 1)
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


def is_tabu(clustering, clustering_means, clustering_var, cluster_order,
            pos_arr):
    y_pred = clustering.fit_predict(pos_arr)
    tabu_pos_arr = (y_pred == any(cluster_order[1:])).astype(
        int)  # only the pos in best cluster will be evaluated
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
    return clustering, clustering_means, clustering_var, clustering_means.argsort(
    )


def get_k_nearest_neighbor(pos, pos_arr_his, val_arr_his, k=3):
    heap = []
    for i, his_pos in enumerate(pos_arr_his):
        dis = np.linalg.norm(pos - his_pos)
        heappush(heap, (dis, val_arr_his[i]))
    k_nearest_neighbors = nsmallest(k, heap)
    k_nearest_neighbors_dis = [item[1] for item in k_nearest_neighbors]
    if k_nearest_neighbors_dis.count(float("inf")) >= k / 2:
        return float("inf")
    return sum(
        list(filter(lambda a: a != float("inf"), k_nearest_neighbors_dis))) / (
            k - k_nearest_neighbors_dis.count(float("inf")))


def pso_clustering(fileName, inst, rules, bounds, parameters, solve):

    print("\n{}".format(Path(fileName).stem))
    print("PSO initialization start ...\n")
    total_elapsed = 0

    # create topology
    deg_sequence = get_deg_sequence(parameters["particles"],
                                    parameters["topo"])
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
                    bounds["min"][i] * getattr(inst, parameters['base']),
                    bounds["max"][i] * getattr(inst, parameters['base']),
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
                    bounds["min"][i] * getattr(inst, parameters['base']),
                    bounds["max"][i] * getattr(inst, parameters['base']),
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
                bounds["min"][i] * getattr(inst, parameters['base']) -
                bounds["max"][i] * getattr(inst, parameters['base']),
                bounds["max"][i] * getattr(inst, parameters['base']) -
                bounds["min"][i] * getattr(inst, parameters['base']),
                parameters["particles"],
            ).reshape(parameters["particles"], 1)
            for i in range(len(bounds["min"]))
        ],
        axis=1,
    ).astype(float)
    val_arr = np.zeros((parameters["particles"])).astype(float)
    sol_arr = np.zeros((parameters["particles"])).astype(
        float)  # array for storing the minimum solution
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
                            (1 - parameters["mixed"]) * min_cost +
                            parameters["mixed"] * avg_cost,
                            10,
                        )),
                )
                lbd_flag = 1
                print("lambda: {}".format(lbd))
            val = ((1 - parameters["mixed"]) * min_cost +
                   parameters["mixed"] * avg_cost +
                   parameters['reg'] * 10**lbd * LA.norm(lbd_coef * pos)**2)
            sol = min_cost
        # cur_cost = min(sol, cur_cost)
        if sol < cur_cost:
            cur_cost = sol
            cur_perm = min_perm
        save_solution(fileName, -1, pos, min_perm, min_cost, val, elapsed_time,
                      parameters)
        if val < cur_mixed:
            min_pos = pos
            cur_mixed = val
        val_arr[idx] = val
        sol_arr[idx] = sol
    total_elapsed += epoch_elapsed
    save_epoch(fileName, -1, cur_cost, cur_cost, cur_mixed, min_pos,
               epoch_elapsed, parameters)
    print("\nInitialization finished")
    print(f"Best solution found after initialization: {cur_cost}")
    print(f"Best mixed solution found after initialization: {cur_mixed}")
    for i in range(len(bounds["min"])):
        print(f"Best {chr(65 + i)} found after initialization: {min_pos[i]}")

    # record all the particles and their values
    pos_arr_his = pos_arr.copy()
    val_arr_his = val_arr.copy()
    sol_arr_his = sol_arr.copy()
    tabu_pos_arr = np.zeros((parameters["particles"], )).astype(float)
    y_pred = np.zeros((parameters["particles"], ))
    clustering, clustering_means, clustering_var, cluster_order = None, None, None, None

    # start optimization
    for e in range(parameters["epoch"]):
        print(f"\nEpoch {e + 1} start:")
        vel_arr = (round(
            parameters["w"] + (parameters["w0"] - parameters["w"]) *
            (1 - (e + 1) / parameters["epoch"]),
            2,
        ) * vel_arr + round(
            parameters["c1"] + (parameters["c10"] - parameters["c1"]) *
            (1 - (e + 1) / parameters["epoch"]),
            2,
        ) * np.random.uniform(0, 1, len(bounds["min"])) *
                   (best_pos_arr - pos_arr) + round(
                       parameters["c2"] +
                       (parameters["c20"] - parameters["c2"]) *
                       (1 - (e + 1) / parameters["epoch"]),
                       2,
                   ) * np.random.uniform(0, 1, len(bounds["min"])) *
                   (get_min_pos(pos_arr, val_arr, neighbor) - pos_arr))
        pos_arr = np.concatenate(
            [
                np.array([
                    min(
                        bounds["max"][j] * getattr(inst, parameters['base']),
                        max(
                            bounds["min"][j] *
                            getattr(inst, parameters['base']),
                            pos_arr[i][j] + vel_arr[i][j],
                        ),
                    ) for i in range(parameters["particles"])
                ]).reshape(parameters["particles"], 1)
                for j in range(len(bounds["min"]))
            ],
            axis=1,
        ).astype(float)
        if e > 0:
            y_pred, tabu_pos_arr = is_tabu(clustering, clustering_means,
                                           clustering_var, cluster_order,
                                           pos_arr)

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
                        ),
                    ),
                )
                thrs.append(thr)
                thr.start()
            else:
                sol = get_k_nearest_neighbor(pos, pos_arr_his, sol_arr_his)
                val = get_k_nearest_neighbor(
                    pos, pos_arr_his,
                    val_arr_his)  # average of k nearest neighbor
                sol_arr[idx] = sol
                # cur_cost = min(sol, cur_cost)
                if sol < cur_cost:
                    cur_cost = sol
                    cur_perm = min_perm
                epoch_min = min(sol, epoch_min)
                if val < val_arr[idx]:
                    best_pos_arr[idx] = pos
                    val_arr[idx] = val
                    ratio = max(cur_mixed / val_arr[idx],
                                val_arr[idx] / cur_mixed)
                    if (val_arr[idx] < cur_mixed and sol_arr[idx] <=
                            ratio * cur_cost):  # some relaxation is allowed
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
                val = ((1 - parameters["mixed"]) * min_cost +
                       parameters["mixed"] * avg_cost +
                       parameters['reg'] * 10**lbd * LA.norm(lbd_coef * pos)**2)
                sol = min_cost
            sol_arr[idx] = sol
            # cur_cost = min(sol, cur_cost)
            if sol < cur_cost:
                cur_cost = sol
                cur_perm = min_perm
            epoch_min = min(sol, epoch_min)
            save_solution(fileName, e, pos, min_perm, min_cost, val,
                          elapsed_time, parameters)
            if val < val_arr[idx]:
                best_pos_arr[idx] = pos
                val_arr[idx] = val
                #                 print("min_pos: {}".format(str(min_pos)))
                ratio = max(cur_mixed / val_arr[idx], val_arr[idx] / cur_mixed)
                if (val_arr[idx] < cur_mixed and sol_arr[idx] <=
                        ratio * cur_cost):  # some relaxation is allowed
                    #                     print("val: {}, cur_mixed:{}, min_pos: {}".format(val, cur_mixed, str(min_pos)))
                    min_pos = best_pos_arr[idx].copy()
                    cur_mixed = val_arr[idx].copy()
                    #                     print("val: {}, cur_mixed:{}, min_pos: {}".format(val, cur_mixed, str(min_pos)))
                    best_epoch = e
        # clustering on all the historical particles positions
        pos_arr_his = np.concatenate([pos_arr_his, pos_arr],
                                     axis=0).astype(float)
        val_arr_his = np.concatenate([val_arr_his, val_arr],
                                     axis=0).astype(float)
        sol_arr_his = np.concatenate([sol_arr_his, sol_arr],
                                     axis=0).astype(float)
        clustering, clustering_means, clustering_var, cluster_order = get_clustering(
            pos_arr_his, val_arr_his)

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
        )
        total_elapsed += epoch_elapsed
        # print out the current best
        print("{}".format(Path(fileName).stem))
        print(f"Best solution found at {e + 1} Epoch: {epoch_min}")
        print(f"Best solution found after {e + 1} Epoch: {cur_cost}")
        print(f"Best mixed solution found after {e + 1} Epoch: {cur_mixed}")
        for i in range(len(bounds["min"])):
            print(
                f"Best {chr(65 + i)} found after {e + 1} Epoch: {min_pos[i]}")
    return cur_perm, cur_cost, cur_mixed, min_pos, best_epoch, total_elapsed


def pso(fileName, inst, rules, bounds, parameters, solve):

    print("\n{}".format(Path(fileName).stem))
    print("PSO initialization start ...\n")

    total_elapsed = 0

    # create topology
    deg_sequence = get_deg_sequence(parameters["particles"],
                                    parameters["topo"])
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
                    bounds["min"][i] * getattr(inst, parameters['base']),
                    bounds["max"][i] * getattr(inst, parameters['base']),
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
                    bounds["min"][i] * getattr(inst, parameters['base']),
                    bounds["max"][i] * getattr(inst, parameters['base']),
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
                bounds["min"][i] * getattr(inst, parameters['base']) -
                bounds["max"][i] * getattr(inst, parameters['base']),
                bounds["max"][i] * getattr(inst, parameters['base']) -
                bounds["min"][i] * getattr(inst, parameters['base']),
                parameters["particles"],
            ).reshape(parameters["particles"], 1)
            for i in range(len(bounds["min"]))
        ],
        axis=1,
    ).astype(float)
    val_arr = np.zeros((parameters["particles"])).astype(float)
    sol_arr = np.zeros((parameters["particles"])).astype(
        float)  # array for storing the minimum solution
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
                            (1 - parameters["mixed"]) * min_cost +
                            parameters["mixed"] * avg_cost,
                            10,
                        )),
                )
                lbd_flag = 1
                print("lambda: {}".format(lbd))
            val = ((1 - parameters["mixed"]) * min_cost +
                   parameters["mixed"] * avg_cost +
                   parameters['reg'] * 10**lbd * LA.norm(lbd_coef * pos)**2)
            sol = min_cost
        # cur_cost = min(sol, cur_cost)
        if sol < cur_cost:
            cur_cost = sol
            cur_perm = min_perm
        save_solution(fileName, -1, pos, min_perm, min_cost, val, elapsed_time,
                      parameters)
        if val < cur_mixed:
            min_pos = pos
            cur_mixed = val
        val_arr[idx] = val
        sol_arr[idx] = sol

    save_epoch(fileName, -1, cur_cost, cur_cost, cur_mixed, min_pos,
               epoch_elapsed, parameters)
    print("\nInitialization finished")
    print(f"Best solution found after initialization: {cur_cost}")
    print(f"Best mixed solution found after initialization: {cur_mixed}")
    for i in range(len(bounds["min"])):
        print(f"Best {chr(65 + i)} found initialization: {min_pos[i]}")

    # start optimization
    for e in range(parameters["epoch"]):
        print(f"\nEpoch {e + 1} start:")
        start = time.time()
        vel_arr = (round(
            parameters["w"] + (parameters["w0"] - parameters["w"]) *
            (1 - (e + 1) / parameters["epoch"]),
            2,
        ) * vel_arr + round(
            parameters["c1"] + (parameters["c10"] - parameters["c1"]) *
            (1 - (e + 1) / parameters["epoch"]),
            2,
        ) * np.random.uniform(0, 1, len(bounds["min"])) *
                   (best_pos_arr - pos_arr) + round(
                       parameters["c2"] +
                       (parameters["c20"] - parameters["c2"]) *
                       (1 - (e + 1) / parameters["epoch"]),
                       2,
                   ) * np.random.uniform(0, 1, len(bounds["min"])) *
                   (get_min_pos(pos_arr, val_arr, neighbor) - pos_arr))
        pos_arr = np.concatenate(
            [
                np.array([
                    min(
                        bounds["max"][j] * getattr(inst, parameters['base']),
                        max(
                            bounds["min"][j] *
                            getattr(inst, parameters['base']),
                            pos_arr[i][j] + vel_arr[i][j],
                        ),
                    ) for i in range(parameters["particles"])
                ]).reshape(parameters["particles"], 1)
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
                val = ((1 - parameters["mixed"]) * min_cost +
                       parameters["mixed"] * avg_cost +
                       parameters['reg'] * 10**lbd * LA.norm(lbd_coef * pos)**2)
                sol = min_cost
            sol_arr[idx] = sol
            # cur_cost = min(sol, cur_cost)
            if sol < cur_cost:
                cur_cost = sol
                cur_perm = min_perm
            epoch_min = min(sol, epoch_min)
            save_solution(fileName, e, pos, min_perm, min_cost, val,
                          elapsed_time, parameters)
            if val < val_arr[idx]:
                best_pos_arr[idx] = pos
                val_arr[idx] = val
                ratio = max(cur_mixed / val_arr[idx], val_arr[idx] / cur_mixed)
                if (val_arr[idx] < cur_mixed and sol_arr[idx] <=
                        ratio * cur_cost):  # some relaxation is allowed
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
        )
        total_elapsed += epoch_elapsed
        # print out the current best
        print("{}".format(Path(fileName).stem))
        print(f"Best solution found at {e + 1} Epoch: {epoch_min}")
        print(f"Best solution found after {e + 1} Epoch: {cur_cost}")
        print(f"Best mixed solution found after {e + 1} Epoch: {cur_mixed}")
        for i in range(len(bounds["min"])):
            print(
                f"Best {chr(65 + i)} found after {e + 1} Epoch: {min_pos[i]}")
    return cur_perm, cur_cost, cur_mixed, min_pos, best_epoch, total_elapsed


def hyperopt(fileName, inst, rules, bounds, parameters, solve):
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
        A, min_perm, min_cost, avg_cost, elapsed_time = solve(
            inst,
            rules,
            [x[chr(65 + i)] for i in range(len(bounds['min']))],
            parameters["num_run"],
            parameters["num_iteration"][0],
        )
        total_elapsed += elapsed_time
        mixed_cost = (1 - parameters["mixed"]
                      ) * min_cost + parameters["mixed"] * avg_cost
        cur_mixed = min(cur_mixed, mixed_cost)
        save_solution(fileName, epoch, A, min_perm, min_cost, mixed_cost,
                      elapsed_time, parameters)
        if len(min_perm) == 0:
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
        for i in range(len(bounds['min'])):
            space[chr(i + 65)] = hp.uniform(
                chr(i + 65),
                bounds["min"][i] * getattr(inst, parameters['base']),
                bounds["max"][i] * getattr(inst, parameters['base']),
            )
    else:
        for i in range(len(bounds['min'])):
            space[chr(i + 65)] = hp.uniform(
                chr(i + 65),
                bounds["min"][i] * getattr(inst, parameters['base']),
                bounds["max"][i] * getattr(inst, parameters['base'])
                if i + 1 != len(bounds['min']) else bounds["min"][i] *
                getattr(inst, parameters['base']),
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

    # find best A
    trials = Trials()
    best = fmin(
        lambda x: _objective(x),
        space=space,
        algo=partial(tpe.suggest,
                     n_startup_jobs=min(parameters["max_evals"] // 2, 20)),
        max_evals=parameters["max_evals"],
        trials=trials,
        rstate=None,
        # points_to_evaluate=[{'A':0.75 * getattr(inst, parameters['base']), 'B': 0.75 * getattr(inst, parameters['base'])}]
    )
    return cur_perm, cur_cost, cur_mixed, [
        best[chr(65 + i)] for i in range(len(bounds['min']))
    ], best_epoch, total_elapsed


def optune(fileName, inst, rules, bounds, parameters, solve):
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

        space = {}
        if not parameters["fixed"]:
            for i in range(len(bounds['min'])):
                space[chr(i + 65)] = trial.suggest_uniform(
                    chr(i + 65),
                    bounds["min"][i] * getattr(inst, parameters['base']),
                    bounds["max"][i] * getattr(inst, parameters['base']),
                )
        else:
            for i in range(len(bounds['min'])):
                space[chr(i + 65)] = trial.suggest_uniform(
                    chr(i + 65),
                    bounds["min"][i] * getattr(inst, parameters['base']),
                    bounds["max"][i] * getattr(inst, parameters['base'])
                    if i + 1 != len(bounds['min']) else bounds["min"][i] *
                    getattr(inst, parameters['base']),
                )

        A_, min_perm, min_cost, avg_cost, elapsed_time = solve(
            inst, rules,
            [space[chr(65 + i)] for i in range(len(bounds['min']))],
            parameters["num_run"], parameters["num_iteration"][0])
        total_elapsed += elapsed_time
        mixed_cost = (1 - parameters["mixed"]
                      ) * min_cost + parameters["mixed"] * avg_cost
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
        )
        if len(min_perm) == 0:
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

    # find best A
    study = optuna.create_study()
    optuna.logging.disable_default_handler()
    study.optimize(_objective, n_trials=parameters["max_evals"])
    return (cur_perm, cur_cost, cur_mixed, [
        study.best_params[chr(65 + i)] for i in range(len(bounds['min']))
    ], best_epoch, total_elapsed)


def fixer(fileName, inst, rules, bounds, parameters, solve, ratio=0.75938):
    # instance
    print("\n{}".format(Path(fileName).stem))

    # time
    total_elapsed = 0

    # parameter
    space = {}
    if not parameters["fixed"]:
        for i in range(len(bounds['min'])):
            space[chr(i + 65)] = ratio * getattr(inst, parameters['base'])
    else:
        for i in range(len(bounds['min'])):
            space[chr(i + 65)] = ratio * getattr(
                inst, parameters['base']) if i + 1 != len(
                    bounds['min']) else bounds["min"][i] * getattr(
                        inst, parameters['base'])

    A, min_perm, min_cost, avg_cost, elapsed_time = solve(
        inst, rules, [space[chr(65 + i)] for i in range(len(bounds['min']))],
        parameters["num_run"], parameters["num_iteration"][0])
    total_elapsed += elapsed_time
    mixed_cost = (
        1 - parameters["mixed"]) * min_cost + parameters["mixed"] * avg_cost
    save_solution(
        fileName,
        0,
        A,
        min_perm,
        min_cost,
        mixed_cost,
        elapsed_time,
        parameters,
    )
    return (min_perm, min_cost, mixed_cost,
            [space[chr(65 + i)]
             for i in range(len(bounds['min']))], 0, total_elapsed)


def randomer_seq(fileName,
                 inst,
                 rules,
                 bounds,
                 parameters,
                 solve,
                 dist='uniform'):

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

    for epoch in range(max(1, parameters['max_evals'])):

        space = {}
        if dist == 'uniform':
            if not parameters["fixed"]:
                for i in range(len(bounds['min'])):
                    space[chr(i + 65)] = random.uniform(
                        bounds["min"][i], bounds["max"][i]) * getattr(
                            inst, parameters['base'])
            else:
                for i in range(len(bounds['min'])):
                    space[chr(i + 65)] = random.uniform(
                        bounds["min"][i], bounds["max"][i]) * getattr(
                            inst, parameters['base']) if i + 1 != len(
                                bounds['min']) else bounds["min"][i] * getattr(
                                    inst, parameters['base'])
        elif parameters["dist"] == 'normal':
            mu = 0.7593874498315781
            sigma_2 = 0.0141
            if not parameters["fixed"]:
                for i in range(len(bounds['min'])):
                    space[chr(i + 65)] = random.normal(
                        mu, sqrt(sigma_2)) * getattr(inst, parameters['base'])
            else:
                for i in range(len(bounds['min'])):
                    space[chr(
                        i + 65)] = random.normal(mu, sqrt(sigma_2)) * getattr(
                            inst, parameters['base']) if i + 1 != len(
                                bounds['min']) else bounds["min"][i] * getattr(
                                    inst, parameters['base'])

        A_, min_perm, min_cost, avg_cost, elapsed_time = solve(
            inst, rules,
            [space[chr(65 + i)] for i in range(len(bounds['min']))],
            parameters["num_run"], parameters["num_iteration"][0])
        total_elapsed += elapsed_time
        mixed_cost = (1 - parameters["mixed"]
                      ) * min_cost + parameters["mixed"] * avg_cost
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
        )
        if min_cost < cur_cost:
            cur_cost = min_cost
            cur_perm = min_perm
            best_epoch = epoch
            best_params = space
    return cur_perm, cur_cost, cur_mixed, [
        best_params[chr(65 + i)] for i in range(len(bounds['min']))
    ], best_epoch, total_elapsed


def randomer_thr(fileName,
                 inst,
                 rules,
                 bounds,
                 parameters,
                 solve,
                 dist='uniform'):

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
    for _ in range(max(1, parameters['max_evals'])):

        space = {}
        if dist == 'uniform':
            if not parameters["fixed"]:
                for i in range(len(bounds['min'])):
                    space[chr(i + 65)] = random.uniform(
                        bounds["min"][i], bounds["max"][i]) * getattr(
                            inst, parameters['base'])
            else:
                for i in range(len(bounds['min'])):
                    space[chr(i + 65)] = random.uniform(
                        bounds["min"][i], bounds["max"][i]) * getattr(
                            inst, parameters['base']) if i + 1 != len(
                                bounds['min']) else bounds["min"][i] * getattr(
                                    inst, parameters['base'])
        elif parameters["dist"] == 'normal':
            mu = 0.7593874498315781
            sigma_2 = 0.0141
            if not parameters["fixed"]:
                for i in range(len(bounds['min'])):
                    space[chr(i + 65)] = random.normal(
                        mu, sqrt(sigma_2)) * getattr(inst, parameters['base'])
            else:
                for i in range(len(bounds['min'])):
                    space[chr(
                        i + 65)] = random.normal(mu, sqrt(sigma_2)) * getattr(
                            inst, parameters['base']) if i + 1 != len(
                                bounds['min']) else bounds["min"][i] * getattr(
                                    inst, parameters['base'])
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
                    [space[chr(65 + i)] for i in range(len(bounds['min']))],
                    parameters["num_run"],
                    parameters["num_iteration"][0],
                ),
            ),
        )
        thrs.append(thr)
        thr.start()
    for thr in thrs:
        thr.join()

    while not que.empty():
        idx, A, min_perm, min_cost, avg_cost, elapsed_time = que.get()
        mixed_cost = (1 - parameters["mixed"]
                      ) * min_cost + parameters["mixed"] * avg_cost
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
        )
        if min_cost < cur_cost:
            cur_cost = min_cost
            cur_perm = min_perm
            best_trial = idx
            best_params = A
    total_elapsed = time.time() - start
    return cur_perm, cur_cost, cur_mixed, best_params, best_trial, total_elapsed


def ratio_predictor(fileName, inst, rules, bounds, parameters, solve):
    # instance
    print("\n{}".format(Path(fileName).stem))

    # time
    total_elapsed = 0

    # predict the ratio
    model = joblib.load('./mlp/ratio_nn')
    scaler = joblib.load('./mlp/min_max_scaler')

    features = inst.calculate_features()
    features = scaler.transform(np.array([[8] + list(features)]))
    ratio = model.predict(features)[0][0]

    predicted_parameter = ratio * getattr(
        inst, parameters['base'])  # base is the "_max_edge"

    A, min_perm, min_cost, avg_cost, elapsed_time = solve(
        inst, rules, [predicted_parameter], parameters["num_run"],
        parameters["num_iteration"][0])
    total_elapsed += elapsed_time
    mixed_cost = (
        1 - parameters["mixed"]) * min_cost + parameters["mixed"] * avg_cost
    save_solution(
        fileName,
        0,
        A,
        min_perm,
        min_cost,
        mixed_cost,
        elapsed_time,
        parameters,
    )
    return (min_perm, min_cost, mixed_cost, [predicted_parameter], 0,
            total_elapsed)


def ratio_predictor_thr(fileName, inst, rules, bounds, parameters, solve):
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
    model = joblib.load('./mlp/ratio_nn')
    scaler = joblib.load('./mlp/min_max_scaler')

    features = inst.calculate_features()
    features = scaler.transform(np.array([[8] + list(features)]))
    ratio = model.predict(features)[0][0]

    predicted_parameter = ratio * getattr(
        inst, parameters['base'])  # base is the "_max_edge"

    # list of identical predicted parameters
    parameter_lst = [predicted_parameter] * parameters['max_eval']

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
                ),
            ),
        )
        thrs.append(thr)
        thr.start()
    for thr in thrs:
        thr.join()

    while not que.empty():
        idx, A, min_perm, min_cost, avg_cost, elapsed_time = que.get()
        mixed_cost = (1 - parameters["mixed"]
                      ) * min_cost + parameters["mixed"] * avg_cost
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
        )
        if min_cost < cur_cost:
            cur_cost = min_cost
            cur_perm = min_perm
            best_trial = idx
    total_elapsed = time.time() - start
    return cur_perm, cur_cost, cur_mixed, [predicted_parameter
                                           ], best_trial, total_elapsed
