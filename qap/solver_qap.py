import itertools
import glob
import os
import math
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import optuna
import pandas as pd
import pathlib
import queue
import random
import shutil
import time
import pickle
import neal # simulatedAnnealingSampler
import gc # garbage collection

from celluloid import Camera
from datetime import date, datetime
from matplotlib import cm
from matplotlib.animation import PillowWriter
from pathlib import Path
from python_fjda import fjda_client
from dwave_qbsolv import QBSolv # qbsolv
from threading import Thread
from tqdm import tqdm_notebook as tqdm
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn import preprocessing
from collections import defaultdict

from builder_qap import *
from sequence import *
from tsp import *
from asgmt import *
from max_cut import *
from qap import *
from stitcher import *
from dau import *

param = {
        "offset_mode": 3,
        "offset_inc_rate": 100,
        "tmp_st": 1,
        "tmp_decay": 0,
        "tmp_mode": 0,
        "tmp_interval": 100,
    }

server_list = [
        "dau2-05-0.da.labs.fujitsu.com",
        "dau2-05-1.da.labs.fujitsu.com",
        # "dau2-06-0.da.labs.fujitsu.com",
        # "dau2-06-1.da.labs.fujitsu.com",
    ]

### Max Cut
def solve_avg_max_cut_config(cut, rules, para, num_run, num_iteration, prep_config, conf=None, seed=None):
    global param, server_list

    # prepare the QUBO
    W, V, S, D = cut.W, cut.V, cut.S, cut.D
    qubo = build_cut_QUBO(para, V, rules)
    N_state, bias, constant, weight, s, local_field, E = prep_config(qubo, conf, seed)

    args = {
        "weight": weight,
        "lf_i_n": np.tile(local_field, num_run),
        "state_min_i_n": np.tile(s, num_run),
        "state_cur_i_n": np.tile(s, num_run),
        "eg_min_i_n": np.full(num_run, E, dtype=int),
        "eg_cur_i_n": np.full(num_run, E, dtype=int),
        "num_iteration": num_iteration,
        "num_bit": N_state,
        "num_run": num_run,
        "ope_mode": 2,
    }

    da = fjda_client.fjda(server=random.choice(server_list))

    # send to DA
    da.setAnnealParameterMM(param)

    received = False
    res = da.doAnnealMM(args, format="numpy", compression="lz4")
    while not received:
        if res:
            print(f"Received")
            received = True

    # start time
    start = time.time()
    # derive solution
    min_dec, min_cost, min_conf = None, float("inf"), None
    min_energy = float("inf")
    costs = []
    feasible_flag = False # flag to indicate whether exists a feasible solution
    for idx, conf in enumerate(res["state_min_o_n.numpy"]):
        dec, cost = compute_cut_cost(W, V, S, D, conf)

        if cost < 0: # feasible solution
            feasible_flag = True
            costs.append(cost)
            if cost < min_cost: 
                min_dec, min_cost, min_conf = dec, cost, conf
                min_energy = res["eg_min_o_n.numpy"][idx]
            
        if cost >= 0: # infeasible solution
            energy = res["eg_min_o_n.numpy"][idx]
            if energy < min_energy and not feasible_flag:
                min_conf = conf
                min_energy = energy
    
    # elapsed_time
    elapsed_time = time.time() - start + res["anneal_time"] / 1000
    if min_dec is None:
        return np.array([*para]), None, float("inf"), float("inf"), min_conf, elapsed_time, min_energy
    return np.array([*para]), min_dec, min_cost, np.mean(costs), min_conf, elapsed_time, min_energy

def solve_min_max_cut_config(cut, rules, para, num_run, num_iteration, prep_config, conf=None, seed=None):
    global param, server_list

    # prepare the QUBO
    W, V, S, D = cut.W, cut.V, cut.S, cut.D
    qubo = build_cut_QUBO(para, V, rules)
    N_state, bias, constant, weight, s, local_field, E = prep_config(qubo, conf, seed)

    args = {
        "weight": weight,
        "lf_i_n": np.tile(local_field, num_run),
        "state_min_i_n": np.tile(s, num_run),
        "state_cur_i_n": np.tile(s, num_run),
        "eg_min_i_n": np.full(num_run, E, dtype=int),
        "eg_cur_i_n": np.full(num_run, E, dtype=int),
        "num_iteration": num_iteration,
        "num_bit": N_state,
        "num_run": num_run,
        "ope_mode": 2,
    }

    da = fjda_client.fjda(server=random.choice(server_list))

    # send to DA
    da.setAnnealParameterMM(param)

    received = False
    res = da.doAnnealMM(args, format="numpy", compression="lz4")
    while not received:
        if res:
            print(f"Received")
            received = True

    # start time
    start = time.time()
    # derive solution based on the minimum energy achieved
    min_energy = np.min(res['eg_min_o_n.numpy'])
    min_dec, min_cost, min_conf = None, float("inf"), None
    feasible_flag = False # flag to indicate whether exists a feasible solution
    min_idx = res['eg_min_o_n.numpy'].argmin()
    min_conf = res["state_min_o_n.numpy"][min_idx]
    dec, cost = compute_cut_cost(W, V, S, D, conf)

    if cost < 0 and cost < min_cost: # feasible solution
        feasible_flag = True
        min_dec, min_cost = dec, cost
    
    # elapsed_time
    elapsed_time = time.time() - start + res["anneal_time"] / 1000
    if not feasible_flag:
        return np.array([*para]), None, float("inf"), float("inf"), min_conf, elapsed_time, min_energy
    return np.array([*para]), min_dec, min_cost, min_cost, min_conf, elapsed_time, min_energy

def main_max_cut(dirName, tuner, bounds, parameters, solve, prep_config):
    def _sub(filePath):
        print(filePath)
        nonlocal bounds
        nonlocal parameters

        fileName = Path(filePath).stem
        with open(filePath, "rb") as f:
            cut = pickle.load(f)
        W, V, S, D = cut.W, cut.V, cut.S, cut.D
        rules = dict(weight=build_cut_weight(W, V),
            same=build_cut_same_rule(V, S),
            diff=build_cut_diff_rule(V, D),
        )
        start = time.time()
        dec, cost, mixed_cost, best_para, best_iter, elapsed_time = tuner(
            fileName, cut, rules, bounds, parameters, solve, prep_config
        )
        run_time = time.time() - start
        print(f"\n{fileName}")
        print(f"Best solution found: {cost}")
        for i in range(len(bounds['min'])):
            print(f"Best {chr(65 + i)} found: {best_para[i]}")
        print(f"Elapsed time: {run_time}\n")
        print("*" * 36)

        # record the best solution found
        infoName = parameters["info_path"]
        if not os.path.exists(infoName):
            with open(infoName, "a") as f:
                f.write(
                    "Max_Cut, Best_Iter, "
                    + ", ".join(["Best_" + chr(65 + i) for i in range(len(bounds['min']))])
                    + ", Best_Solution, Best_Mixed, Anneal_Time, Elapsed_Time\n"
                )
        with open(infoName, "a") as f:
            f.write(
                f"{Path(fileName).stem}, {best_iter + 1}, "
                + ", ".join([f"{best_para[i]}" for i in range(len(bounds['min']))])
                + f", {cost}, {mixed_cost}, {elapsed_time}, {run_time}\n"
            )

    pathlib.Path(parameters["folder_path"]).mkdir(parents=True, exist_ok=True)

    for fileName in sorted(glob.glob(dirName)):
        _sub(fileName)

### Constrained Assignment
def solve_avg_cstr_asgmt_config(asgmt, rules, para, num_run, num_iteration, prep_config, conf=None, seed=None):
    """ prep_config is an argument not fixed anymore
    """
    global param, server_list

    # prepare the QUBO
    A, T, weights, extra, pairs = asgmt.A, asgmt.T, asgmt.weights, asgmt.extra, asgmt.pairs
    qubo = build_cstr_asgmt_QUBO(para, A, T, rules)
    N_state, bias, constant, weight, s, local_field, E = prep_config(qubo, conf, seed)

    args = {
        "weight": weight,
        "lf_i_n": np.tile(local_field, num_run),
        "state_min_i_n": np.tile(s, num_run),
        "state_cur_i_n": np.tile(s, num_run),
        "eg_min_i_n": np.full(num_run, E, dtype=int),
        "eg_cur_i_n": np.full(num_run, E, dtype=int),
        "num_iteration": num_iteration,
        "num_bit": N_state,
        "num_run": num_run,
        "ope_mode": 2,
    }

    da = fjda_client.fjda(server=random.choice(server_list))

    # send to DA
    da.setAnnealParameterMM(param)

    received = False
    res = da.doAnnealMM(args, format="numpy", compression="lz4")
    while not received:
        if res:
            print(f"Received")
            received = True

    # start time
    start = time.time()
    # derive solution
    min_dec, min_cost, min_conf = None, float("inf"), None
    min_energy = float("inf")
    costs = []
    feasible_flag = False # flag to indicate whether exists a feasible solution
    for idx, conf in enumerate(res["state_min_o_n.numpy"]):
        dec, cost = compute_cstr_asgmt_cost(A, T, weights, extra, pairs, conf)

        if cost < 0: # feasible solution
            feasible_flag = True
            costs.append(cost)
            if cost < min_cost: 
                min_dec, min_cost, min_conf = dec, cost, conf
                min_energy = res["eg_min_o_n.numpy"][idx]
            
        if cost >= 0: # infeasible solution
            energy = res["eg_min_o_n.numpy"][idx]
            if energy < min_energy and not feasible_flag:
                min_conf = conf
                min_energy = energy
    
    # elapsed_time
    elapsed_time = time.time() - start + res["anneal_time"] / 1000
    if min_dec is None:
        return np.array([*para]), None, float("inf"), float("inf"), min_conf, elapsed_time, min_energy
    return np.array([*para]), min_dec, min_cost, np.mean(costs), min_conf, elapsed_time, min_energy

def solve_min_cstr_asgmt_config(asgmt, rules, para, num_run, num_iteration, prep_config, conf=None, seed=None):
    """ prep_config is an argument not fixed anymore
    """
    global param, server_list

    # prepare the QUBO
    A, T, weights, extra, pairs = asgmt.A, asgmt.T, asgmt.weights, asgmt.extra, asgmt.pairs
    qubo = build_cstr_asgmt_QUBO(para, A, T, rules)
    N_state, bias, constant, weight, s, local_field, E = prep_config(qubo, conf, seed)

    args = {
        "weight": weight,
        "lf_i_n": np.tile(local_field, num_run),
        "state_min_i_n": np.tile(s, num_run),
        "state_cur_i_n": np.tile(s, num_run),
        "eg_min_i_n": np.full(num_run, E, dtype=int),
        "eg_cur_i_n": np.full(num_run, E, dtype=int),
        "num_iteration": num_iteration,
        "num_bit": N_state,
        "num_run": num_run,
        "ope_mode": 2,
    }

    da = fjda_client.fjda(server=random.choice(server_list))

    # send to DA
    da.setAnnealParameterMM(param)

    received = False
    res = da.doAnnealMM(args, format="numpy", compression="lz4")
    while not received:
        if res:
            print(f"Received")
            received = True

    # start time
    start = time.time()
    # derive solution based on the minimum energy achieved
    min_energy = np.min(res['eg_min_o_n.numpy'])
    min_dec, min_cost, min_conf = None, float("inf"), None
    feasible_flag = False # flag to indicate whether exists a feasible solution
    min_idx = res['eg_min_o_n.numpy'].argmin()
    min_conf = res["state_min_o_n.numpy"][min_idx]
    dec, cost = compute_cstr_asgmt_cost(A, T, weights, extra, pairs, min_conf)

    if cost < 0 and cost < min_cost: # feasible solution
        feasible_flag = True
        min_dec, min_cost = dec, cost
    
    # elapsed_time
    elapsed_time = time.time() - start + res["anneal_time"] / 1000
    if not feasible_flag:
        return np.array([*para]), None, float("inf"), float("inf"), min_conf, elapsed_time, min_energy
    return np.array([*para]), min_dec, min_cost, min_cost, min_conf, elapsed_time, min_energy

def main_cstr_asgmt(dirName, tuner, bounds, parameters, solve, prep_config):
    def _sub(filePath):
        print(filePath)
        nonlocal bounds
        nonlocal parameters

        fileName = Path(filePath).stem
        with open(filePath, "rb") as f:
            asgmt = pickle.load(f)
        A, T, weights, extra, pairs = asgmt.A, asgmt.T, asgmt.weights, asgmt.extra, asgmt.pairs
        rules = dict(
            agent=build_agent_rule(A, T),
            task=build_task_rule(A, T),
            weight=build_weights_rule(weights, A, T),
            extra=build_extra_rule(extra, A, T),
            pair=build_pairs_rule(pairs, A, T),
        )
        start = time.time()
        dec, cost, mixed_cost, best_para, best_iter, elapsed_time = tuner(
            fileName, asgmt, rules, bounds, parameters, solve, prep_config
        )
        run_time = time.time() - start
#         if len(bounds) > 1:
#             if "pso" in parameters["folder_path"]:
#                 plot_trajectory(parameters, fileName)
#                 plot_avg_parameters(parameters, fileName)
#                 plot_avg_sol(parameters, fileName)
        print(f"\n{fileName}")
        print(f"Best solution found: {cost}")
        for i in range(len(bounds['min'])):
            print(f"Best {chr(65 + i)} found: {best_para[i]}")
        print(f"Elapsed time: {run_time}\n")
        print("*" * 36)

        # record the best solution found
        infoName = parameters["info_path"]
        if not os.path.exists(infoName):
            with open(infoName, "a") as f:
                f.write(
                    "CSTR_ASGMT, Best_Iter, "
                    + ", ".join(["Best_" + chr(65 + i) for i in range(len(bounds['min']))])
                    + ", Best_Solution, Best_Mixed, Anneal_Time, Elapsed_Time\n"
                )
        with open(infoName, "a") as f:
            f.write(
                f"{Path(fileName).stem}, {best_iter + 1}, "
                + ", ".join([f"{best_para[i]}" for i in range(len(bounds['min']))])
                + f", {cost}, {mixed_cost}, {elapsed_time}, {run_time}\n"
            )

    pathlib.Path(parameters["folder_path"]).mkdir(parents=True, exist_ok=True)

    for fileName in sorted(glob.glob(dirName)):
        _sub(fileName)

### Assignment
def solve_avg_asgmt_config(asgmt, rules, para, num_run, num_iteration, prep_config, conf=None, seed=None):
    """ prep_config is an argument not fixed anymore
    """
    global param, server_list

    # prepare the QUBO
    A, T, weights, extra = asgmt.A, asgmt.T, asgmt.weights, asgmt.extra
    qubo = build_asgmt_QUBO(para, A, T, rules)
    N_state, bias, constant, weight, s, local_field, E = prep_config(qubo, conf, seed)

    args = {
        "weight": weight,
        "lf_i_n": np.tile(local_field, num_run),
        "state_min_i_n": np.tile(s, num_run),
        "state_cur_i_n": np.tile(s, num_run),
        "eg_min_i_n": np.full(num_run, E, dtype=int),
        "eg_cur_i_n": np.full(num_run, E, dtype=int),
        "num_iteration": num_iteration,
        "num_bit": N_state,
        "num_run": num_run,
        "ope_mode": 2,
    }

    da = fjda_client.fjda(server=random.choice(server_list))

    # send to DA
    da.setAnnealParameterMM(param)

    received = False
    res = da.doAnnealMM(args, format="numpy", compression="lz4")
    while not received:
        if res:
            print(f"Received")
            received = True

    # start time
    start = time.time()
    # derive solution
    min_dec, min_cost, min_conf = None, float("inf"), None
    min_energy = float("inf")
    costs = []
    feasible_flag = False # flag to indicate whether exists a feasible solution
    for idx, conf in enumerate(res["state_min_o_n.numpy"]):
        dec, cost = compute_asgmt_cost(A, T, weights, extra, conf)

        if cost < 0: # feasible solution
            feasible_flag = True
            costs.append(cost)
            if cost < min_cost: 
                min_dec, min_cost, min_conf = dec, cost, conf
                min_energy = res["eg_min_o_n.numpy"][idx]
            
        if cost >= 0: # infeasible solution
            energy = res["eg_min_o_n.numpy"][idx]
            if energy < min_energy and not feasible_flag:
                min_conf = conf
                min_energy = energy
    
    # elapsed_time
    elapsed_time = time.time() - start + res["anneal_time"] / 1000
    if min_dec is None:
        return np.array([*para]), None, float("inf"), float("inf"), min_conf, elapsed_time, min_energy
    return np.array([*para]), min_dec, min_cost, np.mean(costs), min_conf, elapsed_time, min_energy

def solve_min_asgmt_config(asgmt, rules, para, num_run, num_iteration, prep_config, conf=None, seed=None):
    """ prep_config is an argument not fixed anymore
    """
    global param, server_list

    # prepare the QUBO
    A, T, weights, extra = asgmt.A, asgmt.T, asgmt.weights, asgmt.extra
    qubo = build_asgmt_QUBO(para, A, T, rules)
    N_state, bias, constant, weight, s, local_field, E = prep_config(qubo, conf, seed)

    args = {
        "weight": weight,
        "lf_i_n": np.tile(local_field, num_run),
        "state_min_i_n": np.tile(s, num_run),
        "state_cur_i_n": np.tile(s, num_run),
        "eg_min_i_n": np.full(num_run, E, dtype=int),
        "eg_cur_i_n": np.full(num_run, E, dtype=int),
        "num_iteration": num_iteration,
        "num_bit": N_state,
        "num_run": num_run,
        "ope_mode": 2,
    }

    da = fjda_client.fjda(server=random.choice(server_list))

    # send to DA
    da.setAnnealParameterMM(param)

    received = False
    res = da.doAnnealMM(args, format="numpy", compression="lz4")
    while not received:
        if res:
            print(f"Received")
            received = True

    # start time
    start = time.time()
    # derive solution based on the minimum energy achieved
    min_energy = np.min(res['eg_min_o_n.numpy'])
    min_dec, min_cost, min_conf = None, float("inf"), None
    feasible_flag = False # flag to indicate whether exists a feasible solution
    min_idx = res['eg_min_o_n.numpy'].argmin()
    min_conf = res["state_min_o_n.numpy"][min_idx]
    dec, cost = compute_asgmt_cost(A, T, weights, extra, min_conf)

    if cost < 0 and cost < min_cost: # feasible solution
        feasible_flag = True
        min_dec, min_cost = dec, cost
    
    # elapsed_time
    elapsed_time = time.time() - start + res["anneal_time"] / 1000
    if not feasible_flag:
        return np.array([*para]), None, float("inf"), float("inf"), min_conf, elapsed_time, min_energy
    return np.array([*para]), min_dec, min_cost, min_cost, min_conf, elapsed_time, min_energy

def main_asgmt(dirName, tuner, bounds, parameters, solve, prep_config):
    def _sub(filePath):
        print(filePath)
        nonlocal bounds
        nonlocal parameters

        fileName = Path(filePath).stem
        with open(filePath, "rb") as f:
            asgmt = pickle.load(f)
        A, T, weights, extra = asgmt.A, asgmt.T, asgmt.weights, asgmt.extra
        rules = dict(
            agent=build_agent_rule(A, T),
            task=build_task_rule(A, T),
            weight=build_weights_rule(weights, A, T),
            extra=build_extra_rule(extra, A, T),
        )
        start = time.time()
        dec, cost, mixed_cost, best_para, best_iter, elapsed_time = tuner(
            fileName, asgmt, rules, bounds, parameters, solve, prep_config
        )
        run_time = time.time() - start
#         if len(bounds) > 1:
#             if "pso" in parameters["folder_path"]:
#                 plot_trajectory(parameters, fileName)
#                 plot_avg_parameters(parameters, fileName)
#                 plot_avg_sol(parameters, fileName)
        print(f"\n{fileName}")
        print(f"Best solution found: {cost}")
        for i in range(len(bounds['min'])):
            print(f"Best {chr(65 + i)} found: {best_para[i]}")
        print(f"Elapsed time: {run_time}\n")
        print("*" * 36)

        # record the best solution found
        infoName = parameters["info_path"]
        if not os.path.exists(infoName):
            with open(infoName, "a") as f:
                f.write(
                    "ASGMT, Best_Iter, "
                    + ", ".join(["Best_" + chr(65 + i) for i in range(len(bounds['min']))])
                    + ", Best_Solution, Best_Mixed, Anneal_Time, Elapsed_Time\n"
                )
        with open(infoName, "a") as f:
            f.write(
                f"{Path(fileName).stem}, {best_iter + 1}, "
                + ", ".join([f"{best_para[i]}" for i in range(len(bounds['min']))])
                + f", {cost}, {mixed_cost}, {elapsed_time}, {run_time}\n"
            )

    pathlib.Path(parameters["folder_path"]).mkdir(parents=True, exist_ok=True)

    for fileName in sorted(glob.glob(dirName)):
        _sub(fileName)

### FSP
def solve_fsp(seq, rules, A, num_run, num_iteration):
    
    global param, server_list

    # prepare the QUBO
    qubo = build_QUBO(seq._jobs, A, rules)
    N_state, bias, constant, weight, s, local_field, E = prepare_config(qubo)

    args = {
        "weight": weight,
        "lf_i_n": np.tile(local_field, num_run),
        "state_min_i_n": np.tile(s, num_run),
        "state_cur_i_n": np.tile(s, num_run),
        "eg_min_i_n": np.full(num_run, E, dtype=int),
        "eg_cur_i_n": np.full(num_run, E, dtype=int),
        "num_iteration": num_iteration,
        "num_bit": N_state,
        "num_run": num_run,
        "ope_mode": 2,
    }

    da = fjda_client.fjda(server=random.choice(server_list))

    # send to DA
    da.setAnnealParameterMM(param)

    received = False
    res = da.doAnnealMM(args, format="numpy", compression="lz4")
    while not received:
        if res:
            print(f"Received")
            received = True

    # start time
    start = time.time()
    # derive solution
    min_perm, min_makespan = [], float("inf")
    for sol in res["state_min_o_n.numpy"]:
        da_perm, da_makespan = compute_perm_makespan(seq._jobs, sol, seq)
        if da_makespan > 0 and da_makespan < min_makespan:
            min_perm, min_makespan = da_perm, da_makespan

    # elapsed_time
    elapsed_time = time.time() - start + res["anneal_time"] / 1000
    if len(min_perm) == 0:
        return A, [], float("inf"), float("inf"), elapsed_time
    return A, min_perm, min_makespan, min_makespan, elapsed_time


def solve_fsp_avg(seq, rules, A, num_run, num_iteration):
    
    global param, server_list

    # prepare the QUBO
    qubo = build_QUBO(seq._jobs, A, rules)
    N_state, bias, constant, weight, s, local_field, E = prepare_config(qubo)

    args = {
        "weight": weight,
        "lf_i_n": np.tile(local_field, num_run),
        "state_min_i_n": np.tile(s, num_run),
        "state_cur_i_n": np.tile(s, num_run),
        "eg_min_i_n": np.full(num_run, E, dtype=int),
        "eg_cur_i_n": np.full(num_run, E, dtype=int),
        "num_iteration": num_iteration,
        "num_bit": N_state,
        "num_run": num_run,
        "ope_mode": 2,
    }

    da = fjda_client.fjda(server=random.choice(server_list))

    # send to DA
    da.setAnnealParameterMM(param)

    received = False
    res = da.doAnnealMM(args, format="numpy", compression="lz4")
    while not received:
        if res:
            print(f"Received")
            received = True

    start = time.time()
    # derive solution
    min_perm, min_makespan, mksp_list = [], float("inf"), []
    for sol in res["state_min_o_n.numpy"]:
        da_perm, da_makespan = compute_perm_makespan(seq._jobs, sol, seq)
        if da_makespan > 0:
            mksp_list.append(da_makespan)
            if da_makespan < min_makespan:
                min_makespan = da_makespan
                min_perm = da_perm

    # elapsed_time
    elapsed_time = time.time() - start + res["anneal_time"] / 1000
    return (
        A,
        min_perm,
        min_makespan,
        np.mean(mksp_list) if len(mksp_list) > 0 else float("inf"),
        elapsed_time,
    )


def main_fsp(dirName, tuner, bounds, parameters, solve):
    def _sub(filePath):
        nonlocal bounds
        nonlocal parameters
        parser = Parser(filePath)
        fileName = parser._name
        seq = parser._seq
        distance = seq.job_distance(parameters["dis"])
        rules = dict(
            column=build_column_rule(seq._jobs),
            row=build_row_rule(seq._jobs),
            edge=build_edge_rule(perturb(seq._distance), seq._jobs),
            edge_unperturbed=build_edge_rule_unperturbed(seq._distance, seq._jobs),
        )
        start = time.time()
        cur_perm, cur_mksp, cur_mixed, min_pos, best_epoch, elapsed_time = tuner(
            fileName, seq, rules, bounds, parameters, solve
        )
        run_time = time.time() - start
        start = time.time()
        two_opt_mksp, _ = seq.two_opt(
            cur_perm, cur_mksp
        )  # conduct two opt given the best solution found
        post_time = time.time() - start
        if len(bounds["min"]) > 1:
            if "pso" in parameters["folder_path"]:
                plot_trajectory(parameters, fileName)
                plot_avg_parameters(parameters, fileName)
                plot_avg_sol(parameters, fileName)
            plot_parameters(parameters, fileName)
            plot_sol(parameters, fileName)
        print(f"\n{parser._name}")
        print(f"Best solution found: {cur_mksp}")
        print(f"Two Opt solution found: {two_opt_mksp}")
        for i in range(len(bounds["min"])):
            print(f"Best {chr(65 + i)} found: {min_pos[i]}")
        print(f"Elapsed time: {run_time}\n")
        print("*" * 36)

        # record the best solution found
        infoName = parameters["info_path"]
        if not os.path.exists(infoName):
            with open(infoName, "a") as f:
                f.write(
                    "FSP, Best_Epoch, "
                    + ", ".join(
                        ["Best_" + chr(65 + i) for i in range(len(bounds["min"]))]
                    )
                    + ", Best_Solution, Two_Opt, Anneal_Time, Elapsed_Time, Two_Opt_Time\n"
                )
        with open(infoName, "a") as f:
            f.write(
                f"{Path(fileName).stem},{best_epoch + 1},"
                + ",".join([f"{min_pos[i]}" for i in range(len(bounds["min"]))])
                + f",{cur_mksp},{two_opt_mksp},{elapsed_time},{run_time},{post_time}\n"
            )

    #         pathlib.Path(f"./data/completed/").mkdir(parents=True, exist_ok=True)
    #         shutil.move(filePath, "./data/completed/{}".format(Path(filePath).stem))
    #         completed = "./data/completed/{}".format(Path(filePath).stem)

    pathlib.Path(parameters["folder_path"]).mkdir(parents=True, exist_ok=True)

    if os.path.isdir(dirName):
        for fileName in os.listdir(dirName):
            filePath = os.path.join(dirName, fileName)
            _sub(filePath)
    else:
        filePath = dirName
        _sub(filePath)


### TSP
def solve_tsp(tsp, rules, A, num_run, num_iteration, conf=None):
    
    global param, server_list

    # prepare the QUBO
    qubo = build_QUBO(tsp._no_nodes, A, rules)
    N_state, bias, constant, weight, s, local_field, E = prepare_config(qubo, conf)

    args = {
        "weight": weight,
        "lf_i_n": np.tile(local_field, num_run),
        "state_min_i_n": np.tile(s, num_run),
        "state_cur_i_n": np.tile(s, num_run),
        "eg_min_i_n": np.full(num_run, E, dtype=int),
        "eg_cur_i_n": np.full(num_run, E, dtype=int),
        "num_iteration": num_iteration,
        "num_bit": N_state,
        "num_run": num_run,
        "ope_mode": 2,
    }

    da = fjda_client.fjda(server=random.choice(server_list))
    #     da = fjda_client.fjda()

    # send to DA
    da.setAnnealParameterMM(param)

    received = False
    res = da.doAnnealMM(args, format="numpy", compression="lz4")
    while not received:
        if res:
            print(f"Received")
            received = True

    # start time
    start = time.time()
    # derive solution
    min_path, min_distance, min_conf = [], float("inf"), None
    for sol in res["state_min_o_n.numpy"]:
        da_path, da_distance = compute_path_distance(tsp._no_nodes, sol, tsp._g)
        if da_distance > 0 and da_distance < min_distance:
            min_path, min_distance, min_conf = da_path, da_distance, sol
    # elapsed_time
    elapsed_time = time.time() - start + res["anneal_time"] / 1000
    if len(min_path) == 0:
        return np.array([*A]), [], float("inf"), float("inf"), min_conf, elapsed_time
    return np.array([*A]), min_path, min_distance, min_distance, min_conf, elapsed_time


def solve_tsp_rl(tsp, rules, A, num_run, num_iteration, conf=None, seed=None):
    # solve tsp for RL
    # return state including different features
    # ? energy
    
    global param, server_list

    # prepare the QUBO
    qubo = build_QUBO(tsp._no_nodes, A, rules)
    N_state, bias, constant, weight, s, local_field, E = prepare_config_rl(qubo, conf, seed)

    args = {
        "weight": weight,
        "lf_i_n": np.tile(local_field, num_run),
        "state_min_i_n": np.tile(s, num_run),
        "state_cur_i_n": np.tile(s, num_run),
        "eg_min_i_n": np.full(num_run, E, dtype=int),
        "eg_cur_i_n": np.full(num_run, E, dtype=int),
        "num_iteration": num_iteration,
        "num_bit": N_state,
        "num_run": num_run,
        "ope_mode": 2,
    }

    da = fjda_client.fjda(server=random.choice(server_list))

    # send to DA
    da.setAnnealParameterMM(param)

    received = False
    res = da.doAnnealMM(args, format="numpy", compression="lz4")
    while not received:
        if res:
            print(f"Received")
            received = True

    # start time
    start = time.time()
    # derive solution
    min_path, min_distance, min_conf = [], float("inf"), None
    min_energy, max_edge, min_edge, avg_edge, std_edge, sum_edge = float("inf"), 0, 0, 0, 0, 0
    feasible_flag = False # flag to indicate whether exists a feasible solution
    features = ['min_energy', 'max_edge', 'min_edge', 'avg_edge', 'std_edge', 'sum_edge']
    additional_features = dict(zip(features, [0] * len(features)))
    for sol in res["state_min_o_n.numpy"]:
        da_path, da_distance = compute_path_distance(tsp._no_nodes, sol, tsp._g)

        if da_distance > 0 and da_distance < min_distance: # feasible solution and the tour length is the minimum
            feasible_flag = True
            min_path, min_distance, min_conf = da_path, da_distance, sol
            min_energy = getEnergy(qubo, sol)
            max_edge, min_edge, avg_edge, std_edge, sum_edge = extract_tsp_features(tsp._no_nodes, sol, tsp._g)
            
        if da_distance < 0: # infeasible solution
            energy = getEnergy(qubo, sol)
            if energy < min_energy and not feasible_flag:
                min_conf = sol
                min_energy = energy
                max_edge, min_edge, avg_edge, std_edge, sum_edge = extract_tsp_features(tsp._no_nodes, sol, tsp._g)
    
    additional_features['min_energy'] = min_energy
    additional_features['max_edge'] = max_edge
    additional_features['min_edge'] = min_edge
    additional_features['avg_edge'] = avg_edge
    additional_features['std_edge'] = std_edge
    additional_features['sum_edge'] = sum_edge
    # elapsed_time
    elapsed_time = time.time() - start + res["anneal_time"] / 1000
    if len(min_path) == 0:
        return np.array([*A]), [], float("inf"), float("inf"), min_conf, elapsed_time, additional_features
    return np.array([*A]), min_path, min_distance, min_distance, min_conf, elapsed_time, additional_features

# One Less
def solve_tsp_one_less(tsp, rules, A, num_run, num_iteration, prep_config, conf=None, seed=None):
    # solve tsp for RL
    # return state including different features
    
    global param, server_list

    # prepare the QUBO
    qubo = build_QUBO_one_less(tsp._no_nodes, A, rules)
    N_state, bias, constant, weight, s, local_field, E = prep_config(qubo, conf, seed)

    args = {
        "weight": weight,
        "lf_i_n": np.tile(local_field, num_run),
        "state_min_i_n": np.tile(s, num_run),
        "state_cur_i_n": np.tile(s, num_run),
        "eg_min_i_n": np.full(num_run, E, dtype=int),
        "eg_cur_i_n": np.full(num_run, E, dtype=int),
        "num_iteration": num_iteration,
        "num_bit": N_state,
        "num_run": num_run,
        "ope_mode": 2,
    }

    da = fjda_client.fjda(server=random.choice(server_list))

    # send to DA
    da.setAnnealParameterMM(param)

    received = False
    res = da.doAnnealMM(args, format="numpy", compression="lz4")
    while not received:
        if res:
            print(f"Received")
            received = True

    # start time
    start = time.time()
    # derive solution
    min_path, min_distance, min_conf = [], float("inf"), None
    min_energy, max_edge, min_edge, avg_edge, std_edge, sum_edge = float("inf"), 0, 0, 0, 0, 0
    feasible_flag = False # flag to indicate whether exists a feasible solution
    features = ['min_energy', 'max_edge', 'min_edge', 'avg_edge', 'std_edge', 'sum_edge']
    additional_features = dict(zip(features, [0] * len(features)))
    for sol in res["state_min_o_n.numpy"]:
        da_path, da_distance = compute_path_distance_one_less(tsp._no_nodes, sol, tsp._g)

        if da_distance > 0 and da_distance < min_distance: # feasible solution and the tour length is the minimum
            feasible_flag = True
            min_path, min_distance, min_conf = da_path, da_distance, sol
            
        if da_distance < 0 and not feasible_flag: # infeasible solution
            min_conf = sol
            
    # elapsed_time
    elapsed_time = time.time() - start + res["anneal_time"] / 1000
    if len(min_path) == 0:
        return np.array([*A]), [], float("inf"), float("inf"), min_conf, elapsed_time, additional_features
    return np.array([*A]), min_path, min_distance, min_distance, min_conf, elapsed_time, additional_features

# ALM
def solve_tsp_lambda(tsp, rules, A, lambda_param, num_run, num_iteration, prep_config, conf=None, seed=None):
    # solve tsp for AML
    # return state including different features
    
    global param, server_list

    # prepare the QUBO
    qubo = build_QUBO_lambda(tsp._no_nodes, A, lambda_param, rules)
    N_state, bias, constant, weight, s, local_field, E = prep_config(qubo, conf, seed)

    args = {
        "weight": weight,
        "lf_i_n": np.tile(local_field, num_run),
        "state_min_i_n": np.tile(s, num_run),
        "state_cur_i_n": np.tile(s, num_run),
        "eg_min_i_n": np.full(num_run, E, dtype=int),
        "eg_cur_i_n": np.full(num_run, E, dtype=int),
        "num_iteration": num_iteration,
        "num_bit": N_state,
        "num_run": num_run,
        "ope_mode": 2,
    }

    da = fjda_client.fjda(server=random.choice(server_list))

    # send to DA
    da.setAnnealParameterMM(param)

    received = False
    res = da.doAnnealMM(args, format="numpy", compression="lz4")
    while not received:
        if res:
            print(f"Received")
            received = True

    # start time
    start = time.time()
    # derive solution
    min_path, min_distance, min_conf = [], float("inf"), None
    min_energy, max_edge, min_edge, avg_edge, std_edge, sum_edge = float("inf"), 0, 0, 0, 0, 0
    feasible_flag = False # flag to indicate whether exists a feasible solution
    features = ['min_energy', 'max_edge', 'min_edge', 'avg_edge', 'std_edge', 'sum_edge']
    additional_features = dict(zip(features, [0] * len(features)))
    for sol in res["state_min_o_n.numpy"]:
        da_path, da_distance = compute_path_distance(tsp._no_nodes, sol, tsp._g)

        if da_distance > 0 and da_distance < min_distance: # feasible solution and the tour length is the minimum
            feasible_flag = True
            min_path, min_distance, min_conf = da_path, da_distance, sol
            min_energy = getEnergy(qubo, sol)
            max_edge, min_edge, avg_edge, std_edge, sum_edge = extract_tsp_features(tsp._no_nodes, sol, tsp._g)
            
        if da_distance < 0: # infeasible solution
            energy = getEnergy(qubo, sol)
            if energy < min_energy and not feasible_flag:
                min_conf = sol
                min_energy = energy
                max_edge, min_edge, avg_edge, std_edge, sum_edge = extract_tsp_features(tsp._no_nodes, sol, tsp._g)
    
    additional_features['min_energy'] = min_energy
    additional_features['max_edge'] = max_edge
    additional_features['min_edge'] = min_edge
    additional_features['avg_edge'] = avg_edge
    additional_features['std_edge'] = std_edge
    additional_features['sum_edge'] = sum_edge
    # elapsed_time
    elapsed_time = time.time() - start + res["anneal_time"] / 1000
    if len(min_path) == 0:
        return np.array([*A]), lambda_param, [], float("inf"), float("inf"), min_conf, elapsed_time, additional_features
    return np.array([*A]), lambda_param, min_path, min_distance, min_distance, min_conf, elapsed_time, additional_features

def solve_tsp_config(tsp, rules, A, num_run, num_iteration, prep_config, conf=None, seed=None):
    """ prep_config is an argument not fixed anymore
    """
    # solve tsp for RL
    # return state including different features
    
    global param, server_list

    # prepare the QUBO
    qubo = build_QUBO(tsp._no_nodes, A, rules)
    N_state, bias, constant, weight, s, local_field, E = prep_config(qubo, conf, seed)

    args = {
        "weight": weight,
        "lf_i_n": np.tile(local_field, num_run),
        "state_min_i_n": np.tile(s, num_run),
        "state_cur_i_n": np.tile(s, num_run),
        "eg_min_i_n": np.full(num_run, E, dtype=int),
        "eg_cur_i_n": np.full(num_run, E, dtype=int),
        "num_iteration": num_iteration,
        "num_bit": N_state,
        "num_run": num_run,
        "ope_mode": 2,
    }

    da = fjda_client.fjda(server=random.choice(server_list))

    # send to DA
    da.setAnnealParameterMM(param)

    received = False
    res = da.doAnnealMM(args, format="numpy", compression="lz4")
    while not received:
        if res:
            print(f"Received")
            received = True

    # start time
    start = time.time()
    # derive solution
    min_path, min_distance, min_conf = [], float("inf"), None
    min_energy, max_edge, min_edge, avg_edge, std_edge, sum_edge = float("inf"), 0, 0, 0, 0, 0
    feasible_flag = False # flag to indicate whether exists a feasible solution
    features = ['min_energy', 'max_edge', 'min_edge', 'avg_edge', 'std_edge', 'sum_edge']
    additional_features = dict(zip(features, [0] * len(features)))
    for sol in res["state_min_o_n.numpy"]:
        da_path, da_distance = compute_path_distance(tsp._no_nodes, sol, tsp._g)

        if da_distance > 0 and da_distance < min_distance: # feasible solution and the tour length is the minimum
            feasible_flag = True
            min_path, min_distance, min_conf = da_path, da_distance, sol
            min_energy = getEnergy(qubo, sol)
            max_edge, min_edge, avg_edge, std_edge, sum_edge = extract_tsp_features(tsp._no_nodes, sol, tsp._g)
            
        if da_distance < 0: # infeasible solution
            energy = getEnergy(qubo, sol)
            if energy < min_energy and not feasible_flag:
                min_conf = sol
                min_energy = energy
                max_edge, min_edge, avg_edge, std_edge, sum_edge = extract_tsp_features(tsp._no_nodes, sol, tsp._g)
    
    additional_features['min_energy'] = min_energy
    additional_features['max_edge'] = max_edge
    additional_features['min_edge'] = min_edge
    additional_features['avg_edge'] = avg_edge
    additional_features['std_edge'] = std_edge
    additional_features['sum_edge'] = sum_edge
    # elapsed_time
    elapsed_time = time.time() - start + res["anneal_time"] / 1000
    if len(min_path) == 0:
        return np.array([*A]), [], float("inf"), float("inf"), min_conf, elapsed_time, additional_features
    return np.array([*A]), min_path, min_distance, min_distance, min_conf, elapsed_time, additional_features

def solve_min_tsp_config(tsp, rules, A, num_run, num_iteration, prep_config, conf=None, seed=None):
    """ prep_config is an argument not fixed anymore
    """
    # solve tsp for RL
    # return state including different features
    
    global param, server_list

    # prepare the QUBO
    qubo = build_QUBO(tsp._no_nodes, A, rules)
    N_state, bias, constant, weight, s, local_field, E = prep_config(qubo, conf, seed)

    args = {
        "weight": weight,
        "lf_i_n": np.tile(local_field, num_run),
        "state_min_i_n": np.tile(s, num_run),
        "state_cur_i_n": np.tile(s, num_run),
        "eg_min_i_n": np.full(num_run, E, dtype=int),
        "eg_cur_i_n": np.full(num_run, E, dtype=int),
        "num_iteration": num_iteration,
        "num_bit": N_state,
        "num_run": num_run,
        "ope_mode": 2,
    }
    
    da = fjda_client.fjda(server=random.choice(server_list))

    # send to DA
    da.setAnnealParameterMM(param)

    received = False
    res = da.doAnnealMM(args, format="numpy", compression="lz4")
    while not received:
        if res:
            print(f"Received")
            received = True

    # start time
    start = time.time()
    # derive solution
    min_path, min_distance, min_conf = [], float("inf"), None
    feasible_flag = False # flag to indicate whether exists a feasible solution
    # retrieve the solution with minimum energy
    min_energy = np.min(res['eg_min_o_n.numpy'])
    min_idx = res['eg_min_o_n.numpy'].argmin()
    min_conf = res["state_min_o_n.numpy"][min_idx]
    da_path, da_distance = compute_path_distance(tsp._no_nodes, min_conf, tsp._g)

    if da_distance > 0 and da_distance < min_distance: # feasible solution and the tour length is the minimum
        feasible_flag = True
        min_path, min_distance = da_path, da_distance
    
    # elapsed_time
    elapsed_time = time.time() - start + res["anneal_time"] / 1000
    if not feasible_flag:
        return np.array([*A]), [], float("inf"), float("inf"), min_conf, elapsed_time, min_energy
    return np.array([*A]), min_path, min_distance, min_distance, min_conf, elapsed_time, min_energy

def solve_avg_tsp_config(tsp, rules, para, num_run, num_iteration, prep_config, conf=None, seed=None):

    global param, server_list
    
    # prepare the QUBO
    qubo = build_QUBO(tsp._no_nodes, para, rules)
    N_state, bias, constant, weight, s, local_field, E = prep_config(qubo, conf)

    args = {
        "weight": weight,
        "lf_i_n": np.tile(local_field, num_run),
        "state_min_i_n": np.tile(s, num_run),
        "state_cur_i_n": np.tile(s, num_run),
        "eg_min_i_n": np.full(num_run, E, dtype=int),
        "eg_cur_i_n": np.full(num_run, E, dtype=int),
        "num_iteration": num_iteration,
        "num_bit": N_state,
        "num_run": num_run,
        "ope_mode": 2,
    }

    da = fjda_client.fjda(server=random.choice(server_list))

    # send to DA
    da.setAnnealParameterMM(param)

    received = False
    res = da.doAnnealMM(args, format="numpy", compression="lz4")
    while not received:
        if res:
            print(f"Received")
            received = True

    start = time.time()
    # derive solution
    min_path, min_distance, costs, min_conf = [], float("inf"), [], None
    min_energy = np.min(res['eg_min_o_n.numpy'])
    for conf in res["state_min_o_n.numpy"]:
        da_path, da_distance = compute_path_distance(tsp._no_nodes, conf, tsp._g)
        if da_distance > 0:
            costs.append(da_distance)
            if da_distance < min_distance:
                min_distance = da_distance
                min_path = da_path
                min_conf = conf

    # elapsed_time
    elapsed_time = time.time() - start + res["anneal_time"] / 1000
    return (
        np.array([*para]),
        min_path,
        min_distance,
        np.mean(costs) if len(costs) > 0 else float("inf"),
        min_conf,
        elapsed_time,
        min_energy
    )

### PSO
def solve_tsp_avg(tsp, rules, A, num_run, num_iteration, conf=None):
    
    global param, server_list

    # prepare the QUBO
    qubo = build_QUBO(tsp._no_nodes, A, rules)
    N_state, bias, constant, weight, s, local_field, E = prepare_config(qubo, conf)

    args = {
        "weight": weight,
        "lf_i_n": np.tile(local_field, num_run),
        "state_min_i_n": np.tile(s, num_run),
        "state_cur_i_n": np.tile(s, num_run),
        "eg_min_i_n": np.full(num_run, E, dtype=int),
        "eg_cur_i_n": np.full(num_run, E, dtype=int),
        "num_iteration": num_iteration,
        "num_bit": N_state,
        "num_run": num_run,
        "ope_mode": 2,
    }

    da = fjda_client.fjda(server=random.choice(server_list))

    # send to DA
    da.setAnnealParameterMM(param)

    received = False
    res = da.doAnnealMM(args, format="numpy", compression="lz4")
    while not received:
        if res:
            print(f"Received")
            received = True

    start = time.time()
    # derive solution
    min_path, min_distance, dis_list, min_conf = [], float("inf"), [], None
    for sol in res["state_min_o_n.numpy"]:
        da_path, da_distance = compute_path_distance(tsp._no_nodes, sol, tsp._g)
        if da_distance > 0:
            dis_list.append(da_distance)
            if da_distance < min_distance:
                min_distance = da_distance
                min_path = da_path
                min_conf = sol

    # elapsed_time
    elapsed_time = time.time() - start + res["anneal_time"] / 1000
    return (
        np.array([*A]),
        min_path,
        min_distance,
        np.mean(dis_list) if len(dis_list) > 0 else float("inf"),
        min_conf,
        elapsed_time,
    )

def solve_tsp_all(tsp, rules, A, num_run, num_iteration, conf=None):
    
    global param, server_list

    # prepare the QUBO
    qubo = build_QUBO(tsp._no_nodes, A, rules)
    N_state, bias, constant, weight, s, local_field, E = prepare_config(qubo, conf)

    args = {
        "weight": weight,
        "lf_i_n": np.tile(local_field, num_run),
        "state_min_i_n": np.tile(s, num_run),
        "state_cur_i_n": np.tile(s, num_run),
        "eg_min_i_n": np.full(num_run, E, dtype=int),
        "eg_cur_i_n": np.full(num_run, E, dtype=int),
        "num_iteration": num_iteration,
        "num_bit": N_state,
        "num_run": num_run,
        "ope_mode": 2,
    }

    da = fjda_client.fjda(server=random.choice(server_list))

    # send to DA
    da.setAnnealParameterMM(param)

    received = False
    res = da.doAnnealMM(args, format="numpy", compression="lz4")
    while not received:
        if res:
            print(f"Received")
            received = True

    # start time
    start = time.time()
    # derive solution
    min_path, min_distance, min_sol = [], float("inf"), None
    for sol in res["state_min_o_n.numpy"]:
        da_path, da_distance = compute_path_distance(tsp._no_nodes, sol, tsp._g)
        if da_distance > 0 and da_distance < min_distance:
            min_path, min_distance, min_sol = da_path, da_distance, sol

    # elapsed_time
    elapsed_time = time.time() - start + res["anneal_time"] / 1000
    if len(min_path) == 0:
        return (
            np.array([*A, num_run, num_iteration]),
            [],
            float("inf"),
            float("inf"),
            min_sol,
            elapsed_time,
        )
    return (
        np.array([*A, num_run, num_iteration]),
        min_path,
        min_distance,
        min_distance,
        min_sol,
        elapsed_time,
    )


def solve_tsp_avg_all(tsp, rules, A, num_run, num_iteration, conf=None):
    
    global param, server_list

    # prepare the QUBO
    qubo = build_QUBO(tsp._no_nodes, A, rules)
    N_state, bias, constant, weight, s, local_field, E = prepare_config(qubo, conf)

    args = {
        "weight": weight,
        "lf_i_n": np.tile(local_field, num_run),
        "state_min_i_n": np.tile(s, num_run),
        "state_cur_i_n": np.tile(s, num_run),
        "eg_min_i_n": np.full(num_run, E, dtype=int),
        "eg_cur_i_n": np.full(num_run, E, dtype=int),
        "num_iteration": num_iteration,
        "num_bit": N_state,
        "num_run": num_run,
        "ope_mode": 2,
    }

    da = fjda_client.fjda(server=random.choice(server_list))

    # send to DA
    da.setAnnealParameterMM(param)

    received = False
    res = da.doAnnealMM(args, format="numpy", compression="lz4")
    while not received:
        if res:
            print(f"Received")
            received = True

    start = time.time()
    # derive solution
    min_path, min_distance, dis_list, min_sol = [], float("inf"), [], None
    for sol in res["state_min_o_n.numpy"]:
        da_path, da_distance = compute_path_distance(tsp._no_nodes, sol, tsp._g)
        if da_distance > 0:
            dis_list.append(da_distance)
            if da_distance < min_distance:
                min_distance = da_distance
                min_path = da_path
                min_sol = sol

    # elapsed_time
    elapsed_time = time.time() - start + res["anneal_time"] / 1000
    return (
        np.array([*A, num_run, num_iteration]),
        min_path,
        min_distance,
        np.mean(dis_list) if len(dis_list) > 0 else float("inf"),
        min_sol,
        elapsed_time,
    )


def main_tsp(dirName, tuner, bounds, parameters, solve):
    def _sub(filePath):
        print(filePath)
        nonlocal bounds
        nonlocal parameters
        # graph = load_graph(filePath)
        # tsp = TSP(fileName, graph)
        fileName = Path(filePath).stem
        tsp = TSP(tsplib95.load(filePath))
        rules = dict(
            column=build_column_rule(tsp._no_nodes),
            row=build_row_rule(tsp._no_nodes),
            edge=build_edge_rule(perturb(tsp._g), tsp._no_nodes),
            edge_unperturbed=build_edge_rule_unperturbed(tsp._g, tsp._no_nodes),
        )
        start = time.time()
        cur_perm, cur_dis, cur_mixed, min_pos, best_epoch, elapsed_time = tuner(
            fileName, tsp, rules, bounds, parameters, solve
        )
        run_time = time.time() - start
        if len(bounds) > 1:
            if "pso" in parameters["folder_path"]:
                plot_trajectory(parameters, fileName)
                plot_avg_parameters(parameters, fileName)
                plot_avg_sol(parameters, fileName)
        #             plot_parameters(parameters, fileName)
        #             plot_sol(parameters, fileName)
        print(f"\n{fileName}")
        print(f"Best solution found: {cur_dis}")
        for i, k in enumerate(sorted(bounds.keys())):
            print(f"Best {k[1:]} found: {min_pos[i]}")
        print(f"Elapsed time: {run_time}\n")
        print("*" * 36)

        # record the best solution found
        infoName = parameters["info_path"]
        if not os.path.exists(infoName):
            with open(infoName, "a") as f:
                f.write(
                    "TSP, Best_Epoch, "
                    + ", ".join(["Best_" + k[1:] for k in sorted(bounds.keys())])
                    + ", Best_Solution, Best_Mixed, Anneal_Time, Elapsed_Time\n"
                )
        with open(infoName, "a") as f:
            f.write(
                f"{Path(fileName).stem},{best_epoch + 1},"
                + ",".join([f"{min_pos[i]}" for i in range(len(bounds))])
                + f",{cur_dis},{cur_mixed},{elapsed_time},{run_time}\n"
            )

    #         pathlib.Path(f"./data/completed/").mkdir(parents=True, exist_ok=True)
    #         shutil.move(filePath, "./data/completed/{}".format(Path(filePath).stem))
    #         completed = "./data/completed/{}".format(Path(filePath).stem)

    pathlib.Path(parameters["folder_path"]).mkdir(parents=True, exist_ok=True)

    # pso
    #     if os.path.isdir(dirName):
    # for fileName in os.listdir(dirName):
    for fileName in sorted(glob.glob(dirName)):
        #         filePath = os.path.join(dirName, fileName)
        #         _sub(filePath)
        _sub(fileName)

#     else:
#         filePath = dirName
#         _sub(filePath)

# One Less
def main_tsp_one_less(dirName, tuner, bounds, parameters, solve, prep_config, mode='perturbed'):
    def _sub(filePath):
        print(filePath)
        nonlocal bounds
        nonlocal parameters

        fileName = Path(filePath).stem
        tsp = TSP(tsplib95.load(filePath))
        rules = dict(
            column_one_less=build_column_rule_one_less(tsp._no_nodes),
            row_one_less=build_row_rule_one_less(tsp._no_nodes),
            edge_one_less=build_edge_rule_one_less(perturb(tsp._g) if mode == 'perturbed' else tsp._g, tsp._no_nodes)
        )
        start = time.time()
        cur_perm, cur_dis, cur_mixed, min_pos, best_epoch, elapsed_time = tuner(
            fileName, tsp, rules, bounds, parameters, solve, prep_config
        )
        run_time = time.time() - start
        if len(bounds) > 1:
            if "pso" in parameters["folder_path"]:
                plot_trajectory(parameters, fileName)
                plot_avg_parameters(parameters, fileName)
                plot_avg_sol(parameters, fileName)
        print(f"\n{fileName}")
        print(f"Best solution found: {cur_dis}")
        for i in range(len(bounds['min'])):
            print(f"Best {chr(65 + i)} found: {min_pos[i]}")
        print(f"Elapsed time: {run_time}\n")
        print("*" * 36)

        # record the best solution found
        infoName = parameters["info_path"]
        if not os.path.exists(infoName):
            with open(infoName, "a") as f:
                f.write(
                    "TSP, Best_Epoch, "
                    + ", ".join(["Best_" + chr(65 + i) for i in range(len(bounds['min']))])
                    + ", Best_Solution, Best_Mixed, Anneal_Time, Elapsed_Time\n"
                )
        with open(infoName, "a") as f:
            f.write(
                f"{Path(fileName).stem}, {best_epoch + 1}, "
                + ", ".join([f"{min_pos[i]}" for i in range(len(bounds['min']))])
                + f", {cur_dis}, {cur_mixed}, {elapsed_time}, {run_time}\n"
            )

    pathlib.Path(parameters["folder_path"]).mkdir(parents=True, exist_ok=True)

    for fileName in sorted(glob.glob(dirName)):
        _sub(fileName)

# Mode
def main_tsp_mode(dirName, tuner, bounds, parameters, solve, prep_config, mode='perturbed'):
    def _sub(filePath):
        print(filePath)
        nonlocal bounds
        nonlocal parameters

        fileName = Path(filePath).stem
        tsp = TSP(tsplib95.load(filePath))
        rules = dict(
            column=build_column_rule(tsp._no_nodes),
            row=build_row_rule(tsp._no_nodes),
            edge=build_edge_rule(perturb(tsp._g) if mode == 'perturbed' else tsp._g, tsp._no_nodes),
            edge_unperturbed=build_edge_rule_unperturbed(tsp._g, tsp._no_nodes),
        )
        start = time.time()
        cur_perm, cur_dis, cur_mixed, min_pos, best_epoch, elapsed_time = tuner(
            fileName, tsp, rules, bounds, parameters, solve, prep_config
        )
        run_time = time.time() - start
#         if len(bounds) > 1:
#             if "pso" in parameters["folder_path"]:
#                 plot_trajectory(parameters, fileName)
#                 plot_avg_parameters(parameters, fileName)
#                 plot_avg_sol(parameters, fileName)
#                     plot_parameters(parameters, fileName)
#                     plot_sol(parameters, fileName)
        print(f"\n{fileName}")
        print(f"Best solution found: {cur_dis}")
        for i in range(len(bounds['min'])):
            print(f"Best {chr(65 + i)} found: {min_pos[i]}")
        print(f"Elapsed time: {run_time}\n")
        print("*" * 36)

        # record the best solution found
        infoName = parameters["info_path"]
        if not os.path.exists(infoName):
            with open(infoName, "a") as f:
                f.write(
                    "TSP, Best_Iter, "
                    + ", ".join(["Best_" + chr(65 + i) for i in range(len(bounds['min']))])
                    + ", Best_Solution, Best_Mixed, Anneal_Time, Elapsed_Time\n"
                )
        with open(infoName, "a") as f:
            f.write(
                f"{Path(fileName).stem}, {best_epoch + 1}, "
                + ", ".join([f"{min_pos[i]}" for i in range(len(bounds['min']))])
                + f", {cur_dis}, {cur_mixed}, {elapsed_time}, {run_time}\n"
            )

    pathlib.Path(parameters["folder_path"]).mkdir(parents=True, exist_ok=True)

    for fileName in sorted(glob.glob(dirName)):
        _sub(fileName)
        
def main_tsp_mod(dirName, tuner, bounds, parameters, solve):
    def _sub(filePath):
        print(filePath)
        nonlocal bounds
        nonlocal parameters

        fileName = Path(filePath).stem
        tsp = TSP(tsplib95.load(filePath))
        rules = dict(
            column=build_column_rule(tsp._no_nodes),
            row=build_row_rule(tsp._no_nodes),
            edge=build_edge_rule(tsp._g, tsp._no_nodes),
            # edge=build_edge_rule(perturb(tsp._g), tsp._no_nodes),
            edge_unperturbed=build_edge_rule_unperturbed(tsp._g, tsp._no_nodes),
        )
        start = time.time()
        cur_perm, cur_dis, cur_mixed, min_pos, best_epoch, elapsed_time = tuner(
            fileName, tsp, rules, bounds, parameters, solve
        )
        run_time = time.time() - start
        if len(bounds) > 1:
            if "pso" in parameters["folder_path"]:
                plot_trajectory(parameters, fileName)
                plot_avg_parameters(parameters, fileName)
                plot_avg_sol(parameters, fileName)
        #             plot_parameters(parameters, fileName)
        #             plot_sol(parameters, fileName)
        print(f"\n{fileName}")
        print(f"Best solution found: {cur_dis}")
        for i in range(len(bounds['min'])):
            print(f"Best {chr(65 + i)} found: {min_pos[i]}")
        print(f"Elapsed time: {run_time}\n")
        print("*" * 36)

        # record the best solution found
        infoName = parameters["info_path"]
        if not os.path.exists(infoName):
            with open(infoName, "a") as f:
                f.write(
                    "TSP, Best_Epoch, "
                    + ", ".join(["Best_" + chr(65 + i) for i in range(len(bounds['min']))])
                    + ", Best_Solution, Best_Mixed, Anneal_Time, Elapsed_Time\n"
                )
        with open(infoName, "a") as f:
            f.write(
                f"{Path(fileName).stem}, {best_epoch + 1}, "
                + ", ".join([f"{min_pos[i]}" for i in range(len(bounds['min']))])
                + f", {cur_dis}, {cur_mixed}, {elapsed_time}, {run_time}\n"
            )

    pathlib.Path(parameters["folder_path"]).mkdir(parents=True, exist_ok=True)

    for fileName in sorted(glob.glob(dirName)):
        _sub(fileName)


def save_solution(
    fileName, e, pos, min_perm, min_cost, mixed_cost, elapsed_time, parameters, bounds
):
    pathlib.Path(parameters["folder_path"]).mkdir(parents=True, exist_ok=True)
    ftspName = Path(fileName).stem
    outputName = f"{parameters['folder_path']}{ftspName}.csv"
    if not os.path.exists(outputName):
        with open(outputName, "a") as f:
            f.write(
                "iter, "
                + ", ".join([chr(65 + i) for i in range(len(bounds['min']))])
#                 + ", ".join([k[1:] for k in sorted(bounds.keys())])
                + ", cost, mixed_cost, elapsed_time\n"
            )
    with open(outputName, "a") as f:
        f.write(
            f"{e + 1 if isinstance(e, int) else e}, "
            + ", ".join([f"{pos[i]}" for i in range(len(pos))])
            + f", {min_cost}, {mixed_cost}, {elapsed_time}\n"
        )
        
def save_pop_solution(
    fileName, e, i, pos, min_perm, min_cost, mixed_cost, elapsed_time, parameters, bounds
):
    pathlib.Path(parameters["folder_path"]).mkdir(parents=True, exist_ok=True)
    ftspName = Path(fileName).stem
    outputName = f"{parameters['folder_path']}{ftspName}.csv"
    if not os.path.exists(outputName):
        with open(outputName, "a") as f:
            f.write(
                "iter, index, "
                + ", ".join([chr(65 + i) for i in range(len(bounds['min']))])
                + ", cost, mixed_cost, elapsed_time\n"
            )
    with open(outputName, "a") as f:
        f.write(
            f"{e + 1 if isinstance(e, int) else e}, {i}, "
            + ", ".join([f"{pos[i]}" for i in range(len(pos))])
            + f", {min_cost}, {mixed_cost}, {elapsed_time}\n"
        )


def save_epoch(
    fileName,
    e,
    cur_cost,
    epoch_min,
    cur_mixed,
    min_pos,
    elapsed_time,
    parameters,
    bounds,
):
    pathlib.Path(parameters["folder_path"]).mkdir(parents=True, exist_ok=True)
    ftspName = Path(fileName).stem
    outputName = f"{parameters['folder_path']}{ftspName}-Iter.csv"
    if not os.path.exists(outputName):
        with open(outputName, "a") as f:
            f.write(
                "Iter, "
#                 + ", ".join([f"Current_Best_{k[1:]}" for k in sorted(bounds.keys())])
                + ", ".join([f"Current_Best_{chr(65 + i)}" for i in range(len(bounds['min']))])
                + ", Current_Best_Cost, Current_Best_Mixed, Current_Epoch_cost, Elapsed_Time\n"
            )
    with open(outputName, "a") as f:
        f.write(
            f"{e + 1 if isinstance(e, int) else e}, "
            + ", ".join([f"{min_pos[i]}" for i in range(len(min_pos))])
            + f", {cur_cost}, {cur_mixed}, {epoch_min}, {elapsed_time}\n"
        )


def plot_trajectory(parameters, fileName):
    ftspName = Path(fileName).stem
    outputName = f"{parameters['folder_path']}{ftspName}.csv"
    df = pd.read_csv(outputName)
    df = df.rename(columns=lambda x: x.strip().replace(" ", "_"))

    fig = plt.figure(dpi=300)
    ax = plt.axes()
    ax.set_xlabel("A")
    ax.set_ylabel("B")
    camera = Camera(fig)
    for i in range(df["epoch"].max() + 1):
        points = np.array(
            [
                list(df["A"].iloc[df.index[df["epoch"] == i]]),
                list(df["B"].iloc[df.index[df["epoch"] == i]]),
            ]
        )
        numpoints = len(df.index[df["epoch"] == i].tolist())
        colors = cm.rainbow(np.linspace(0, 1, numpoints))
        t = plt.scatter(*points, c=colors, s=100)
        plt.legend((t,), [f"iter {i}"])
        plt.title(ftspName)
        camera.snap()
    anim = camera.animate(blit=True)
    writer = PillowWriter(fps=1)
    anim.save(filename=f"{parameters['folder_path']}{ftspName}.gif", writer=writer)
    plt.close()


def plot_parameters(parameters, fileName):
    ftspName = Path(fileName).stem
    outputName = f"{parameters['folder_path']}{ftspName}.csv"
    df = pd.read_csv(outputName)
    df = df.rename(columns=lambda x: x.strip().replace(" ", "_"))

    fig = plt.figure(dpi=300)
    # plot size
    ax = plt.axes()
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.set_ylim([0.45 * df["A"].max(), 1.05 * df["A"].max()])
    ax.set_xlabel("")
    ax.set_ylabel("")
    # A
    x = np.linspace(0, len(df["A"]), len(df["A"]))
    ax.plot(x, list(df["A"]), label="A")
    # B
    x = np.linspace(0, len(df["B"]), len(df["B"]))
    ax.plot(x, list(df["B"]), label="B")
    ax.legend()
    # save the plot
    plt.title(ftspName)
    plt.savefig(f"{parameters['folder_path']}{ftspName}-para.png")
    plt.close()


def plot_avg_parameters(parameters, fileName):
    def _autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                "{:.0f}".format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 5),  # 3 points vertical offset
                textcoords="offset points",
                ha="left",
                va="bottom",
            )

    ftspName = Path(fileName).stem
    outputName = f"{parameters['folder_path']}{ftspName}.csv"
    df = pd.read_csv(outputName)
    df = df.rename(columns=lambda x: x.strip().replace(" ", "_"))

    # aggregation
    A_mean = df.groupby("epoch")["A"].mean().tolist()
    A_std = df.groupby("epoch")["A"].std().tolist()
    B_mean = df.groupby("epoch")["B"].mean().tolist()
    B_std = df.groupby("epoch")["B"].std().tolist()

    # plot
    labels = [f"Iter {i}" for i in range(1 + max(df["epoch"]))]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(
        x - width / 2,
        A_mean,
        yerr=A_std,
        width=width,
        label="A",
        align="center",
        alpha=1,
        ecolor="black",
        capsize=5,
    )
    rects2 = ax.bar(
        x + width / 2,
        B_mean,
        yerr=B_std,
        width=width,
        label="B",
        align="center",
        alpha=1,
        ecolor="black",
        capsize=5,
    )

    ax.set_ylabel("Average Parameter")
    ax.set_title("Average Parameters by Iteration")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # label
    _autolabel(rects1)
    _autolabel(rects2)

    fig.tight_layout()

    plt.title(ftspName)
    plt.savefig(f"{parameters['folder_path']}{ftspName}-avg-para.png", dpi=300)
    plt.close()


def plot_sol(parameters, fileName):

    ftspName = Path(fileName).stem
    outputName = f"{parameters['folder_path']}{ftspName}.csv"
    df = pd.read_csv(outputName)
    df = df.rename(columns=lambda x: x.strip().replace(" ", "_"))
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

    fig = plt.figure(dpi=300)
    # plot size
    ax = plt.axes()
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.set_xlabel("")
    ax.set_ylabel("Solution")
    # cost
    x = np.linspace(0, len(df["da_cost"]), len(df["da_cost"]))
    ax.plot(x, list(df["da_cost"]))
    # save the plot
    plt.title(ftspName)
    plt.grid(True)
    plt.savefig(f"{parameters['folder_path']}{ftspName}-sol.png", dpi=300)
    plt.close()


def plot_avg_sol(parameters, fileName):
    def _autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                "{:.0f}".format(height),
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 5),  # 3 points vertical offset
                textcoords="offset points",
                ha="left",
                va="bottom",
            )

    ftspName = Path(fileName).stem
    outputName = f"{parameters['folder_path']}{ftspName}.csv"
    df = pd.read_csv(outputName)
    df = df.rename(columns=lambda x: x.strip().replace(" ", "_"))
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

    # aggregation
    cost_mean = df.groupby("epoch")["da_cost"].mean().tolist()
    cost_std = df.groupby("epoch")["da_cost"].std().tolist()

    # plot
    labels = [f"Iter {i}" for i in range(1 + max(df["epoch"]))]
    x = np.arange(len(labels))
    width = 0.5

    fig, ax = plt.subplots()
    rects1 = ax.bar(
        x,
        cost_mean,
        yerr=cost_std,
        width=width,
        label="solution",
        align="center",
        alpha=1,
        ecolor="black",
        capsize=5,
    )

    ax.set_ylabel("Average Solution")
    ax.set_title("Average Solution by Iteration")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # label
    _autolabel(rects1)

    fig.tight_layout()

    plt.title(ftspName)
    plt.savefig(f"{parameters['folder_path']}{ftspName}-avg-sol.png", dpi=300)
    plt.close()
    
def get_results_path(variant, parameters):
    comb = f"{'Fixed' if parameters['fixed'] else 'Non-Fixed'}-M{parameters['mixed']}-R{parameters['num_run']}-I{int(math.log10(parameters['num_iteration'][0]))}"
    if "pso" in variant:
#         comb += f"-{'U' if parameters['U'] else ''}{parameters['topo']}-{'Random' if parameters['ini'] == 0 else 'Even'}-E{parameters['epoch']}-P{parameters['particles']}-W{parameters['w']}-C1{parameters['c1']}-C2{parameters['c2']}"
        comb += f"-{'U' if parameters['U'] else ''}{parameters['topo']}-{'Random' if parameters['ini'] == 0 else 'Even'}-E{parameters['max_evals']}-P{parameters['population_size']}-W{parameters['w']}-C1{parameters['c1']}-C2{parameters['c2']}"
    else:
        comb += f"-E{parameters['max_evals']}"
    if "fsp" in variant:
        comb += f"-{parameters['dis']}"
    # population
    if "pop" in variant:
        comb += f"-P{parameters['population_size']}-INR{parameters['update_interval']}"
    folder_path = f"./results/{variant}/{date.today()}/{comb}/"
    setting_path = f"{folder_path}/{comb}-Setting.txt"
    info_path = f"{folder_path}/{comb}-Info.csv"
    return comb, folder_path, setting_path, info_path


def get_results_path_all(variant, parameters):
    comb = f"{'Fixed' if parameters['fixed'] else 'Non-Fixed'}-M{parameters['mixed']}"
    if "pso" in variant:
        comb += f"-{'U' if parameters['U'] else ''}{parameters['topo']}-{'Random' if parameters['ini'] == 0 else 'Even'}-E{parameters['epoch']}-P{parameters['particles']}-W{parameters['w']}~{parameters['w0']}-C1{parameters['c1']}~{parameters['c10']}-C2{parameters['c2']}~{parameters['c20']}"
    else:
        comb += f"-E{parameters['max_evals']}"
    if "fsp" in variant:
        comb += f"-{parameters['dis']}"
    folder_path = f"./results/{variant}/{date.today()}/{comb}/"
    setting_path = f"{folder_path}/{comb}-Setting.txt"
    info_path = f"{folder_path}/{comb}-Info.csv"
    return comb, folder_path, setting_path, info_path

### large FSP
def restore(order, id_dict):
    """Restore the jobs' id of the seq_back to original one 
    """
    reversed_dict = dict(zip(id_dict.values(), id_dict.keys()))
    orig_order = [reversed_dict[i] for i in order]
    return orig_order


def main_fsp_cluster_iter(dirName, tuner, bounds, parameters, solve):
    def _sub(filePath):
        nonlocal bounds
        nonlocal parameters
        parser = Parser(filePath)
        fileName = parser._name
        seq = parser._seq
        dis = seq.job_distance(parameters["dis"])
        clusters = seq.cluster(
            parameters["clustering"], a=parameters["offset"], k=parameters["n_clusters"]
        )

        order = []
        start = time.time()
        for c in clusters:
            seq_c = Sequence(c[0])
            dis_c = seq_c.job_distance("nocarry")

            rules = dict(
                column=build_column_rule(seq_c._jobs),
                row=build_row_rule(seq_c._jobs),
                edge=build_edge_rule(perturb(seq_c._distance), seq_c._jobs),
                edge_unperturbed=build_edge_rule_unperturbed(
                    seq_c._distance, seq_c._jobs
                ),
            )

            cur_perm, cur_dis, cur_mixed, min_pos, best_epoch, elapsed_time = tuner(
                fileName, seq_c, rules, bounds, parameters, solve
            )
            seq_c_order = restore(cur_perm, c[-1])
            seq_c_order.remove(0)
            order.append(seq_c_order)
        run_time = time.time() - start

        min_makespan = float("inf")
        for o in itertools.permutations(order):
            o = sum(list(o), [0])
            seq.reorder(o)
            min_makespan = min(min_makespan, seq._makespan)
        print(f"\n{parser._name}")
        print(f"Best solution found: {min_makespan}")
        print(f"Elapsed time: {run_time}\n")
        print("*" * 36)

        # record the best solution found
        infoName = parameters["info_path"]
        if not os.path.exists(infoName):
            with open(infoName, "a") as f:
                f.write("FSP, Best_Solution, Elapsed_Time\n")
        with open(infoName, "a") as f:
            f.write("{},{},{}\n".format(Path(fileName).stem, min_makespan, run_time,))

    pathlib.Path(parameters["folder_path"]).mkdir(parents=True, exist_ok=True)

    if os.path.isdir(dirName):
        for fileName in os.listdir(dirName):
            filePath = os.path.join(dirName, fileName)
            _sub(filePath)
    else:
        filePath = dirName
        _sub(filePath)


def main_fsp_cluster_iter_thr(dirName, tuner, bounds, parameters, solve):
    def _sub(filePath):
        nonlocal bounds
        nonlocal parameters
        parser = Parser(filePath)
        fileName = parser._name
        seq = parser._seq
        dis = seq.job_distance(parameters["dis"])
        clusters = seq.cluster(
            parameters["clustering"], a=parameters["offset"], k=parameters["n_clusters"]
        )

        # start
        start = time.time()
        thrs = []
        que = queue.Queue()
        for i, c in enumerate(clusters):
            seq_c = Sequence(c[0])
            dis_c = seq_c.job_distance("nocarry")

            rules = dict(
                column=build_column_rule(seq_c._jobs),
                row=build_row_rule(seq_c._jobs),
                edge=build_edge_rule(perturb(seq_c._distance), seq_c._jobs),
                edge_unperturbed=build_edge_rule_unperturbed(
                    seq_c._distance, seq_c._jobs
                ),
            )

            thr = Thread(
                target=lambda q, arg: q.put((i, *tuner(*arg))),
                args=(que, (fileName, seq_c, rules, bounds, parameters, solve),),
            )
            thrs.append(thr)
            thr.start()
        for thr in thrs:
            thr.join()

        order = []
        while not que.empty():
            (
                i,
                cur_perm,
                cur_mksp,
                cur_mixed,
                min_pos,
                best_epoch,
                elapsed_time,
            ) = que.get()
            seq_c, dic = Sequence(clusters[i][0]), clusters[i][-1]
            seq_c_order = restore(cur_perm, c[-1])
            seq_c_order.remove(0)
            order.append(seq_c_order)
        run_time = time.time() - start

        min_makespan = float("inf")
        for o in itertools.permutations(order):
            o = sum(list(o), [0])
            seq.reorder(o)
            min_makespan = min(min_makespan, seq._makespan)
        print(f"\n{parser._name}")
        print(f"Best solution found: {min_makespan}")
        print(f"Elapsed time: {run_time}\n")
        print("*" * 36)

        # record the best solution found
        infoName = parameters["info_path"]
        if not os.path.exists(infoName):
            with open(infoName, "a") as f:
                f.write("FSP, Best_Solution, Elapsed_Time\n")
        with open(infoName, "a") as f:
            f.write("{},{},{}\n".format(Path(fileName).stem, min_makespan, run_time,))

    pathlib.Path(parameters["folder_path"]).mkdir(parents=True, exist_ok=True)

    if os.path.isdir(dirName):
        for fileName in os.listdir(dirName):
            filePath = os.path.join(dirName, fileName)
            _sub(filePath)
    else:
        filePath = dirName
        _sub(filePath)


def main_fsp_cluster(dirName, tuner, bounds, parameters, solve):
    def _sub(filePath):
        nonlocal bounds
        nonlocal parameters
        parser = Parser(filePath)
        fileName = parser._name
        seq = parser._seq
        dis = seq.job_distance(parameters["dis"])
        clusters = seq.cluster()

        order = []
        total_proc = []
        start = time.time()
        for c in clusters:
            seq_c = Sequence(c[0])
            dis_c = seq_c.job_distance("nocarry")

            rules = dict(
                column=build_column_rule(seq_c._jobs),
                row=build_row_rule(seq_c._jobs),
                edge=build_edge_rule(perturb(seq_c._distance), seq_c._jobs),
                edge_unperturbed=build_edge_rule_unperturbed(
                    seq_c._distance, seq_c._jobs
                ),
            )

            cur_perm, cur_mksp, cur_mixed, min_pos, best_epoch, elapsed_time = tuner(
                fileName, seq_c, rules, bounds, parameters, solve
            )
            seq_c_order = restore(seq_c, c[-1])
            seq_c_order.remove(0)
            order.append(seq_c_order)
            total_proc.append(np.sum(seq_c._proc) / seq_c._jobs)
        run_time = time.time() - start

        # sort clusters via total processing time in descending order
        sorted_indices = sorted(
            range(len(total_proc)), key=total_proc.__getitem__, reverse=True
        )
        sorted_order = sum([order[i] for i in sorted_indices], [])
        sorted_order.append(0)

        # order the original sequence
        seq.reorder(sorted_order)
        min_makespan = seq._makespan
        print(f"\n{parser._name}")
        print(f"Best solution found: {min_makespan}")
        print(f"Elapsed time: {run_time}\n")
        print("*" * 36)

        # record the best solution found
        infoName = parameters["info_path"]
        if not os.path.exists(infoName):
            with open(infoName, "a") as f:
                f.write("FSP, Best_Solution, Elapsed_Time\n")
        with open(infoName, "a") as f:
            f.write("{},{},{}\n".format(Path(fileName).stem, min_makespan, run_time,))

    pathlib.Path(parameters["folder_path"]).mkdir(parents=True, exist_ok=True)

    if os.path.isdir(dirName):
        for fileName in os.listdir(dirName):
            filePath = os.path.join(dirName, fileName)
            _sub(filePath)
    else:
        filePath = dirName
        _sub(filePath)


### large TSP
def two_opt(g, route):
    graph = g
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route)):
                if j - i == 1:
                    continue
                if (
                    TSP.cost_change(graph, best[i - 1], best[i], best[j - 1], best[j])
                    < 0
                ):
                    best[i:j] = best[j - 1 : i - 1 : -1]
                    improved = True
        route = best
    dist = 0
    for i in range(len(route)):
        dist += graph[route[i]][route[(i + 1) % len(route)]]
    return dist, best


def stitch_tsp_cluster(n, g, num_run=128, num_iteration=10 ** 8):
    
    global param, server_list

    # prepare the QUBO
    rules = dict(
        column=build_column_rule(n),
        row=build_row_rule(n),
        # perturb(g)
        edge=build_edge_rule(g, n),
        edge_unperturbed=build_edge_rule_unperturbed(g, n),
    )

    qubo = build_QUBO(n, [1.5 * g[g != np.inf].max()], rules)
    N_state, bias, constant, weight, s, local_field, E = prepare_config(qubo)

    args = {
        "weight": weight,
        "lf_i_n": np.tile(local_field, num_run),
        "state_min_i_n": np.tile(s, num_run),
        "state_cur_i_n": np.tile(s, num_run),
        "eg_min_i_n": np.full(num_run, E, dtype=int),
        "eg_cur_i_n": np.full(num_run, E, dtype=int),
        "num_iteration": num_iteration,
        "num_bit": N_state,
        "num_run": num_run,
        "ope_mode": 2,
    }

    da = fjda_client.fjda(server=random.choice(server_list))

    # send to DA
    da.setAnnealParameterMM(param)

    received = False
    result = da.doAnnealMM(args, format="numpy", compression="lz4")
    while not received:
        if result:
            print("\nStitched")
            received = True

    # derive solution
    elapsed_time = result["anneal_time"] / 1000
    return (
        {
            idx: bool(res)
            for idx, res in enumerate(
                result["state_min_o_n.numpy"][result["eg_min_o_n.numpy"].argmin()]
            )
            if idx < n ** 2
        },
        result["eg_min_o_n.numpy"][result["eg_min_o_n.numpy"].argmin()],
    )


def main_tsp_cluster(dirName, tuner, bounds, parameters, solve):
    def _sub(filePath):
        nonlocal bounds
        nonlocal parameters
        fileName = Path(filePath).stem
        tsp = TSP(tsplib95.load(filePath))
        clusters, centers = tsp.k_part()
        stitcher = Stitcher(clusters, tsp._g)
        min_cost = float("inf")

        start = time.time()
        for t in stitcher._clusters.values():

            rules = dict(
                column=build_column_rule(len(t._nodes)),
                row=build_row_rule(len(t._nodes)),
                edge=build_edge_rule(perturb(t._g), len(t._nodes)),
                edge_unperturbed=build_edge_rule_unperturbed(t._g, len(t._nodes)),
            )

            cur_perm, cur_dis, cur_mixed, min_pos, best_epoch, elapsed_time = tuner(
                fileName, t, rules, bounds, parameters, solve
            )
            t._path = cur_perm
            t._cost = cur_dis

        # stitch
        join, flip_cost, flip_set = stitcher.generate_join_scheme()
        if len(join) > 0:
            stitcher.stitch(join, flip_cost, flip_set)
            run_time = time.time() - start

            min_cost = min(
                sum(
                    [
                        tsp._g[
                            stitcher._path[i],
                            stitcher._path[(i + 1) % tsp._nodes.__len__()],
                        ]
                        for i in range(tsp._nodes.__len__())
                    ]
                ),
                min_cost,
            )

            # stitcher._cost, stitcher._path = tsp.get_two_opt(stitcher._path)
            stitcher._cost, stitcher._path = tsp.get_two_opt(tsp._g, stitcher._path)
        else:
            min_cost = -1
            run_time = time.time() - start
        print(f"\n{tsp._name}")
        print(f"Best solution found: {min_cost}")
        print(f"Two-opt solution: {stitcher._cost if min_cost != -1 else -1}")
        print(f"Elapsed time: {run_time}\n")
        print("*" * 36)

        # record the best solution found
        infoName = parameters["info_path"]
        if not os.path.exists(infoName):
            with open(infoName, "a") as f:
                f.write("TSP, Best_Solution, Two_Opt, Elapsed_Time\n")
        with open(infoName, "a") as f:
            f.write(
                "{},{},{},{}\n".format(
                    Path(fileName).stem,
                    min_cost,
                    stitcher._cost if min_cost != -1 else -1,
                    run_time,
                )
            )

    pathlib.Path(parameters["folder_path"]).mkdir(parents=True, exist_ok=True)

    if os.path.isdir(dirName):
        for fileName in os.listdir(dirName):
            filePath = os.path.join(dirName, fileName)
            _sub(filePath)
    else:
        filePath = dirName
        _sub(filePath)

### QAP
# qbsolv da
def solve_min_qap_config_qbsolv_da(qap, rules, para, num_run, num_iteration, prep_config, conf, seed=None, 
                                num_repeats=1, # Determines the number of times to repeat the main loop in qbsolv after determining a better sample. Default 50.
                                verbosity=0, # Prints more detail about qbsolv’s internal process as this number increases.
                                timeout=60, # Number of seconds before routine halts. Default is 2592000.
                                solver_limit=900, # Maximum number of variables in a sub problem.
                                find_max=False # Switches from searching for minimization to maximization. Default is False (minimization)
                                ):
    
    global param, server_list

    # prepare the QUBO
    qubo = build_QAP(qap, para, rules)
#     print(np.count_nonzero(qubo.array))
#     idx = np.transpose(np.nonzero(qubo.array))
    # transform to dictionary format
    Q = {}
    for i in range(qubo.array.shape[0]):
        for j in range(qubo.array.shape[1]):
            if qubo.array[i, j] != 0:
                Q[(i, j)] = qubo.array[i, j]
#     for i in range(qubo.array.shape[0]):
#         Q.update(dict(zip([(i,j) for j in  range(qubo.array.shape[1])], qubo.array[i])))
    # DA sampler
    da = DAU(param, server_list, num_run, num_iteration, prep_config, seed, solver_limit)
    sampler = da.sample
    print("Start")

    # send to qbsolv
    start = time.time()
    response = QBSolv().sample_qubo(Q, solver=sampler, num_repeats=num_repeats, verbosity=verbosity, timeout=timeout, solver_limit=solver_limit, find_max=find_max)
    results = list(response.samples()) # [{0: 1, 1: 1, 2: 1}, ...]
    energies = list(response.data_vectors['energy'])

    # derive solution based on the minimum energy achieved
    min_energy = np.min(energies)
    min_dec, min_cost, min_conf = None, float("inf"), None
    feasible_flag = False # flag to indicate whether exists a feasible solution
    min_idx = np.argmin(energies)
    min_conf = results[min_idx]
    dec, cost = compute_qap_cost(qap, min_conf)
    
    if cost >= 0 and cost < min_cost:
        feasible_flag = True
        min_dec, min_cost = dec, cost

    # elapsed_time
    elapsed_time = time.time() - start - da.elapsed_time + da.anneal_time
    if not feasible_flag:
        return np.array([*para]), None, float("inf"), float("inf"), min_conf, elapsed_time, min_energy
    return np.array([*para]), min_dec, min_cost, min_cost, min_conf, elapsed_time, min_energy

# qbsolv neal
def solve_min_qap_config_qbsolv(qap, rules, para, 
                                num_run=None, num_iteration=None, prep_config=None, conf=None, seed=None, # unused but tuners are implemented with these args
                                solver=neal.SimulatedAnnealingSampler(), # Sampling method for qbsolv to use;
                                num_repeats=1, # Determines the number of times to repeat the main loop in qbsolv after determining a better sample. Default 50.
                                verbosity=0, # Prints more detail about qbsolv’s internal process as this number increases.
                                timeout=60, # Number of seconds before routine halts. Default is 2592000.
                                solver_limit=900, # Maximum number of variables in a sub problem.
                                find_max=False # Switches from searching for minimization to maximization. Default is False (minimization)
                                ):

    # prepare the QUBO
    qubo = build_QAP(qap, para, rules)
    # transform to dictionary format
    Q = {}
    for i in range(qubo.array.shape[0]):
        for j in range(qubo.array.shape[1]):
            if qubo.array[i, j] != 0:
                Q[(i, j)] = qubo.array[i, j]
    print("Start")

    # send to qbsolv
    response = QBSolv().sample_qubo(Q, solver=solver, num_repeats=num_repeats, verbosity=verbosity, timeout=timeout, solver_limit=solver_limit, find_max=find_max)
    # results = [response.samples()[sampleIdx] for sampleIdx in range(response.samples().__len__())]
    # energies = [response.energies()[sampleIdx] for sampleIdx in range(response.energies().__len__())]
    results = list(response.samples()) # [{0: 1, 1: 1, 2: 1}, ...]
    # energies = list(response.energies())
    energies = list(response.data_vectors['energy'])

    # start time
    start = time.time()
    # derive solution based on the minimum energy achieved
    min_energy = np.min(energies)
    min_dec, min_cost, min_conf = None, float("inf"), None
    feasible_flag = False # flag to indicate whether exists a feasible solution
    min_idx = np.argmin(energies)
    min_conf = results[min_idx]
    dec, cost = compute_qap_cost(qap, min_conf)
    
    if cost >= 0 and cost < min_cost:
        feasible_flag = True
        min_dec, min_cost = dec, cost

    # elapsed_time
    elapsed_time = time.time() - start
    if not feasible_flag:
        return np.array([*para]), None, float("inf"), float("inf"), min_conf, elapsed_time, min_energy
    return np.array([*para]), min_dec, min_cost, min_cost, min_conf, elapsed_time, min_energy

# DA
def solve_min_qap_config(qap, rules, para, num_run, num_iteration, prep_config, conf=None, seed=None):
    
    global param, server_list

    # prepare the QUBO
    qubo = build_QAP(qap, para, rules)
    N_state, bias, constant, weight, s, local_field, E = prep_config(qubo, conf, seed)

    args = {
        "weight": weight,
        "lf_i_n": np.tile(local_field, num_run),
        "state_min_i_n": np.tile(s, num_run),
        "state_cur_i_n": np.tile(s, num_run),
        "eg_min_i_n": np.full(num_run, E, dtype=int),
        "eg_cur_i_n": np.full(num_run, E, dtype=int),
        "num_iteration": num_iteration,
        "num_bit": N_state,
        "num_run": num_run,
        "ope_mode": 2,
    }

    da = fjda_client.fjda(server=random.choice(server_list))

    # send to DA
    da.setAnnealParameterMM(param)

    received = False
    res = da.doAnnealMM(args, format="numpy", compression="lz4")
    while not received:
        if res:
            print(f"Received")
            received = True

    # start time
    start = time.time()
    # derive solution based on the minimum energy achieved
    min_energy = np.min(res['eg_min_o_n.numpy'])
    min_dec, min_cost, min_conf = None, float("inf"), None
    feasible_flag = False # flag to indicate whether exists a feasible solution
    min_idx = res['eg_min_o_n.numpy'].argmin()
    min_conf = res["state_min_o_n.numpy"][min_idx]
    dec, cost = compute_qap_cost(qap, min_conf)
    
    if cost >= 0 and cost < min_cost:
        feasible_flag = True
        min_dec, min_cost = dec, cost

    # elapsed_time
    elapsed_time = time.time() - start + res["anneal_time"] / 1000
    if not feasible_flag:
        return np.array([*para]), None, float("inf"), float("inf"), min_conf, elapsed_time, min_energy
    return np.array([*para]), min_dec, min_cost, min_cost, min_conf, elapsed_time, min_energy


def solve_avg_qap_config(qap, rules, A, num_run, num_iteration):
    
    global param, server_list

    # prepare the QUBO
    qubo = build_QAP(qap, para, rules)
    N_state, bias, constant, weight, s, local_field, E = prep_config(qubo, conf, seed)

    args = {
        "weight": weight,
        "lf_i_n": np.tile(local_field, num_run),
        "state_min_i_n": np.tile(s, num_run),
        "state_cur_i_n": np.tile(s, num_run),
        "eg_min_i_n": np.full(num_run, E, dtype=int),
        "eg_cur_i_n": np.full(num_run, E, dtype=int),
        "num_iteration": num_iteration,
        "num_bit": N_state,
        "num_run": num_run,
        "ope_mode": 2,
    }

    da = fjda_client.fjda(server=random.choice(server_list))

    # send to DA
    da.setAnnealParameterMM(param)

    received = False
    res = da.doAnnealMM(args, format="numpy", compression="lz4")
    while not received:
        if res:
            print(f"Received")
            received = True

    start = time.time()
    # derive solution
    min_dec, min_cost, min_conf = None, float("inf"), None
    min_energy = float("inf")
    costs = []
    feasible_flag = False # flag to indicate whether exists a feasible solution
    for idx, conf in enumerate(res["state_min_o_n.numpy"]):
        dec, cost = compute_qap_cost(qap, conf)

        if cost > 0: # feasible solution
            feasible_flag = True
            costs.append(cost)
            if cost < min_cost: 
                min_dec, min_cost, min_conf = dec, cost, conf
                min_energy = res["eg_min_o_n.numpy"][idx]
            
        if cost <= 0: # infeasible solution
            energy = res["eg_min_o_n.numpy"][idx]
            if energy < min_energy and not feasible_flag:
                min_conf = conf
                min_energy = energy
    
    # elapsed_time
    elapsed_time = time.time() - start + res["anneal_time"] / 1000
    if min_dec is None:
        return np.array([*para]), None, float("inf"), float("inf"), min_conf, elapsed_time, min_energy
    return np.array([*para]), min_dec, min_cost, np.mean(costs), min_conf, elapsed_time, min_energy

def main_qap(dirName, tuner, bounds, parameters, solve, prep_config):
    def _parse(fileName):
        with open(fileName, "r") as f:
            lines = list(filter(None, (line.rstrip() for line in f)))
            n = int(lines[0])
            a = sum([list(map(int, line.split())) for line in lines[1:]], [])
            d = np.array(a[:n**2]).reshape(n, n).astype(int)
            f = np.array(a[n**2:]).reshape(n, n).astype(int)
        return n, d, f

    def _sub(filePath):
        nonlocal bounds
        nonlocal parameters
        
        fileName = Path(filePath).stem
        if filePath.endswith(".dat"):
            n, d, f = _parse(filePath)
            qap = QAP(d, f, parameters["perturbed"])
            # create rules for building qubo
            rules = dict(
            row=build_qap_row_sum(qap), 
            column=build_qap_col_sum(qap), 
            obj=build_qap_obj(qap),
            )
        elif filePath.endswith(".pk"):
            with open(filePath, 'rb') as f:
                qap = pickle.load(f)
            p = Path(filePath)
            with open(p.parent / "{}-{}.npy".format(p.stem, 'con1'), 'rb') as f1:
                con1 = np.load(f1)
            with open(p.parent / "{}-{}.npy".format(p.stem, 'con2'), 'rb') as f2:
                con2 = np.load(f2)
            with open(p.parent / "{}-{}.npy".format(p.stem, 'obj'), 'rb') as f3:
                obj = np.load(f3)
            rules = dict(
                row=con1, 
                column=con2, 
                obj=obj,
            )
        start = time.time()
        dec, cost, mixed_cost, best_para, best_iter, elapsed_time = tuner(
            fileName, qap, rules, bounds, parameters, solve, prep_config
        )
        run_time = time.time() - start
        # if len(bounds["min"]) > 1:
        #     if "pso" in parameters["folder_path"]:
        #         plot_trajectory(parameters, fileName)
        #         plot_avg_parameters(parameters, fileName)
        #         plot_avg_sol(parameters, fileName)
        #     plot_parameters(parameters, fileName)
        #     plot_sol(parameters, fileName)
        print(f"\n{fileName}")
        print(f"Best solution found: {cost}")
        for i in range(len(bounds['min'])):
            print(f"Best {chr(65 + i)} found: {best_para[i]}")
        print(f"Elapsed time: {run_time}\n")
        print("*" * 36)

        # record the best solution found
        infoName = parameters["info_path"]
        if not os.path.exists(infoName):
            with open(infoName, "a") as f:
                f.write(
                    "QAP, Best_Iter, "
                    + ", ".join(["Best_" + chr(65 + i) for i in range(len(bounds['min']))])
                    + ", Best_Solution, Best_Mixed, Anneal_Time, Elapsed_Time\n"
                )
        with open(infoName, "a") as f:
            f.write(
                f"{Path(fileName).stem}, {best_iter + 1}, "
                + ", ".join([f"{best_para[i]}" for i in range(len(bounds["min"]))])
                + f", {cost}, {mixed_cost}, {elapsed_time}, {run_time}\n"
            )

    pathlib.Path(parameters["folder_path"]).mkdir(parents=True, exist_ok=True)

    # if os.path.isdir(dirName):
    #     for fileName in os.listdir(dirName):
    #         filePath = os.path.join(dirName, fileName)
    #         _sub(filePath)
    # else:
    #     filePath = dirName
    #     _sub(filePath)
    for fileName in sorted(glob.glob(dirName)):
        try:
            _sub(fileName)
            gc.collect()
            time.sleep(120)
        except:
            gc.collect()
            time.sleep(120)
            continue
        
def main_qap_clustering(dirName, tuner, bounds, parameters, solve, prep_config):
    def _parse(fileName):
        with open(fileName, "r") as f:
            lines = list(filter(None, (line.rstrip() for line in f)))
            n = int(lines[0])
            a = sum([list(map(int, line.split())) for line in lines[1:]], [])
            d = np.array(a[:n**2]).reshape(n, n).astype(int)
            f = np.array(a[n**2:]).reshape(n, n).astype(int)
        return n, d, f
    
    def _cal_product(n, d, f, sta):
        if sta == 'mean':
            f_m = [np.mean(f[i]) for i in range(n)]
            d_m = [np.mean(d[i]) for i in range(n)]
        elif sta == 'median':
            f_m = [np.median(f[i]) for i in range(n)]
            d_m = [np.median(d[i]) for i in range(n)]
        else:
            f_m = [np.sum(f[i]) for i in range(n)]
            d_m = [np.sum(d[i]) for i in range(n)]
        f_order = list(np.argsort(f_m))[::-1]
        d_order = np.argsort(d_m)
        products = []
        pairs = []
        for i in range(n):
            products.append(f_m[f_order[i]] * d_m[d_order[i]])
            pairs.append((f_order[i], d_order[i]))
        return products, pairs
    
    def _create_cluster(n, d, f, sta, n_clusters, c=40):
        products, pairs = _cal_product(n, d, f, sta)
        X = np.array(products).reshape(n, 1)
        # clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
        clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(X)
        labels = clustering.labels_
        clusters = []
        for i in range(n_clusters):
            index = np.where(labels==i)[0].tolist()
            if len(index) > 60:
                # k = len(index) // 30
                k = math.ceil(len(index) / c)
                for j in range(k):
                    if j != k - 1:
                        sub_index = index[j*c:(j+1)*c]
                        clusters.append([pairs[x] for x in sub_index])
                    else:
                        sub_index = index[j*c:]
                        clusters.append([pairs[x] for x in sub_index])
            else:
                clusters.append([pairs[x] for x in index])
        return clusters
    
    def _form_sub_qap(d, f, cluster):
        d_cluster = d[np.array([x[1] for x in cluster])][:, np.array([x[1] for x in cluster])]
#         d_cluster = d_cluster[:, np.array([x[1] for x in cluster])]
        f_cluster = f[np.array([x[0] for x in cluster])][:, np.array([x[0] for x in cluster])]
#         f_cluster = f_cluster[:, np.array([x[0] for x in cluster])]
        return d_cluster, f_cluster
    
    def _project_full_dec(full_dec, cluster, cluster_dec):
        f_back_dict = dict(zip(range(len(cluster)), [x[0] for x in cluster]))
        d_back_dict = dict(zip(range(len(cluster)), [x[1] for x in cluster]))
        for (i, j) in cluster_dec.keys():
            full_dec[(f_back_dict[i], d_back_dict[j])] = cluster_dec[(i, j)]
        return full_dec
    
    def _compute_cost(full_dec, n, d, f):
        cost = 0
        for (i, j, k, l) in itertools.product(range(n), range(n), range(n), range(n)):
            cost += f[i][j] * d[k][l] * full_dec.get((i, k), 0) * full_dec.get((j, l), 0)
        return cost

    def _sub(filePath):
        nonlocal bounds
        nonlocal parameters

        n, d, f = _parse(filePath)
        fileName = Path(filePath).stem
        print(f'\n{fileName}')
        n_clusters = n // 40 # 30
        clusters = _create_cluster(n, d, f, parameters["sta"], n_clusters)
        print('Number of the clusters: {}'.format(len(clusters)))
        print("Size of the clusters: {}".format([len(c) for c in clusters]))
        full_dec = {}
        total_elapsed, total_run = 0, 0
        for idx, cluster in enumerate(clusters):
            print('\nSize of cluster {}: {}'.format(idx, len(cluster)))
            if len(cluster) > 1:
                d_cluster, f_cluster = _form_sub_qap(d, f, cluster)
                sub_qap = QAP(d_cluster, f_cluster, parameters["perturbed"])
                rules = dict(row=build_qap_row_sum(sub_qap), 
                            column=build_qap_col_sum(sub_qap), 
                            obj=build_qap_obj(sub_qap),
                )
                start = time.time()
                dec, cost, mixed_cost, best_para, best_iter, elapsed_time = tuner(
                    '{}-{}'.format(fileName, idx), sub_qap, rules, bounds, parameters, solve, prep_config
                )
                run_time = time.time() - start
                total_elapsed += elapsed_time
                total_run += run_time
            else:
                dec = {cluster[0]:1}
                cost, mixed_cost, best_para, best_iter, elapsed_time, run_time = 0, 0, [0]*len(bounds['min']), -1, 0, 0
            print(f"\n{fileName}-{idx}")
            print(f"Best solution found: {cost}")
            for i in range(len(bounds['min'])):
                print(f"Best {chr(65 + i)} found: {best_para[i]}")
            print(f"Elapsed time: {run_time}\n")
            print("*" * 36)
            # no feasible solution
            if cost == float('inf'):
                # return random permutation
                f_order = list(range(len(cluster)))
                d_order = list(range(len(cluster)))
                random.shuffle(f_order)
                random.shuffle(d_order)
                dec = {}
                for i in range(len(cluster)):
                    dec[(f_order[i], d_order[i])] = 1

            # record the best solution found
            infoName = parameters["info_path"]
            if not os.path.exists(infoName):
                with open(infoName, "a") as f1:
                    f1.write(
                        "QAP, Best_Iter, "
                        + ", ".join(["Best_" + chr(65 + i) for i in range(len(bounds['min']))])
                        + ", Best_Solution, Best_Mixed, Anneal_Time, Elapsed_Time\n"
                    )
            with open(infoName, "a") as f2:
                f2.write(
                    f"{Path(fileName).stem}-{idx}, {best_iter + 1}, "
                    + ", ".join([f"{best_para[i]}" for i in range(len(bounds["min"]))])
                    + f", {cost}, {mixed_cost}, {elapsed_time}, {run_time}\n"
                )
            # project to full solution
            if len(cluster) > 1:
                full_dec = _project_full_dec(full_dec, cluster, dec)
            else:
                full_dec[cluster[0]] = 1
        # calculate the cost
        print("Calculate full solution")
        full_cost = _compute_cost(full_dec, n, d, f)
        print(f"Full solution found: {full_cost}")
        # record the full solution found
        with open(infoName, "a") as f3:
            f3.write(
                f"{Path(fileName).stem}, 0, "
                + ", ".join(["0" for i in range(len(bounds["min"]))])
                + f", {full_cost}, {full_cost}, {total_elapsed}, {total_run}\n"
            )

    pathlib.Path(parameters["folder_path"]).mkdir(parents=True, exist_ok=True)

    for fileName in sorted(glob.glob(dirName)):
        _sub(fileName)
        
def main_qap_clustering_mod(dirName, tuner, bounds, parameters, solve, prep_config):
    def _parse(fileName):
        with open(fileName, "r") as f:
            lines = list(filter(None, (line.rstrip() for line in f)))
            n = int(lines[0])
            a = sum([list(map(int, line.split())) for line in lines[1:]], [])
            d = np.array(a[:n**2]).reshape(n, n).astype(int)
            f = np.array(a[n**2:]).reshape(n, n).astype(int)
        return n, d, f
    
    def _cal_product(n, d, f, sta):
        if sta == 'mean':
            f_m = [np.mean(f[i]) for i in range(n)]
            d_m = [np.mean(d[i]) for i in range(n)]
        elif sta == 'median':
            f_m = [np.median(f[i]) for i in range(n)]
            d_m = [np.median(d[i]) for i in range(n)]
        else:
            f_m = [np.sum(f[i]) for i in range(n)]
            d_m = [np.sum(d[i]) for i in range(n)]
        f_order = list(np.argsort(f_m))[::-1]
        d_order = np.argsort(d_m)
        products = []
        pairs = []
        for i in range(n):
            products.append(f_m[f_order[i]] * d_m[d_order[i]])
            pairs.append((f_order[i], d_order[i]))
        f_std = [np.std(f[i]) for i in range(n)]
        d_std = [np.std(d[i]) for i in range(n)]
        f_median = [np.median(f[i]) for i in range(n)]
        d_median = [np.median(d[i]) for i in range(n)]
        return np.array(f_std)[f_order], np.array(d_std)[d_order], np.array(f_median)[f_order], np.array(d_median)[d_order], products, pairs
    
    def _create_cluster(n, d, f, sta, n_clusters, c=40):
        f_std, d_std, f_median, d_median, products, pairs = _cal_product(n, d, f, sta)
        X = np.array([f_std, d_std, f_median, d_median, products]).reshape(5, n).T
        scaler = preprocessing.StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        # clustering = KMeans(n_clusters=n_clusters, random_state=0).fit(X_scaled)
        clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(X_scaled)
        labels = clustering.labels_
        clusters = []
        for i in range(n_clusters):
            index = np.where(labels==i)[0].tolist()
            if len(index) > 60: # 40 when c == 30
                # k = len(index) // 30
                k = math.ceil(len(index) / c)
                for j in range(k):
                    if j != k - 1:
                        sub_index = index[j*c:(j+1)*c]
                        clusters.append([pairs[x] for x in sub_index])
                    else:
                        sub_index = index[j*c:]
                        clusters.append([pairs[x] for x in sub_index])
            else:
                clusters.append([pairs[x] for x in index])
        return clusters
    
    def _form_sub_qap(d, f, cluster):
        d_cluster = d[np.array([x[1] for x in cluster])][:, np.array([x[1] for x in cluster])]
#         d_cluster = d_cluster[:, np.array([x[1] for x in cluster])]
        f_cluster = f[np.array([x[0] for x in cluster])][:, np.array([x[0] for x in cluster])]
#         f_cluster = f_cluster[:, np.array([x[0] for x in cluster])]
        return d_cluster, f_cluster
    
    def _project_full_dec(full_dec, cluster, cluster_dec):
        f_back_dict = dict(zip(range(len(cluster)), [x[0] for x in cluster]))
        d_back_dict = dict(zip(range(len(cluster)), [x[1] for x in cluster]))
        for (i, j) in cluster_dec.keys():
            full_dec[(f_back_dict[i], d_back_dict[j])] = cluster_dec[(i, j)]
        return full_dec
    
    def _compute_cost(full_dec, n, d, f):
        cost = 0
        for (i, j, k, l) in itertools.product(range(n), range(n), range(n), range(n)):
            cost += f[i][j] * d[k][l] * full_dec.get((i, k), 0) * full_dec.get((j, l), 0)
        return cost

    def _sub(filePath):
        nonlocal bounds
        nonlocal parameters
        
        n, d, f = _parse(filePath)
        fileName = Path(filePath).stem
        print(f'\n{fileName}')
        n_clusters = n // 40
        clusters = _create_cluster(n, d, f, parameters["sta"], n_clusters)
        print('Number of the clusters: {}'.format(len(clusters)))
        print("Size of the clusters: {}".format([len(c) for c in clusters]))
        full_dec = {}
        total_elapsed, total_run = 0, 0
        for idx, cluster in enumerate(clusters):
            print('\nSize of cluster {}: {}'.format(idx, len(cluster)))
            if len(cluster) > 1:
                d_cluster, f_cluster = _form_sub_qap(d, f, cluster)
                sub_qap = QAP(d_cluster, f_cluster, parameters["perturbed"])
                rules = dict(row=build_qap_row_sum(sub_qap), 
                            column=build_qap_col_sum(sub_qap), 
                            obj=build_qap_obj(sub_qap),
                )
                start = time.time()
                dec, cost, mixed_cost, best_para, best_iter, elapsed_time = tuner(
                    '{}-{}'.format(fileName, idx), sub_qap, rules, bounds, parameters, solve, prep_config
                )
                run_time = time.time() - start
                total_elapsed += elapsed_time
                total_run += run_time
            else:
                dec = {cluster[0]:1}
                cost, mixed_cost, best_para, best_iter, elapsed_time, run_time = 0, 0, [0]*len(bounds['min']), -1, 0, 0
            print(f"\n{fileName}-{idx}")
            print(f"Best solution found: {cost}")
            for i in range(len(bounds['min'])):
                print(f"Best {chr(65 + i)} found: {best_para[i]}")
            print(f"Elapsed time: {run_time}\n")
            print("*" * 36)
            # no feasible solution
            if cost == float('inf'):
                # return random permutation
                f_order = list(range(len(cluster)))
                d_order = list(range(len(cluster)))
                random.shuffle(f_order)
                random.shuffle(d_order)
                dec = {}
                for i in range(len(cluster)):
                    dec[(f_order[i], d_order[i])] = 1

            # record the best solution found
            infoName = parameters["info_path"]
            if not os.path.exists(infoName):
                with open(infoName, "a") as f1:
                    f1.write(
                        "QAP, Best_Iter, "
                        + ", ".join(["Best_" + chr(65 + i) for i in range(len(bounds['min']))])
                        + ", Best_Solution, Best_Mixed, Anneal_Time, Elapsed_Time\n"
                    )
            with open(infoName, "a") as f2:
                f2.write(
                    f"{Path(fileName).stem}-{idx}, {best_iter + 1}, "
                    + ", ".join([f"{best_para[i]}" for i in range(len(bounds["min"]))])
                    + f", {cost}, {mixed_cost}, {elapsed_time}, {run_time}\n"
                )
            # project to full solution
            if len(cluster) > 1:
                full_dec = _project_full_dec(full_dec, cluster, dec)
            else:
                full_dec[cluster[0]] = 1
        # calculate the cost
        print("Calculate full solution")
        full_cost = _compute_cost(full_dec, n, d, f)
        print(f"Full solution found: {full_cost}")
        # record the full solution found
        with open(infoName, "a") as f3:
            f3.write(
                f"{Path(fileName).stem}, 0, "
                + ", ".join(["0" for i in range(len(bounds["min"]))])
                + f", {full_cost}, {full_cost}, {total_elapsed}, {total_run}\n"
            )

    pathlib.Path(parameters["folder_path"]).mkdir(parents=True, exist_ok=True)

    for fileName in sorted(glob.glob(dirName)):
        _sub(fileName)