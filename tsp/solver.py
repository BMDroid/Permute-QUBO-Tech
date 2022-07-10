import itertools
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

from celluloid import Camera
from datetime import date, datetime
from matplotlib import cm
from matplotlib.animation import PillowWriter
from pathlib import Path
from python_fjda import fjda_client
from threading import Thread
from tqdm import tqdm_notebook as tqdm

from builder import *
from sequence import *
from tsp import *
from qap import *
from stitcher import *


def solve_fsp(seq, rules, A, num_run, num_iteration):

    # prepare the QUBO
    qubo = build_QUBO(seq._jobs, A, rules)
    N_state, bias, constant, weight, s, local_field, E = prepare_config(qubo)

    # args and params
    param = {
        "offset_mode": 3,
        "offset_inc_rate": 1000,
        "tmp_st": 1,
        "tmp_decay": 0,
        "tmp_mode": 0,
        "tmp_interval": 100,
    }

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

    server_list = [
        "dau2-05-0.da.labs.fujitsu.com",
        "dau2-05-1.da.labs.fujitsu.com",
        "dau2-06-0.da.labs.fujitsu.com",
        "dau2-06-1.da.labs.fujitsu.com",
    ]
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

    # prepare the QUBO
    qubo = build_QUBO(seq._jobs, A, rules)
    N_state, bias, constant, weight, s, local_field, E = prepare_config(qubo)

    # args and params
    param = {
        "offset_mode": 3,
        "offset_inc_rate": 1000,
        "tmp_st": 1,
        "tmp_decay": 0,
        "tmp_mode": 0,
        "tmp_interval": 100,
    }

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

    server_list = [
        "dau2-05-0.da.labs.fujitsu.com",
        "dau2-05-1.da.labs.fujitsu.com",
        "dau2-06-0.da.labs.fujitsu.com",
        "dau2-06-1.da.labs.fujitsu.com",
    ]
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
            edge_unperturbed=build_edge_rule_unperturbed(
                seq._distance, seq._jobs),
        )
        start = time.time()
        cur_perm, cur_mksp, cur_mixed, min_pos, best_epoch, elapsed_time = tuner(
            fileName, seq, rules, bounds, parameters, solve)
        run_time = time.time() - start
        start = time.time()
        two_opt_mksp, _ = seq.two_opt(cur_perm, cur_mksp) # conduct two opt given the best solution found
        post_time = time.time() - start
        if len(bounds['min']) > 1:
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
                f.write("FSP, Best_Epoch, " + ', '.join(
                    ["Best_" + chr(65 + i) for i in range(len(bounds['min']))]
                ) + ", Best_Solution, Two_Opt, Anneal_Time, Elapsed_Time, Two_Opt_Time\n"
                        )
        with open(infoName, "a") as f:
            f.write(
                f"{Path(fileName).stem},{best_epoch + 1}," +
                ','.join([f"{min_pos[i]}"
                          for i in range(len(bounds['min']))]) +
                f",{cur_mksp},{two_opt_mksp},{elapsed_time},{run_time},{post_time}\n"
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
def solve_tsp(tsp, rules, A, num_run, num_iteration):

    # prepare the QUBO
    qubo = build_QUBO(tsp._no_nodes, A, rules)
    N_state, bias, constant, weight, s, local_field, E = prepare_config(qubo)

    # args and params
    param = {
        "offset_mode": 3,
        "offset_inc_rate": 1000,
        "tmp_st": 1,
        "tmp_decay": 0,
        "tmp_mode": 0,
        "tmp_interval": 100,
    }

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

    server_list = [
        "dau2-05-0.da.labs.fujitsu.com",
        "dau2-05-1.da.labs.fujitsu.com",
        "dau2-06-0.da.labs.fujitsu.com",
        "dau2-06-1.da.labs.fujitsu.com",
    ]
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
    min_path, min_distance = [], float("inf")
    for sol in res["state_min_o_n.numpy"]:
        da_path, da_distance = compute_path_distance(tsp._no_nodes, sol,
                                                     tsp._g)
        if da_distance > 0 and da_distance < min_distance:
            min_path, min_distance = da_path, da_distance

    # elapsed_time
    elapsed_time = time.time() - start + res["anneal_time"] / 1000
    if len(min_path) == 0:
        return A, [], float("inf"), float("inf"), elapsed_time
    return A, min_path, min_distance, min_distance, elapsed_time


def solve_tsp_avg(tsp, rules, A, num_run, num_iteration):

    # prepare the QUBO
    qubo = build_QUBO(tsp._no_nodes, A, rules)
    N_state, bias, constant, weight, s, local_field, E = prepare_config(qubo)

    # args and params
    param = {
        "offset_mode": 3,
        "offset_inc_rate": 1000,
        "tmp_st": 1,
        "tmp_decay": 0,
        "tmp_mode": 0,
        "tmp_interval": 100,
    }

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

    server_list = [
        "dau2-05-0.da.labs.fujitsu.com",
        "dau2-05-1.da.labs.fujitsu.com",
        "dau2-06-0.da.labs.fujitsu.com",
        "dau2-06-1.da.labs.fujitsu.com",
    ]
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
    min_path, min_distance, dis_list = [], float("inf"), []
    for sol in res["state_min_o_n.numpy"]:
        da_path, da_distance = compute_path_distance(tsp._no_nodes, sol,
                                                     tsp._g)
        if da_distance > 0:
            dis_list.append(da_distance)
            if da_distance < min_distance:
                min_distance = da_distance
                min_path = da_path

    # elapsed_time
    elapsed_time = time.time() - start + res["anneal_time"] / 1000
    return (
        A,
        min_path,
        min_distance,
        np.mean(dis_list) if len(dis_list) > 0 else float("inf"),
        elapsed_time,
    )


def main_tsp(dirName, tuner, bounds, parameters, solve):
    def _sub(filePath):
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
            edge_unperturbed=build_edge_rule_unperturbed(
                tsp._g, tsp._no_nodes),
        )
        start = time.time()
        cur_perm, cur_dis, cur_mixed, min_pos, best_epoch, elapsed_time = tuner(
            fileName, tsp, rules, bounds, parameters, solve)
        run_time = time.time() - start
        if len(bounds['min']) > 1:
            if "pso" in parameters["folder_path"]:
                plot_trajectory(parameters, fileName)
                plot_avg_parameters(parameters, fileName)
                plot_avg_sol(parameters, fileName)
            plot_parameters(parameters, fileName)
            plot_sol(parameters, fileName)
        print(f"\n{fileName}")
        print(f"Best solution found: {cur_dis}")
        for i in range(len(bounds["min"])):
            print(f"Best {chr(65 + i)} found: {min_pos[i]}")
        print(f"Elapsed time: {run_time}\n")
        print("*" * 36)

        # record the best solution found
        infoName = parameters["info_path"]
        if not os.path.exists(infoName):
            with open(infoName, "a") as f:
                f.write("TSP, Best_Epoch, " + ', '.join(
                    ["Best_" + chr(65 + i) for i in range(len(bounds['min']))]
                ) + ", Best_Solution, Best_Mixed, Anneal_Time, Elapsed_Time\n")
        with open(infoName, "a") as f:
            f.write(
                f"{Path(fileName).stem},{best_epoch + 1}," +
                ','.join([f"{min_pos[i]}"
                          for i in range(len(bounds['min']))]) +
                f",{cur_dis},{cur_mixed},{elapsed_time},{run_time}\n")

    #         pathlib.Path(f"./data/completed/").mkdir(parents=True, exist_ok=True)
    #         shutil.move(filePath, "./data/completed/{}".format(Path(filePath).stem))
    #         completed = "./data/completed/{}".format(Path(filePath).stem)

    pathlib.Path(parameters["folder_path"]).mkdir(parents=True, exist_ok=True)

    # pso
    if os.path.isdir(dirName):
        for fileName in os.listdir(dirName):
            filePath = os.path.join(dirName, fileName)
            _sub(filePath)
    else:
        filePath = dirName
        _sub(filePath)


def save_solution(fileName, e, A, min_perm, min_cost, mixed_cost, elapsed_time,
                  parameters):
    pathlib.Path(parameters["folder_path"]).mkdir(parents=True, exist_ok=True)
    ftspName = Path(fileName).stem
    outputName = f"{parameters['folder_path']}{ftspName}.csv"
    if not os.path.exists(outputName):
        with open(outputName, "a") as f:
            f.write("epoch," + ", ".join([chr(65 + i)
                                          for i in range(len(A))]) +
                    ", da_cost, mixed_cost, elapsed_time\n")
    with open(outputName, "a") as f:
        f.write(f"{e + 1}," + ','.join([f"{A[i]}" for i in range(len(A))]) +
                f",{min_cost},{mixed_cost},{elapsed_time}\n")


def save_epoch(fileName, e, cur_cost, epoch_min, cur_mixed, min_pos,
               elapsed_time, parameters):
    pathlib.Path(parameters["folder_path"]).mkdir(parents=True, exist_ok=True)
    ftspName = Path(fileName).stem
    outputName = f"{parameters['folder_path']}{ftspName}-Epoch.csv"
    if not os.path.exists(outputName):
        with open(outputName, "a") as f:
            f.write("Epoch, " + ', '.join(
                [f"Current_Best_{chr(65 + i)}" for i in range(len(min_pos))]
            ) + ", Current_Best_Cost, Current_Best_Mixed, Current_Epoch_cost, Elapsed_Time\n"
                    )
    with open(outputName, "a") as f:
        f.write(f"{e + 1}," +
                ','.join([f"{min_pos[i]}" for i in range(len(min_pos))]) +
                f",{cur_cost},{cur_mixed},{epoch_min},{elapsed_time}\n")


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
        points = np.array([
            list(df["A"].iloc[df.index[df["epoch"] == i]]),
            list(df["B"].iloc[df.index[df["epoch"] == i]]),
        ])
        numpoints = len(df.index[df["epoch"] == i].tolist())
        colors = cm.rainbow(np.linspace(0, 1, numpoints))
        t = plt.scatter(*points, c=colors, s=100)
        plt.legend((t, ), [f"iter {i}"])
        plt.title(ftspName)
        camera.snap()
    anim = camera.animate(blit=True)
    writer = PillowWriter(fps=1)
    anim.save(filename=f"{parameters['folder_path']}{ftspName}.gif",
              writer=writer)
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
        comb += f"-{'U' if parameters['U'] else ''}{parameters['topo']}-{'Random' if parameters['ini'] == 0 else 'Even'}-E{parameters['epoch']}-P{parameters['particles']}-W{parameters['w']}-C1{parameters['c1']}-C2{parameters['c2']}"
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
        clusters = seq.cluster(parameters['clustering'],
                               a=parameters['offset'],
                               k=parameters['n_clusters'])

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
                    seq_c._distance, seq_c._jobs),
            )

            cur_perm, cur_dis, cur_mixed, min_pos, best_epoch, elapsed_time = tuner(
                fileName, seq_c, rules, bounds, parameters, solve)
            seq_c_order = restore(cur_perm, c[-1])
            seq_c_order.remove(0)
            order.append(seq_c_order)
        run_time = time.time() - start

        min_makespan = float('inf')
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
            f.write("{},{},{}\n".format(
                Path(fileName).stem,
                min_makespan,
                run_time,
            ))

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
        clusters = seq.cluster(parameters['clustering'],
                               a=parameters['offset'],
                               k=parameters['n_clusters'])

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
                    seq_c._distance, seq_c._jobs),
            )

            thr = Thread(
                target=lambda q, arg: q.put((i, *tuner(*arg))),
                args=(
                    que,
                    (fileName, seq_c, rules, bounds, parameters, solve),
                ),
            )
            thrs.append(thr)
            thr.start()
        for thr in thrs:
            thr.join()

        order = []
        while not que.empty():
            i, cur_perm, cur_mksp, cur_mixed, min_pos, best_epoch, elapsed_time = que.get(
            )
            seq_c, dic = Sequence(clusters[i][0]), clusters[i][-1]
            seq_c_order = restore(cur_perm, c[-1])
            seq_c_order.remove(0)
            order.append(seq_c_order)
        run_time = time.time() - start

        min_makespan = float('inf')
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
            f.write("{},{},{}\n".format(
                Path(fileName).stem,
                min_makespan,
                run_time,
            ))

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
                    seq_c._distance, seq_c._jobs),
            )

            cur_perm, cur_mksp, cur_mixed, min_pos, best_epoch, elapsed_time = tuner(
                fileName, seq_c, rules, bounds, parameters, solve)
            seq_c_order = restore(seq_c, c[-1])
            seq_c_order.remove(0)
            order.append(seq_c_order)
            total_proc.append(np.sum(seq_c._proc) / seq_c._jobs)
        run_time = time.time() - start

        # sort clusters via total processing time in descending order
        sorted_indices = sorted(range(len(total_proc)),
                                key=total_proc.__getitem__,
                                reverse=True)
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
            f.write("{},{},{}\n".format(
                Path(fileName).stem,
                min_makespan,
                run_time,
            ))

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
                if j - i == 1: continue
                if TSP.cost_change(graph, best[i - 1], best[i], best[j - 1],
                                   best[j]) < 0:
                    best[i:j] = best[j - 1:i - 1:-1]
                    improved = True
        route = best
    dist = 0
    for i in range(len(route)):
        dist += graph[route[i]][route[(i + 1) % len(route)]]
    return dist, best


def stitch_tsp_cluster(n, g, num_run=128, num_iteration=10**8):

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

    # args and params
    param = {
        "offset_mode": 3,
        "offset_inc_rate": 1000,
        "tmp_st": 1,
        "tmp_decay": 0,
        "tmp_mode": 0,
        "tmp_interval": 100,
    }

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

    server_list = [
        "dau2-05-0.da.labs.fujitsu.com",
        "dau2-05-1.da.labs.fujitsu.com",
        "dau2-06-0.da.labs.fujitsu.com",
        "dau2-06-1.da.labs.fujitsu.com",
    ]
    da = fjda_client.fjda(server=random.choice(server_list))

    # send to DA
    da.setAnnealParameterMM(param)

    received = False
    result = da.doAnnealMM(args, format="numpy", compression="lz4")
    while not received:
        if result:
            print('\nStitched')
            received = True

    # derive solution
    elapsed_time = result["anneal_time"] / 1000
    return {idx: bool(res)
                        for idx, res in enumerate(result['state_min_o_n.numpy'][result['eg_min_o_n.numpy'].argmin()])
                        if idx < n**2}, \
                       result['eg_min_o_n.numpy'][result['eg_min_o_n.numpy'].argmin()]


def main_tsp_cluster(dirName, tuner, bounds, parameters, solve):
    def _sub(filePath):
        nonlocal bounds
        nonlocal parameters
        fileName = Path(filePath).stem
        tsp = TSP(tsplib95.load(filePath))
        clusters, centers = tsp.k_part()
        stitcher = Stitcher(clusters, tsp._g)
        min_cost = float('inf')

        start = time.time()
        for t in stitcher._clusters.values():

            rules = dict(
                column=build_column_rule(len(t._nodes)),
                row=build_row_rule(len(t._nodes)),
                edge=build_edge_rule(perturb(t._g), len(t._nodes)),
                edge_unperturbed=build_edge_rule_unperturbed(
                    t._g, len(t._nodes)),
            )

            cur_perm, cur_dis, cur_mixed, min_pos, best_epoch, elapsed_time = tuner(
                fileName, t, rules, bounds, parameters, solve)
            t._path = cur_perm
            t._cost = cur_dis

        # stitch
        join, flip_cost, flip_set = stitcher.generate_join_scheme()
        if len(join) > 0:
            stitcher.stitch(join, flip_cost, flip_set)
            run_time = time.time() - start

            min_cost = min(
                sum([
                    tsp._g[stitcher._path[i],
                           stitcher._path[(i + 1) % tsp._nodes.__len__()]]
                    for i in range(tsp._nodes.__len__())
                ]), min_cost)

            # stitcher._cost, stitcher._path = tsp.get_two_opt(stitcher._path)
            stitcher._cost, stitcher._path = tsp.get_two_opt(
                tsp._g, stitcher._path)
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
            f.write("{},{},{},{}\n".format(
                Path(fileName).stem, min_cost,
                stitcher._cost if min_cost != -1 else -1, run_time))

    pathlib.Path(parameters["folder_path"]).mkdir(parents=True, exist_ok=True)

    if os.path.isdir(dirName):
        for fileName in os.listdir(dirName):
            filePath = os.path.join(dirName, fileName)
            _sub(filePath)
    else:
        filePath = dirName
        _sub(filePath)


### QAP


def solve_qap(qap, rules, A, num_run, num_iteration):

    # prepare the QUBO
    qubo = build_QAP(qap, A, rules)
    N_state, bias, constant, weight, s, local_field, E = prepare_config(qubo)

    # args and params
    param = {
        "offset_mode": 3,
        "offset_inc_rate": 1000,
        "tmp_st": 1,
        "tmp_decay": 0,
        "tmp_mode": 0,
        "tmp_interval": 100,
    }

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

    server_list = [
        "dau2-05-0.da.labs.fujitsu.com",
        "dau2-05-1.da.labs.fujitsu.com",
        "dau2-06-0.da.labs.fujitsu.com",
        "dau2-06-1.da.labs.fujitsu.com",
    ]
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
    min_sol, min_cost = {}, float("inf")
    for idx, sol in enumerate(res['state_min_o_n.numpy']):
        conf, energy = {id: bool(re)
                        for id, re in enumerate(sol)
                        }, res['eg_min_o_n.numpy'][idx]
        da_sol, da_cost = compute_cost(qap, conf)
        if da_cost > 0 and da_cost < min_cost:
            min_sol, min_cost = da_sol, da_cost

    # elapsed_time
    elapsed_time = time.time() - start + res["anneal_time"] / 1000
    if len(min_sol) == 0:
        return A, {}, float("inf"), float("inf"), elapsed_time
    return A, min_sol, min_cost, min_cost, elapsed_time


def solve_qap_avg(qap, rules, A, num_run, num_iteration):

    # prepare the QUBO
    qubo = build_QAP(qap, A, rules)
    N_state, bias, constant, weight, s, local_field, E = prepare_config(qubo)

    # args and params
    param = {
        "offset_mode": 3,
        "offset_inc_rate": 1000,
        "tmp_st": 1,
        "tmp_decay": 0,
        "tmp_mode": 0,
        "tmp_interval": 100,
    }

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

    server_list = [
        "dau2-05-0.da.labs.fujitsu.com",
        "dau2-05-1.da.labs.fujitsu.com",
        "dau2-06-0.da.labs.fujitsu.com",
        "dau2-06-1.da.labs.fujitsu.com",
    ]
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
    min_sol, min_cost, cost_list = {}, float("inf"), []
    for idx, sol in enumerate(res['state_min_o_n.numpy']):
        conf, energy = {id: bool(re)
                        for id, re in enumerate(sol)
                        }, res['eg_min_o_n.numpy'][idx]
        da_sol, da_cost = compute_cost(qap, conf)
        if da_cost > 0:
            cost_list.append(da_cost)
            if da_cost < min_cost:
                min_cost = da_cost
                min_sol = da_sol

    # elapsed_time
    elapsed_time = time.time() - start + res["anneal_time"] / 1000
    return (
        A,
        min_sol,
        min_cost,
        np.mean(cost_list) if len(cost_list) > 0 else float("inf"),
        elapsed_time,
    )


def main_qap(dirName, tuner, bounds, parameters, solve):
    def _parse(fileName):
        n, d, f = 0, [], []
        with open(fileName, 'rb') as file:
            for i, line in enumerate(file.readlines()):
                if i == 0:
                    n = int(line)
                elif 2 <= i <= n + 1:
                    d.append([int(j) for j in line.split()])
                else:
                    tmp = [int(j) for j in line.split()]
                    if len(tmp) > 0:
                        f.append(tmp)
        return n, d, f

    def _sub(filePath):
        nonlocal bounds
        nonlocal parameters
        n, d, f = _parse(filePath)
        fileName = Path(filePath).stem
        qap = QAP(d, f, parameters['perturbed'])
        rules = dict(
            row=build_row_sum(qap),
            column=build_col_sum(qap),
            obj=build_obj(qap),
        )
        start = time.time()
        cur_sol, cur_cost, cur_mixed, min_pos, best_epoch, elapsed_time = tuner(
            fileName, qap, rules, bounds, parameters, solve)
        run_time = time.time() - start
        if len(bounds['min']) > 1:
            if "pso" in parameters["folder_path"]:
                plot_trajectory(parameters, fileName)
                plot_avg_parameters(parameters, fileName)
                plot_avg_sol(parameters, fileName)
            plot_parameters(parameters, fileName)
            plot_sol(parameters, fileName)
        print(f"\n{Path(fileName).stem}")
        print(f"Best solution found: {cur_cost}")
        for i in range(len(bounds["min"])):
            print(f"Best {chr(65 + i)} found: {min_pos[i]}")
        print(f"Elapsed time: {run_time}\n")
        print("*" * 36)

        # record the best solution found
        infoName = parameters["info_path"]
        if not os.path.exists(infoName):
            with open(infoName, "a") as f:
                f.write("QAP, Best_Epoch, " + ', '.join(
                    ["Best_" + chr(65 + i) for i in range(len(bounds['min']))]
                ) + ", Best_Solution, Best_Mixed, Anneal_Time, Elapsed_Time\n")
        with open(infoName, "a") as f:
            f.write(
                f"{Path(fileName).stem},{best_epoch + 1}," +
                ','.join([f"{min_pos[i]}"
                          for i in range(len(bounds['min']))]) +
                f",{cur_cost},{cur_mixed},{elapsed_time},{run_time}\n")

    pathlib.Path(parameters["folder_path"]).mkdir(parents=True, exist_ok=True)

    if os.path.isdir(dirName):
        for fileName in os.listdir(dirName):
            filePath = os.path.join(dirName, fileName)
            _sub(filePath)
    else:
        filePath = dirName
        _sub(filePath)