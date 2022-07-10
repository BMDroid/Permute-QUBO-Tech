'''
Description: DAU Sampler for Qbsolv
Author: BO Jianyuan
Date: 2021-08-31 14:18:12
LastEditors: BO Jianyuan
LastEditTime: 2021-09-06 21:27:13
'''

import random
import numpy as np
import time

from bisect import bisect
from qpoly import *
from python_fjda import fjda_client
from builder_qap import *

class DAU(object):
    def __init__(self, param, server_list, num_run, num_iteration, prep_config, seed, solver_limit):
        self.da = fjda_client.fjda(server=random.choice(server_list))
        self.da.setAnnealParameterMM(param)
        self.anneal_time = 0
        self.elapsed_time = 0
        if seed is not None:
            self.da.setRndSeed(seed)
        # args
        self.num_run = num_run
        self.num_iteration = num_iteration
        self.prep_config = prep_config
        self.seed = seed
        self.solver_limit = solver_limit
            
    def sample(self, Q, ini_state):
        '''
        Args: 
            Q::{(int, int) :int}
            ini_conf::{int, int}
        Returns: 
            response::{int:int}
        '''
        # transform dictionary to Qpoly array
        qubo = QPoly(self.solver_limit)
        for k, v in Q.items():
            qubo.array[k[0], k[1]] = v
        # transform ini_state to conf
        N_states = [1024, 2048, 4096, 8192]
        N_state = N_states[bisect(N_states, qubo._size)]
        conf = np.zeros(N_state)
        for k, v in ini_state.items():
            conf[k] = ini_state[k]
        N_state, bias, constant, weight, s, local_field, E = self.prep_config(qubo, 
                                                                              conf, 
                                                                              self.seed)
        args = {"weight": weight,
                "lf_i_n": np.tile(local_field, self.num_run),
                "state_min_i_n": np.tile(s, self.num_run),
                "state_cur_i_n": np.tile(s, self.num_run),
                "eg_min_i_n": np.full(self.num_run, E, dtype=int),
                "eg_cur_i_n": np.full(self.num_run, E, dtype=int),
                "num_iteration": self.num_iteration,
                "num_bit": N_state,
                "num_run": self.num_run,
                "ope_mode": 2
                }
        start = time.time()
        received = False
        res = self.da.doAnnealMM(args, format="numpy", compression="lz4")
        while not received:
            if res:
                print("Received")
                received = True
        min_energy = np.min(res['eg_min_o_n.numpy'])        
        min_idx = res['eg_min_o_n.numpy'].argmin()
        min_conf = res["state_min_o_n.numpy"][min_idx][:qubo._size]
        self.anneal_time += res["anneal_time"] / 1000
        self.elapsed_time += time.time() - start
        return {i: min_conf[i] for i in range(qubo._size)}
        