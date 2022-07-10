#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import sys
if sys.version_info[0] == 3:
    import lzma
    long = int
else:
    from backports import lzma
import numpy as np
    
from qpoly import *
from builder import *
from solver import *
from tuner import *
from sequence import *
from tsp import *
from stitcher import *

bounds = {'min': [0.5, 0.5], 'max': [1, 1]}

# fixed number of trails: 40 and 20
bounds = {'min': [0.5], 'max': [1]}

# PSO
parameters = {'w0': 0.5, 'c10': 0.5, 'c20': 0.9, 'w': 0.25, 'c1': 0.25, 'c2': 0.6, 'ini': 0}
parameters['U'], parameters['topo'] = True, 'Square'
parameters['fixed'], parameters['mixed'], parameters['num_run'], parameters['num_iteration'] = False, 1, 64, [10**7] * 4
parameters['particles'], parameters['epoch'] = 10, len(parameters['num_iteration']) - 1
parameters['base'] = '_max_edge'
parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path('tsp_pso_cluster_single', parameters)
# main_tsp_cluster('./data/TSP/ijcai', pso, bounds, parameters, solve_tsp_avg)
# main_tsp_cluster('./data/TSP/tsp', pso, bounds, parameters, solve_tsp_avg)
# main_tsp_cluster('./data/TSP/tsplib_complementary', pso, bounds, parameters, solve_tsp_avg)
main_tsp_cluster('./data/TSP/tnm_complementary', pso, bounds, parameters, solve_tsp_avg)

# Hyperopt
parameters = {}
parameters['fixed'], parameters['mixed'], parameters['num_run'], parameters['num_iteration'] = False, 1, 64, [10**7]
parameters['max_evals'] = 40
parameters['base'] = '_max_edge'
parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path('tsp_hyperopt_cluster_single', parameters)
# main_tsp_cluster('./data/TSP/ijcai', hyperopt, bounds, parameters, solve_tsp_avg)
# main_tsp_cluster('./data/TSP/tsp', hyperopt, bounds, parameters, solve_tsp_avg)
# main_tsp_cluster('./data/TSP/tsplib_complementary', hyperopt, bounds, parameters, solve_tsp_avg)
main_tsp_cluster('./data/TSP/tnm_complementary', hyperopt, bounds, parameters, solve_tsp_avg)

# Optuna
parameters = {}
parameters['fixed'], parameters['mixed'], parameters['num_run'], parameters['num_iteration'] = False, 1, 64, [10**7]
parameters['max_evals'] = 40
parameters['base'] = '_max_edge'
parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path('tsp_optuna_cluster_single', parameters)
# main_tsp_cluster('./data/TSP/ijcai', optune, bounds, parameters, solve_tsp_avg)
# main_tsp_cluster('./data/TSP/tsp', optune, bounds, parameters, solve_tsp_avg)
# main_tsp_cluster('./data/TSP/tsplib_complementary', optune, bounds, parameters, solve_tsp_avg)
main_tsp_cluster('./data/TSP/tnm_complementary', optune, bounds, parameters, solve_tsp_avg)

### Random
# normal
parameters = {}
parameters['fixed'], parameters['mixed'], parameters['num_run'], parameters['num_iteration'] = False, 1, 64, [10**7]
parameters['max_evals'] = 40
parameters['base'] = '_max_edge'
parameters['dist'] = 'normal' 
parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path('tsp_randomer_cluster_single_normal', parameters)
# main_tsp_cluster('./data/TSP/ijcai', randomer_thr, bounds, parameters, solve_tsp_avg)
# main_tsp_cluster('./data/TSP/tsp', randomer_thr, bounds, parameters, solve_tsp_avg)
# main_tsp_cluster('./data/TSP/tsplib_complementary', randomer_thr, bounds, parameters, solve_tsp_avg)
main_tsp_cluster('./data/TSP/tnm_complementary', randomer_thr, bounds, parameters, solve_tsp_avg)

### Random
# uniform
parameters = {}
parameters['fixed'], parameters['mixed'], parameters['num_run'], parameters['num_iteration'] = False, 1, 64, [10**7]
parameters['max_evals'] = 40
parameters['base'] = '_max_edge'
parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path('tsp_randomer_cluster_single_uniform', parameters)
# main_tsp_cluster('./data/TSP/ijcai', randomer_thr, bounds, parameters, solve_tsp_avg)
# main_tsp_cluster('./data/TSP/tsp', randomer_thr, bounds, parameters, solve_tsp_avg)
# main_tsp_cluster('./data/TSP/tsplib_complementary', randomer_thr, bounds, parameters, solve_tsp_avg)
main_tsp_cluster('./data/TSP/tnm_complementary', randomer_thr, bounds, parameters, solve_tsp_avg)

# Ratio Predictor
# parameters = {}
# parameters['fixed'], parameters['mixed'], parameters['num_run'], parameters['num_iteration'] = False, 1, 64, [10**8]
# parameters['max_evals'] = 1
# parameters['base'] = '_max_edge'
# parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path('tsp_mlp_cluster_single', parameters)
# main_tsp_cluster('./data/TSP/ijcai', ratio_predictor, bounds, parameters, solve_tsp_avg)
# main_tsp_cluster('./data/TSP/tsp', randomer_thr, bounds, parameters, solve_tsp_avg)