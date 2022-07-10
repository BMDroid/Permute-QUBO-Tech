'''
Description: 
Author: BO Jianyuan
Date: 2021-02-05 18:16:01
LastEditors: BO Jianyuan
LastEditTime: 2021-02-05 18:20:23
'''
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
from solver_new import *
from tuner_new import *
from sequence import *
from tsp import *

bounds = {'0A': (0.5, 1), '1B': (0.5, 1), '2iteration_num': (10**7, 10**8)}

# PSO with regularization
parameters = {
    'w0': 0.5,
    'c10': 0.5,
    'c20': 0.9,
    'w': 0.25,
    'c1': 0.25,
    'c2': 0.6,
    'ini': 0
}
parameters['U'], parameters['topo'] = True, 'Square'
parameters['fixed'], parameters['mixed'], parameters['num_run'] = False, 1, 72
parameters['particles'], parameters['epoch'] = 10, 5
parameters['base'] = '_max_edge'
parameters['reg'] = 0.5 # regularization term enabled
parameters['comb'], parameters['folder_path'], parameters[
    'setting_path'], parameters['info_path'] = get_results_path_new(
        'tsp_pso_new_reg', parameters)
main_tsp('./data/TSP/tsplib_hard', pso_new, bounds, parameters, solve_tsp_avg)

# PSO without regularization term
parameters = {
    'w0': 0.5,
    'c10': 0.5,
    'c20': 0.9,
    'w': 0.25,
    'c1': 0.25,
    'c2': 0.6,
    'ini': 0
}
parameters['U'], parameters['topo'] = True, 'Square'
parameters['fixed'], parameters['mixed'], parameters['num_run'] = False, 1, 72
parameters['particles'], parameters['epoch'] = 10, 5
parameters['base'] = '_max_edge'
parameters['reg'] = 0 # regularization term disabled
parameters['comb'], parameters['folder_path'], parameters[
    'setting_path'], parameters['info_path'] = get_results_path_new(
        'tsp_pso_new', parameters)
main_tsp('./data/TSP/tsplib_hard', pso_new, bounds, parameters, solve_tsp_avg)

### fixed parameter B
# PSO with regularization
parameters = {
    'w0': 0.5,
    'c10': 0.5,
    'c20': 0.9,
    'w': 0.25,
    'c1': 0.25,
    'c2': 0.6,
    'ini': 0
}
parameters['U'], parameters['topo'] = True, 'Square'
parameters['fixed'], parameters['mixed'], parameters['num_run'] = True, 1, 72
parameters['particles'], parameters['epoch'] = 10, 5
parameters['base'] = '_max_edge'
parameters['reg'] = 0.5 # regularization term enabled
parameters['comb'], parameters['folder_path'], parameters[
    'setting_path'], parameters['info_path'] = get_results_path_new(
        'tsp_pso_new_reg', parameters)
main_tsp('./data/TSP/tsplib_hard', pso_new, bounds, parameters, solve_tsp_avg)

# PSO without regularization term
parameters = {
    'w0': 0.5,
    'c10': 0.5,
    'c20': 0.9,
    'w': 0.25,
    'c1': 0.25,
    'c2': 0.6,
    'ini': 0
}
parameters['U'], parameters['topo'] = True, 'Square'
parameters['fixed'], parameters['mixed'], parameters['num_run'] = True, 1, 72
parameters['particles'], parameters['epoch'] = 10, 5
parameters['base'] = '_max_edge'
parameters['reg'] = 0 # regularization term disabled
parameters['comb'], parameters['folder_path'], parameters[
    'setting_path'], parameters['info_path'] = get_results_path_new(
        'tsp_pso_new', parameters)
main_tsp('./data/TSP/tsplib_hard', pso_new, bounds, parameters, solve_tsp_avg)

### Random
# # normal
# parameters = {}
# parameters['fixed'], parameters['mixed'], parameters['num_run'], parameters['num_iteration'] = False, 1, 72, [10**8]
# parameters['max_evals'] = 40
# parameters['base'] = '_max_edge'
# parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path('tsp_normal_comp', parameters)
# main_tsp('./data/TSP/tsplib_hard', randomer_thr, bounds, parameters, solve_tsp_avg)

# parameters = {}
# parameters['fixed'], parameters['mixed'], parameters['num_run'], parameters['num_iteration'] = True, 1, 72, [10**8]
# parameters['max_evals'] = 40
# parameters['base'] = '_max_edge'
# parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path('tsp_normal_comp', parameters)
# main_tsp('./data/TSP/tsplib_hard', randomer_thr, bounds, parameters, solve_tsp_avg)

# ### Random
# # uniform
# parameters = {}
# parameters['fixed'], parameters['mixed'], parameters['num_run'], parameters['num_iteration'] = False, 1, 72, [10**8]
# parameters['max_evals'] = 40
# parameters['base'] = '_max_edge'
# parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path('tsp_uniform_comp', parameters)
# main_tsp('./data/TSP/tsplib_hard', randomer_thr, bounds, parameters, solve_tsp_avg)

# parameters = {}
# parameters['fixed'], parameters['mixed'], parameters['num_run'], parameters['num_iteration'] = True, 1, 72, [10**8]
# parameters['max_evals'] = 40
# parameters['base'] = '_max_edge'
# parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path('tsp_uniform_comp', parameters)
# main_tsp('./data/TSP/tsplib_hard', randomer_thr, bounds, parameters, solve_tsp_avg)