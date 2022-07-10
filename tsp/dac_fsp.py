'''
Description: DAC FSP
Author: BO Jianyuan
Date: 2020-12-07 14:36:40
LastEditors: BO Jianyuan
LastEditTime: 2020-12-07 14:38:10
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
from solver import *
from tuner import *
from sequence import *
from tsp import *

# two parameters
bounds = {'min': [0.5, 0.5], 'max': [1, 1]}

# single parameter
bounds = {'min': [0.5], 'max': [1]}

### number of trails = 40
# # PSO
# parameters = {'w0': 0.5, 'c10': 0.5, 'c20': 0.9, 'w': 0.25, 'c1': 0.25, 'c2': 0.6, 'ini': 0}
# parameters['U'], parameters['topo'] = True, 'Square'
# parameters['fixed'], parameters['mixed'], parameters['num_run'], parameters['num_iteration'] = False, 1, 64, [10**7] * 4
# parameters['dis'] = 'nocarry'
# parameters['base'] = '_max_edge'
# parameters['particles'], parameters['epoch'] = 10, len(parameters['num_iteration']) - 1
# parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path('fsp_pso_single', parameters)
# main_fsp('./data/FSP/x_5', pso, bounds, parameters, solve_fsp_avg)
# main_fsp('./data/FSP/x_10', pso, bounds, parameters, solve_fsp_avg)

# # Hyperopt
# parameters = {}
# parameters['fixed'], parameters['mixed'], parameters['num_run'], parameters['num_iteration'] = False, 1, 64, [10**7]
# parameters['max_evals'] = 40
# parameters['dis'] = 'nocarry'
# parameters['base'] = '_max_edge'
# parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path('fsp_hyperopt_single', parameters)
# main_fsp('./data/FSP/x_5', hyperopt, bounds, parameters, solve_fsp_avg)
# main_fsp('./data/FSP/x_10', hyperopt, bounds, parameters, solve_fsp_avg)

# # Optuna
# parameters = {}
# parameters['fixed'], parameters['mixed'], parameters['num_run'], parameters['num_iteration'] = False, 1, 64, [10**7]
# parameters['max_evals'] = 40
# parameters['dis'] = 'nocarry'
# parameters['base'] = '_max_edge'
# parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path('fsp_optuna_single', parameters)
# main_fsp('./data/FSP/x_5', optune, bounds, parameters, solve_fsp_avg)
# main_fsp('./data/FSP/x_10', optune, bounds, parameters, solve_fsp_avg)

# # Random
# parameters = {}
# parameters['fixed'], parameters['mixed'], parameters['num_run'], parameters['num_iteration'] = False, 1, 64, [10**7]
# parameters['max_evals'] = 40
# parameters['dis'] = 'nocarry'
# parameters['dist'] = 'normal' 
# parameters['base'] = '_max_edge'
# parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path('fsp_randomer_normal_single', parameters)
# main_fsp('./data/FSP/x_5', randomer_thr, bounds, parameters, solve_fsp_avg)
# main_fsp('./data/FSP/x_10', randomer_thr, bounds, parameters, solve_fsp_avg)

# parameters = {}
# parameters['fixed'], parameters['mixed'], parameters['num_run'], parameters['num_iteration'] = False, 1, 64, [10**7]
# parameters['max_evals'] = 40
# parameters['dis'] = 'nocarry'
# parameters['base'] = '_max_edge'
# parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path('fsp_randomer_uniform_single', parameters)
# main_fsp('./data/FSP/x_5', randomer_thr, bounds, parameters, solve_fsp_avg)
# main_fsp('./data/FSP/x_10', randomer_thr, bounds, parameters, solve_fsp_avg)

# ### number of trails = 20
# # PSO
# parameters = {'w0': 0.5, 'c10': 0.5, 'c20': 0.9, 'w': 0.25, 'c1': 0.25, 'c2': 0.6, 'ini': 0}
# parameters['U'], parameters['topo'] = True, 'Square'
# parameters['fixed'], parameters['mixed'], parameters['num_run'], parameters['num_iteration'] = False, 1, 64, [10**7] * 4
# parameters['dis'] = 'nocarry'
# parameters['base'] = '_max_edge'
# parameters['particles'], parameters['epoch'] = 5, len(parameters['num_iteration']) - 1
# parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path('fsp_pso', parameters)
# main_fsp('./data/FSP/x_5', pso, bounds, parameters, solve_fsp_avg)
# main_fsp('./data/FSP/x_10', pso, bounds, parameters, solve_fsp_avg)

# # Hyperopt
# parameters = {}
# parameters['fixed'], parameters['mixed'], parameters['num_run'], parameters['num_iteration'] = False, 1, 64, [10**7]
# parameters['max_evals'] = 20
# parameters['dis'] = 'nocarry'
# parameters['base'] = '_max_edge'
# parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path('fsp_hyperopt', parameters)
# main_fsp('./data/FSP/x_5', hyperopt, bounds, parameters, solve_fsp_avg)
# main_fsp('./data/FSP/x_10', hyperopt, bounds, parameters, solve_fsp_avg)

# # Optuna
# parameters = {}
# parameters['fixed'], parameters['mixed'], parameters['num_run'], parameters['num_iteration'] = False, 1, 64, [10**7]
# parameters['max_evals'] = 20
# parameters['dis'] = 'nocarry'
# parameters['base'] = '_max_edge'
# parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path('fsp_optuna', parameters)
# main_fsp('./data/FSP/x_5', optune, bounds, parameters, solve_fsp_avg)
# main_fsp('./data/FSP/x_10', optune, bounds, parameters, solve_fsp_avg)

### Random
# normal
# parameters = {}
# parameters['fixed'], parameters['mixed'], parameters['num_run'], parameters['num_iteration'] = False, 1, 64, [10**7]
# parameters['max_evals'] = 20
# parameters['dis'] = 'nocarry'
# parameters['dist'] = 'normal' 
# parameters['base'] = '_max_edge'
# parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path('fsp_randomer_normal', parameters)
# main_fsp('./data/FSP/x_5', randomer_thr, bounds, parameters, solve_fsp_avg)
# main_fsp('./data/FSP/x_10', randomer_thr, bounds, parameters, solve_fsp_avg)

# # uniform
# parameters = {}
# parameters['fixed'], parameters['mixed'], parameters['num_run'], parameters['num_iteration'] = False, 1, 64, [10**7]
# parameters['max_evals'] = 20
# parameters['dis'] = 'nocarry'
# parameters['base'] = '_max_edge'
# parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path('fsp_randomer_uniform', parameters)
# main_fsp('./data/FSP/x_5', randomer_thr, bounds, parameters, solve_fsp_avg)
# main_fsp('./data/FSP/x_10', randomer_thr, bounds, parameters, solve_fsp_avg)

### MLP ratio prediction
parameters = {}
parameters['fixed'], parameters['mixed'], parameters['num_run'], parameters['num_iteration'] = False, 1, 128, [10**8]
parameters['max_evals'] = 1
parameters['dis'] = 'nocarry'
parameters['base'] = '_max_edge'
parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path('fsp_mlp', parameters)
main_fsp('./data/FSP/x_5', ratio_predictor, bounds, parameters, solve_fsp_avg)
main_fsp('./data/FSP/x_10', ratio_predictor, bounds, parameters, solve_fsp_avg)