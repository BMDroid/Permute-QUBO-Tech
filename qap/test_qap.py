'''
Description: 
Author: BO Jianyuan
Date: 2021-06-06 06:07:34
LastEditors: BO Jianyuan
LastEditTime: 2021-09-08 08:06:39
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
from builder_qap import *
from solver_qap import *
from tuner_qap import *
from sequence import *
from tsp import *
from asgmt import *
from max_cut import *
from qap import *

# Optuna
# bounds = {'min': [1e-8, 1e-8], 'max': [2e-8, 2e-8]}
# bounds = {'min': [1e-2, 1e-2], 'max': [2e-2, 2e-2]}
# parameters = {}
# parameters["perturbed"] = False
# parameters['base'] = 'max_product'
# parameters['fixed'], parameters['mixed'] = False, 1
# parameters['max_evals'] = 40
# parameters['num_run'], parameters['num_iteration'] = 72, [10**7] * (parameters['max_evals'] + 1)
# parameters['reg'] = 0
# parameters['reg_coef'] = [1, 1]
# parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path(f"qap_optuna_min{bounds['min'][0]}_max{bounds['max'][0]}_{len(bounds['min'])}b_zero", parameters)
# main_qap('./data/Test/*.dat', optune, bounds, parameters, solve_min_qap_config, prep_config_zero)

# # Hyperopt
# bounds = {'min': [1e-2, 1e-2], 'max': [2e-2, 2e-2]}
# parameters = {}
# parameters["perturbed"] = False
# parameters['base'] = 'max_product'
# parameters['fixed'], parameters['mixed'] = False, 1
# parameters['max_evals'] = 20
# parameters['num_run'], parameters['num_iteration'] = 72, [10**7] * (parameters['max_evals'] + 1)
# parameters['reg'] = 0
# parameters['reg_coef'] = [1, 1]
# parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path(f"qap_hyperopt_min{bounds['min'][0]}_max{bounds['max'][0]}_{len(bounds['min'])}b_zero", parameters)
# main_qap('./data/Test/*.dat', hyperopt, bounds, parameters, solve_min_qap_config, prep_config_zero)

# # PSO
# bounds = {'min': [1e-2, 1e-2], 'max': [2e-2, 2e-2]}
# parameters = {
#     'w0': 0.6,
#     'c10': 0.6,
#     'c20': 0.82,
#     'w': 0.42,
#     'c1': 0.2,
#     'c2': 0.2,
#     'ini': 0
# }

# parameters["perturbed"] = False
# parameters['base'] = 'max_product'
# parameters['U'], parameters['topo'] = True, 'Ring'
# parameters['fixed'], parameters['mixed'] = False, 1
# parameters['population_size'], parameters['max_evals'] = 10, 6
# parameters['num_run'], parameters['num_iteration'] = 72, [10**7] * (parameters['max_evals'] + 1)
# parameters['reg'] = 0
# parameters['reg_coef'] = [1, 1]
# parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path(f"qap_pso_ini_ring_min{bounds['min'][0]}_max{bounds['max'][0]}_{len(bounds['min'])}b_zero", parameters)
# main_qap('./data/Test/*.dat', pso_ini, bounds, parameters, solve_min_qap_config, prep_config_zero)

### qap subclustering
# Optuna
bounds = {'min': [1e-2, 1e-2], 'max': [2e-2, 2e-2]}
parameters = {}
parameters["perturbed"] = True
parameters['base'] = 'max_product'
parameters['sta'], parameters['n_clusters'] = 'sum', 5

parameters['fixed'], parameters['mixed'] = False, 1
parameters['max_evals'] = 20
parameters['num_run'], parameters['num_iteration'] = 72, [10**7] * (parameters['max_evals'] + 1)
parameters['reg'] = 0
parameters['reg_coef'] = [1, 1]
parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path(f"agg_60_perturbed_cluster_qap_optuna_min{bounds['min'][0]}_max{bounds['max'][0]}_{len(bounds['min'])}b_zero", parameters)
main_qap_clustering('./data/Large/*.dat', optune, bounds, parameters, solve_min_qap_config, prep_config_zero)

bounds = {'min': [1e-2, 1e-2], 'max': [2e-2, 2e-2]}
parameters = {}
parameters["perturbed"] = True
parameters['base'] = 'max_product'
parameters['sta'], parameters['n_clusters'] = 'sum', 5

parameters['fixed'], parameters['mixed'] = False, 1
parameters['max_evals'] = 20
parameters['num_run'], parameters['num_iteration'] = 72, [10**7] * (parameters['max_evals'] + 1)
parameters['reg'] = 0
parameters['reg_coef'] = [1, 1]
parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path(f"agg_60_mod_perturbed_cluster_qap_optuna_min{bounds['min'][0]}_max{bounds['max'][0]}_{len(bounds['min'])}b_zero", parameters)
main_qap_clustering_mod('./data/Large/*.dat', optune, bounds, parameters, solve_min_qap_config, prep_config_zero)

# # Hyperopt
# bounds = {'min': [1e-2, 1e-2], 'max': [2e-2, 2e-2]}
# parameters = {}
# parameters["perturbed"] = True
# parameters['base'] = 'max_product'
# parameters['sta'], parameters['n_clusters'] = 'sum', 3

# parameters['fixed'], parameters['mixed'] = False, 1
# parameters['max_evals'] = 20
# parameters['num_run'], parameters['num_iteration'] = 72, [10**7] * (parameters['max_evals'] + 1)
# parameters['reg'] = 0
# parameters['reg_coef'] = [1, 1]
# parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path(f"40_perturbed_cluster_qap_hyperopt_min{bounds['min'][0]}_max{bounds['max'][0]}_{len(bounds['min'])}b_zero", parameters)
# main_qap_clustering('./data/Large/*.dat', hyperopt, bounds, parameters, solve_min_qap_config, prep_config_zero)

# # PSO
# bounds = {'min': [1e-2, 1e-2], 'max': [2e-2, 2e-2]}
# parameters = {
#     'w0': 0.6,
#     'c10': 0.6,
#     'c20': 0.82,
#     'w': 0.42,
#     'c1': 0.2,
#     'c2': 0.2,
#     'ini': 0
# }

# parameters["perturbed"] = True
# parameters['base'] = 'max_product'
# parameters['sta'], parameters['n_clusters'] = 'sum', 3

# parameters['U'], parameters['topo'] = True, 'Ring'
# parameters['fixed'], parameters['mixed'] = False, 1
# parameters['population_size'], parameters['max_evals'] = 10, 6
# parameters['num_run'], parameters['num_iteration'] = 72, [10**7] * (parameters['max_evals'] + 1)
# parameters['reg'] = 0
# parameters['reg_coef'] = [1, 1]
# parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path(f"40_perturbed_cluster_qap_pso_ini_ring_min{bounds['min'][0]}_max{bounds['max'][0]}_{len(bounds['min'])}b_zero", parameters)
# main_qap_clustering('./data/Large/*.dat', pso_ini, bounds, parameters, solve_min_qap_config, prep_config_zero)