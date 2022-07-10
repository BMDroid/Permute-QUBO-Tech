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
from qap import *

bounds = {'min': [0.5], 'max': [1]}

# Base parameters
parameters = {'w0': 0.5, 'c10': 0.5, 'c20': 0.9, 'w': 0.25, 'c1': 0.25, 'c2': 0.6, 'ini': 0}
parameters['U'], parameters['topo'] = True, 'Square'
parameters['fixed'], parameters['mixed'], parameters['num_run'], parameters['num_iteration'] = False, 1, 64, [10**8] * 4
parameters['base'] = 'max_product'
parameters['perturbed'] = 1

# PSO
parameters['U'], parameters['topo'] = True, 'Square'
parameters['particles'], parameters['epoch'] = 10, len(parameters['num_iteration']) - 1
parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path('qap_pso', parameters)
main_qap('./data/QAP', pso, bounds, parameters, solve_qap_avg)

# Hyperopt
parameters['max_evals'] = 40
parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path('qap_hyperopt', parameters)
main_qap('./data/QAP', hyperopt, bounds, parameters, solve_qap_avg)

# Optuna
parameters['max_evals'] = 40
parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path('qap_optuna', parameters)
main_qap('./data/QAP', optune, bounds, parameters, solve_qap_avg)

# Random
parameters['max_evals'] = 40
parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path('qap_random_uniform', parameters)
main_qap('./data/QAP', randomer_thr, bounds, parameters, solve_qap_avg)