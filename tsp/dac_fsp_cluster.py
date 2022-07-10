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

bounds = {'min': [0.5, 0.5], 'max': [1, 1]}

### spectral
parameters = {'w0': 0.5, 'c10': 0.5, 'c20': 0.9, 'w': 0.25, 'c1': 0.25, 'c2': 0.6, 'ini': 0}
parameters['U'], parameters['topo'] = True, 'Square'
parameters['fixed'], parameters['mixed'], parameters['num_run'], parameters['num_iteration'] = False, 1, 64, [10**7] * 6
parameters['particles'], parameters['epoch'] = 10, len(parameters['num_iteration']) - 1
parameters['dis'],  parameters['clustering'], parameters['offset'], parameters['n_clusters'] = 'nocarry', 'spectral', 0, 30
parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path('fsp_spectral_cluster_max_pso', parameters)
main_fsp_cluster_iter('./data/FSP/n_5', pso, bounds, parameters, solve_fsp_avg)

# large
parameters['dis'],  parameters['clustering'], parameters['offset'], parameters['n_clusters'] = 'fhsoph', 'spectral', 0, 50
parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path('fsp_spectral_cluster_max_pso', parameters)
main_fsp_cluster_iter('./data/FSP/100_10', pso, bounds, parameters, solve_fsp_avg)


### agglomerative
parameters = {'w0': 0.5, 'c10': 0.5, 'c20': 0.9, 'w': 0.25, 'c1': 0.25, 'c2': 0.6, 'ini': 0}
parameters['U'], parameters['topo'] = True, 'Square'
parameters['fixed'], parameters['mixed'], parameters['num_run'], parameters['num_iteration'] = False, 1, 64, [10**7] * 6
parameters['particles'], parameters['epoch'] = 10, len(parameters['num_iteration']) - 1
parameters['dis'],  parameters['clustering'], parameters['offset'], parameters['n_clusters'] = 'nocarry', 'agglomerative', 0, 30
parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path('fsp_agglomerative_cluster_max_pso', parameters)
main_fsp_cluster_iter('./data/FSP/n_5', pso, bounds, parameters, solve_fsp_avg)

# large
parameters['dis'],  parameters['clustering'], parameters['offset'], parameters['n_clusters'] = 'fhsoph', 'agglomerative', 0, 50
parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path('fsp_agglomerative_cluster_max_pso', parameters)
main_fsp_cluster_iter('./data/FSP/100_10', pso, bounds, parameters, solve_fsp_avg)





