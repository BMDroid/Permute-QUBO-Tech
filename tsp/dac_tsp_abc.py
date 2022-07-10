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

parameters = {'w0': 0.5, 'c10': 0.5, 'c20': 0.9, 'w': 0.25, 'c1': 0.25, 'c2': 0.6, 'ini': 0}
parameters['U'], parameters['topo'] = True, 'Square'
parameters['fixed'], parameters['mixed'], parameters['num_run'] = False, 1, 72
parameters['base'] = '_max_edge'
parameters['reg'] = 0.5
parameters['coef'] = [1, 1.5, 0, 0]
bounds = {'0A': (0.5, 1), '1B': (0.5, 1), '2num_run': (72, 72), '3iteration_num': (10**7, 10**8)}

parameters['sn'] = 5
parameters['limit'] = 1
parameters['cycle'] = 10
parameters['max_evals'] = parameters['sn'] * 4 * parameters['cycle']
parameters['comb'], parameters['folder_path'], parameters['setting_path'], parameters['info_path'] = get_results_path_new('tsp_abc', parameters)
main_tsp('./data/TSP/tsplib_hard', abc, bounds, parameters, solve_tsp_avg_all)