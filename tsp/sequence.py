import copy
import itertools
import math
import os
import sys

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from pathlib import Path
from operator import itemgetter
from scipy.sparse.csgraph import minimum_spanning_tree as mst
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering


class Parser(object):
    def __init__(self, fileName):
        self._name = Path(fileName).stem
        # _, self._name =os.path.split(fileName)
        self._Jobs = None
        self._jobs = None
        self._machines = None
        self._path = fileName
        self._seq = self.parse()

    def parse(self):
        self._Jobs = []
        with open(self._path, "r") as f:
            lines = f.readlines()
            setting = list(map(int, lines[0].split()))
            self._jobs, self._machines = setting[0], setting[1]
            for i in range(self._jobs):
                proc = list(map(int, lines[i + 1].split()))
                del proc[::2]
                proc = proc[:self._machines]
                self._Jobs.append(Job(i + 1, self._machines, proc))
        self._Jobs.insert(0, Job(0, self._machines, [0] * self._machines))
        self._seq = Sequence(self._Jobs)
        return self._seq


class Job(object):
    '''Job object: id, processing time for all the operations, and total processing time.
    '''
    def __init__(self, id, m, p_time=[None]):
        self._id = id
        self._machines = m
        self._p_time = p_time
        self._total_p = sum(p_time)

    def change_id(self, id):
        self._id = id
        return self

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    def __str__(self):
        return "{0}: {1}".format(self._id, self._p_time)


class Sequence(object):
    '''A sequence of jobs.
    '''
    def __init__(self, jobs):
        self._Jobs = sorted(jobs, key=lambda x: x._id)
        self._machines = jobs[0]._machines
        self._jobs = len(jobs)
        self._order = [j._id for j in self._Jobs]
        self._proc = np.array([np.array(j._p_time) for j in self._Jobs])
        self._f = 0  # self.functional()
        self._f_order = None  # self.functional_order()
        self._span = np.zeros((self._jobs, self._machines))
        self._start = np.zeros((self._jobs, self._machines))
        self._idle = np.zeros((self._machines, self._jobs - 1))
        #         self._makespan = self.makespan()
        self._distance = np.zeros((self._jobs, self._jobs))
        self._perturbed_distance = 0
        self._removed_diag = 0
        self._max_edge = 0
        self._min_edge = 0
        self._mean_edge = 0
        self._median_edge = 0
        self._var_edge = 0
        self._greedy = 0
        self._two_opt = 0
        self._mst = 0
        self._spect_rad, self._cond_num = 0, 0
        self._algebra_con = 0

    def calculate_features(self):
        assert np.sum(
            self._distance
        ) >= 0  # make sure that the distance matrix is calculated
        self._perturbed_distance = Sequence.perturb(self._distance)
        self._removed_diag = Sequence.remove_diag(self._perturbed_distance)
        # self._min_edge = self._removed_diag[self._removed_diag != 0].min() uncomment for self-trained model
        self._min_edge = np.min(
            self._perturbed_distance)  # for ratio_nn, this is 0
        self._max_edge = self._removed_diag.max()
        self._mean_edge = np.mean(self._removed_diag) / 2
        self._median_edge = np.median(self._removed_diag)
        # self._var_edge = self._removed_diag.var() uncomment for self-trained model
        self._var_edge = Sequence.remove_diag_(
            np.triu(self._perturbed_distance)).var()
        self._no_nodes = int(self._perturbed_distance.shape[0])
        self._no_edges = (np.prod(self._perturbed_distance.shape) - len(
            self._perturbed_distance[self._perturbed_distance <= 0])) / 2
        self._greedy, self._greedy_route = Sequence.calculate_greedy(
            self._perturbed_distance)
        self._two_opt, _ = Sequence.calculate_two_opt(self._perturbed_distance,
                                                      self._greedy_route)
        self._mst = mst(self._perturbed_distance).toarray().astype(
            np.int64).sum()
        self._spect_rad, self._cond_num = Sequence.calculate_adj(
            self._perturbed_distance)
        self._algebra_con = Sequence.calculate_algbra_con(
            self._perturbed_distance)
        return self._min_edge, self._max_edge, self._mean_edge, self._median_edge, self._var_edge, self._no_nodes, self._no_edges, self._greedy, self._two_opt, self._mst, self._spect_rad, self._cond_num, self._algebra_con

    def two_opt(self, order, makespan):

        best_order = order.copy()
        best_makespan = makespan
        improved = True
        while improved:
            improved = False
            for i in range(1, len(best_order) - 1):
                for k in range(i + 1, len(best_order)):
                    if k - i == 1:
                        continue
                    new_order = best_order[0:i] + list(
                        reversed(best_order[i:k])) + best_order[k:]
                    idx = new_order.index(0)
                    new_order = new_order[idx + 1:] + new_order[
                        0:idx + 1]  # put the 0 at the end of the sequence
                    self.reorder(new_order)
                    new_makespan = self._makespan
                    if new_makespan < best_makespan:
                        best_order = new_order.copy()
                        best_makespan = new_makespan
                        improved = True
        self._order = best_order
        self._makespan = best_makespan
        return best_makespan, best_order

    def insert(self, index, job):
        '''Insert a job in the sequence and recalculate the makespan.
        '''
        self._jobs += 1
        self._order.insert(index, job._id)
        self._proc = np.insert(self._proc, index, np.array(job._p_time), 0)
        self._span = np.zeros((self._jobs, self._machines))
        self._makespan = self.makespan()

    def swap(self, ind1, ind2, flag='index'):
        '''Swap two jobs in the sequence.
        '''
        if flag != 'index':
            ind1 = self._order.index(ind1)
            ind2 = self._order.index(ind2)

        self._order[ind1], self._order[ind2] = self._order[ind2], self._order[
            ind1]
        self._proc[[ind1, ind2]] = self._proc[[ind2, ind1]]
        self._span = np.zeros((self._jobs, self._machines))
        self._makespan = self.makespan()

    def reorder(self, new_order, flag='job'):
        '''Reorder the sequence by an list of job id.
        '''
        # self._order = new_order
        if flag != 'job':
            self._proc = self._proc[new_order]
            self._order = [self._order[i] for i in new_order]
        else:
            indx = [self._order.index(i) for i in new_order]
            self._proc = self._proc[indx]
            self._order = new_order
        self._span = np.zeros((self._jobs, self._machines))
        self._makespan = self.makespan()

    def makespan(self):
        '''Calculate the makespan of the jobs.
        '''
        self._span[0] = [
            sum(self._proc[0, 0:i + 1]) for i in range(self._machines)
        ]
        for i in range(1, self._jobs):
            for j in range(0, self._machines):
                if j == 0:
                    self._span[i, 0] = self._span[i - 1, 0] + self._proc[i, 0]
                else:
                    self._span[i, j] = max(self._span[i, j - 1],
                                           self._span[i - 1,
                                                      j]) + self._proc[i, j]
        self._makespan = self._span[-1, -1]
        self._start = self._span - self._proc
        return self._makespan

    def machine_idle(self):
        '''Calculate the machine idle time.
        '''
        for i in range(self._jobs - 1):
            self._idle[:, i] = self._start[i + 1] - self._start[i]
        self._idle -= self._proc[:-1].T
        return self._idle

    def job_distance(self, flag):
        self._distance = np.zeros((self._jobs, self._jobs))
        if flag == 'abs':
            distance = self.residual_absolute()
        elif flag == 'square':
            distance = self.residual_square()
        elif flag == 'nocarry':
            distance = self.residual_no_carryover()
        elif flag == 'spirit':
            distance = self.spirit()
        else:
            distance = self.fshoph()
        self._max_edge = np.max(distance)
        return self._distance

    def residual_absolute(self):
        '''Calculate the distance between each job using summation of the absolute of the residual
        '''
        for i in range(self._jobs):
            index_i = self._order.index(i)
            for j in range(self._jobs):
                index_j = self._order.index(j)
                if index_i == index_j:
                    self._distance[i][j] = 0
                else:
                    self._distance[i][j] = sum(
                        abs(self._proc[index_i][1:] -
                            self._proc[index_j][:-1]))
        return self._distance

    def residual_square(self):
        '''Calculate the distance between each job using summation of the square of the residual
        '''
        for i in range(self._jobs):
            index_i = self._order.index(i)
            for j in range(self._jobs):
                index_j = self._order.index(j)
                if index_i == index_j:
                    self._distance[i][j] = 0
                else:
                    self._distance[i][j] = sum(
                        np.power(
                            self._proc[index_i][1:] - self._proc[index_j][:-1],
                            2))
        return self._distance

    def residual_no_carryover(self):
        ''' Calculate the distance between each job.
        '''
        for i in range(self._jobs):
            index_i = self._order.index(i)
            for j in range(self._jobs):
                index_j = self._order.index(j)
                if index_i == index_j:
                    self._distance[i][j] = 0
                else:
                    res = self._proc[index_i][1:] - self._proc[index_j][:-1]
                    self._distance[i][j] = sum(
                        res[res > 0]) + 2 * sum(abs(res[res < 0]))
        return self._distance

    def spirit(self):
        ''' Calculate the SPIRIT distance from Widmer and Hertz
        '''
        weights = np.asarray(list(range(self._machines - 2, -1, -1)))
        for i in range(self._jobs):
            index_i = self._order.index(i)
            for j in range(self._jobs):
                index_j = self._order.index(j)
                if index_i == index_j:
                    self._distance[i][j] = 0
                else:
                    res = self._proc[index_i][1:] - self._proc[index_j][:-1]
                    tmp = np.sum(weights * np.abs(res))
                    dis = self._proc[index_i][0] + tmp + (
                        self._machines - 1) * self._proc[index_j][-1]
                    self._distance[i][j] = dis
        return self._distance

    def fshoph(self):
        ''' Calculate the FSHOPH distance from Moccellin
        '''
        for i in range(self._jobs):
            index_i = self._order.index(i)
            for j in range(self._jobs):
                index_j = self._order.index(j)
                if index_i == index_j:
                    self._distance[i][j] = 0
                else:
                    dis = 0
                    for k in range(1, self._machines):
                        dis = max(
                            0, dis + self._proc[index_j][k - 1] -
                            self._proc[index_i][k])
                    self._distance[i][j] = dis
        return self._distance

    def cluster(self, flag, a=0, k=30):
        seq = copy.deepcopy(self)
        decompositions = []

        dis = seq._distance
        delta = np.mean(dis)
        sim = np.exp(-dis**2 / (2. * delta**2))

        dis = np.max(np.array([dis, dis.T]), axis=0)
        #         dis = np.min(np.array([dis, dis.T]), axis=0)
        #         dis = (dis + dis.T) / 2
        sim = np.exp(-dis**2 / (2. * np.mean(dis)**2))
        n_clusters = max(1, a + seq._jobs // k)
        if flag == 'spectral':
            clustering = SpectralClustering(
                n_clusters, affinity='precomputed').fit_predict(sim)
        elif flag == 'agglomerative':
            clustering = AgglomerativeClustering(
                n_clusters, affinity='precomputed',
                linkage='complete').fit_predict(dis)
        elif flag == 'optics':
            clustering = OPTICS(min_samples=10,
                                metric='precomputed').fit_predict(dis)

        for i in np.unique(clustering):
            cluster = np.where(clustering == i)[0]
            jobs_ids = cluster.tolist()
            try:
                jobs_ids.remove(0)
            except:
                jobs_ids = jobs_ids

            if len(jobs_ids) == 0:
                continue
            elif len(jobs_ids) == 1:
                cluster_jobs = [seq._Jobs[jobs_ids[0]]]
            else:
                cluster_jobs = list(itemgetter(*jobs_ids)(seq._Jobs))

            # keep job0's id unchanged
            ids_dict = dict(zip(jobs_ids, range(1, len(jobs_ids) + 1)))
            cluster_jobs.append(self._Jobs[0])
            jobs_ids.append(0)
            ids_dict[0] = 0

            # change the id of the jobs for distance matrix computation
            cluster_jobs = list(
                map(lambda x: x.change_id(ids_dict[x._id]), cluster_jobs))

            # store decomposition
            decompositions.append((cluster_jobs, ids_dict))

        return decompositions

    def functional(self):
        self._f = {}
        for idx, j in enumerate(self._Jobs):
            j_id = j._id
            A = 1 if self._proc[idx, -1] <= self._proc[idx, 0] else -1
            denom = np.min([
                self._proc[idx, i] + self._proc[idx, i + 1]
                for i in range(self._machines - 1)
            ])
            if denom == 0:
                self._f[j_id] = float('inf')
            else:
                self._f[j_id] = A / denom
        return self._f

    def functional_order(self):
        f_order = sorted(self._f.items(), key=lambda x: x[1])
        self._f_order = [f[0] for f in f_order]
        return self._f_order

    def __str__(self):
        return "{}: {}".format(self._order, self._makespan)

    def plot(self, w=5, sort_colors=True):
        colors = {
            **mcolors.BASE_COLORS,
            **mcolors.CSS4_COLORS,
            **mcolors.TABLEAU_COLORS
        }
        if sort_colors is True:
            by_hsv = sorted(
                (tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))), name)
                for name, color in colors.items())
            colors = [name for hsv, name in by_hsv][::-1]
            colors = colors[:-56]
        else:
            colors = list(colors)
        plt.axes()
        for i in range(self._jobs):
            for j in range(self._machines):
                if j == 0:
                    rectangle = plt.Rectangle(
                        (self._start[i][j], (self._machines - j) * w),
                        self._proc[i][j],
                        w,
                        fc=colors[self._order[i] * 7],
                        label=str(self._order[i]))
                else:
                    rectangle = plt.Rectangle(
                        (self._start[i][j], (self._machines - j) * w),
                        self._proc[i][j],
                        w,
                        fc=colors[self._order[i] * 7])
                plt.gca().add_patch(rectangle)
        frame = plt.gca()
        frame.axes.get_yaxis().set_visible(False)
        plt.axis('auto')
        plt.text(self._makespan * 0.7, (self._machines + 0.45) * w,
                 'makespan: ' + str(self._makespan),
                 bbox=dict(facecolor='red', alpha=0.5))
        plt.legend(loc='lower left')
        plt.grid()
        plt.show()

    def neh(self):
        '''Return the minimum makespan using NEH.
        '''
        min_sequence = Sequence.minimum_sequence(self._Jobs)
        min_order = min_sequence._order
        self.reorder(min_order)
        return self._makespan

    @staticmethod
    def descending_order(jobs):
        '''Return the list of jobs in descending order of total processing time on all the machines.
        '''
        return sorted(jobs, key=lambda x: x._total_p, reverse=True)

    @staticmethod
    def partial_sequence(sequence, job=None):
        ''' Return a sequence whose partial makespan is the minimum.
        '''
        if sequence._jobs <= 1:
            return sequence
        if not job:
            sequence_copy = copy.deepcopy(sequence)
            sequence_copy.swap(0, 1)
            if sequence._makespan >= sequence_copy._makespan:
                return sequence_copy
            return sequence
        min_span = float('inf')
        min_sequence = None
        for i in range(sequence._jobs + 1):
            sequence_copy = copy.deepcopy(sequence)
            sequence_copy.insert(i, job)
            if min_span >= sequence_copy._makespan:
                min_span = sequence_copy._makespan
                min_sequence = sequence_copy
        return min_sequence

    @staticmethod
    def minimum_sequence(jobs):
        '''Return a sequence with the minimum makespan.
        '''
        if len(jobs) <= 1:
            return jobs
        jobs = Sequence.descending_order(jobs)
        jobs_max_2 = jobs[:2]
        count = 1
        sequence = Sequence(jobs_max_2)
        while (len(jobs) > count):
            if count == 1:
                sequence = Sequence.partial_sequence(sequence)
            else:
                sequence = Sequence.partial_sequence(sequence, jobs[count])
            count += 1
        return sequence

    @staticmethod
    def perturb(g):
        n = g.shape[0]
        a = np.full((n, n), -1. / n) + np.identity(n)
        tmp_sum = np.sum(g) / float(n)
        v = np.array([np.sum(g[i, :] + g[:, i]) for i in range(n)])
        v = tmp_sum - v * 0.5
        v = v / (n - 2.0)
        delta = np.linalg.lstsq(a, v, rcond=-1)[0]
        new_g = np.copy(g)
        min_num = float('inf')
        for (i, j) in list(itertools.product(range(n), range(n))):
            if i != j:
                new_g[i, j] += delta[i] + delta[j]
                min_num = min(min_num, new_g[i, j])
        for (i, j) in list(itertools.product(range(n), range(n))):
            if i != j:
                new_g[i, j] -= min_num
        return new_g

    @staticmethod
    def remove_diag(g):
        tmp = g.copy()
        np.fill_diagonal(tmp, -1)
        return tmp[tmp != -1].reshape(tmp.shape[0], tmp.shape[0] - 1)

    @staticmethod
    def remove_diag_(g):  # calculate features for ratio_nn
        no_diag = np.ndarray.flatten(g)
        no_diag = np.delete(no_diag, range(0, len(no_diag), len(g) + 1), 0)
        no_diag = no_diag.reshape(len(g), len(g) - 1)
        return no_diag

    @staticmethod
    def find_best_route(node_no, travel_route, min_distance):
        shortest_travel_route = travel_route[0]
        shortest_min_distance = min_distance.item(0)
        for start_node in range(0, node_no):
            if min_distance[start_node] < shortest_min_distance:
                shortest_min_distance = min_distance.item(start_node)
                shortest_travel_route = travel_route[start_node]
        return shortest_min_distance, shortest_travel_route

    @staticmethod
    def check_unvisited_node(unvisited):
        for u in unvisited:
            if u == 1:
                return True
        return False

    @staticmethod
    def cost_change(graph, n1, n2, n3, n4):
        return graph[n1][n3] + graph[n2][n4] - graph[n1][n2] - graph[n3][n4]

    @staticmethod
    def calculate_greedy(graph):
        # remark: most code comes from https://github.com/theyusko/tsp-heuristics/blob/master/algo/nearest_neighbor.py
        node_no = graph.shape[0]
        min_distance = np.zeros(
            (node_no, ),
            dtype=float)  # distances with starting node as min_distance[i]
        travel_route = [[0 for x in range(0, node_no)]
                        for y in range(0, node_no)]
        # Step 1
        for start_node in range(0, node_no):
            # Step 3
            unvisited = np.ones((node_no, ),
                                dtype=int)  # all nodes are unvisited
            unvisited[start_node] = 0
            travel_route[start_node][
                0] = start_node  # travel route starts with start_node
            node = start_node
            iteration = 1
            while Sequence.check_unvisited_node(
                    unvisited) and iteration < node_no:
                # Step 2
                closest_arc = float('inf')
                closest_node = node_no
                for node2 in range(0, node_no):
                    if unvisited[node2] == 1 and 0 < graph[node][
                            node2] < closest_arc:
                        closest_arc = graph[node][node2]
                        closest_node = node2
                if closest_node >= node_no:
                    min_distance[start_node] = float('inf')
                    break
                node = closest_node
                unvisited[node] = 0
                min_distance[
                    start_node] = min_distance[start_node] + closest_arc
                travel_route[start_node][iteration] = node
                iteration = iteration + 1
            if not math.isinf(min_distance[start_node]):
                last_visited = travel_route[start_node][node_no - 1]
                if graph[last_visited][start_node] > 0:
                    min_distance[start_node] = min_distance[
                        start_node] + graph[last_visited][start_node]
                else:
                    min_distance[start_node] = float('inf')
        [shortest_min_distance, shortest_travel_route
         ] = Sequence.find_best_route(node_no, travel_route, min_distance)
        return shortest_min_distance, shortest_travel_route

    @staticmethod
    def calculate_two_opt(graph, route):
        best = route
        improved = True
        max_iter = 1000
        iter = 0
        while improved and iter <= max_iter:
            iter += 1
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route)):
                    if j - i == 1: continue
                    if Sequence.cost_change(graph, best[i - 1], best[i],
                                            best[j - 1], best[j]) < 0:
                        best[i:j] = best[j - 1:i - 1:-1]
                        improved = True
            route = best
        dist = 0
        for i in range(len(route)):
            dist += graph[route[i]][route[(i + 1) % len(route)]]
        return dist, best

    @staticmethod
    def calculate_adj(graph):
        eigenvalues = sp.linalg.eigvals(graph)
        return sp.absolute(sp.real(eigenvalues.max())), sp.absolute(
            sp.real(eigenvalues.max()) / sp.real(eigenvalues.min()))

    @staticmethod
    def calculate_algbra_con(graph):
        L = Sequence.laplacian(graph, False, graph.shape[0])
        leigenvalues = np.sort(sp.linalg.eigvals(L))
        return sp.real(leigenvalues[1])

    @staticmethod
    def laplacian(g, is_geo, n):
        if is_geo:
            A = np.array([[1 if g[i, j] > 1 else 0 for j in range(n)]
                          for i in range(n)])
        else:
            A = np.array([[1 if g[i, j] > 0 else 0 for j in range(n)]
                          for i in range(n)])
        return np.subtract(np.diag(A.sum(axis=1)), A)
