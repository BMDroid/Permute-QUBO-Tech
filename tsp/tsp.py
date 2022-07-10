'''
@Description: TSP class
@Author: BO Jianyuan
@Date: 2020-02-27 19:12:08
LastEditors: BO Jianyuan
LastEditTime: 2021-02-10 13:47:11
'''

import itertools
import math
import tsplib95

import networkx as nx
import numpy as np
import scipy as sp

from networkx.algorithms.approximation import min_weighted_vertex_cover, ramsey_R2
from scipy.sparse.csgraph import minimum_spanning_tree as mst
from minmax_kmeans import *


class TSP(object):
    def __init__(self, tsp, content=[]):
        def _content_to_coords(content):
            coords = {}
            for node in content:
                if len(node) == 3:
                    coords[int(node[0])] = (np.float(node[1]),
                                            np.float(node[2]))
            return coords

        def _update_coords(coords):
            keys = list(coords.keys())
            for key in keys:
                coords[key - 1] = coords.pop(key)
            return coords

        if tsp:
            if tsp.edge_weight_type == "EUC_2D" or tsp.edge_weight_type is None:  # euclidean distance instance with coords provided
                self._name = tsp.name
                self._coords = _update_coords(tsp.node_coords)
                self._nodes = self._coords.keys()
                self._g = self.get_distance()
                self._no_nodes = tsp.dimension
                self._no_edges = tsp.dimension * (tsp.dimension - 1) / 2
                self._perturbed_g = self.get_perturbed()
                # the rest features are using the perturbed graph
                self._removed_diag = self.get_removed_diag()
                self._min_edge = self._removed_diag[
                    self._removed_diag != 0].min()
                self._max_edge = self._removed_diag.max()
                self._path = []
                self._cost = None
                # self._mean_edge = np.mean(self._removed_diag)
                # self._median_edge = np.median(self._removed_diag)
                # self._var_edge = self._removed_diag.var()
                # upper bound
                # self._greedy, self._greedy_route = self.get_greedy_heuristic()
                # self.two_opt, self.two_opt_route = self.get_two_opt(self._greedy_route)
                # lower bound
                # self._mst = mst(self._perturbed_g).toarray().astype(np.int64).sum()
                # adj matrix
                # self._spect_rad, self._cond_num = self.get_adj()
                # laplacian
                # self._algebra_con = self.get_algbra_con()
            elif tsp.edge_weight_type == "EXPLICIT":
                self._name = tsp.name
                self._coords = None
                self._nodes = list(range(1, tsp.dimension + 1))
                self._g = self.get_distance(tsp.edge_weight_format,
                                            tsp.edge_weights)
                self._no_nodes = tsp.dimension
                self._no_edges = tsp.dimension * (tsp.dimension - 1) / 2
                self._perturbed_g = self.get_perturbed()
                # the rest features are using the perturbed graph
                self._removed_diag = self.get_removed_diag()
                self._min_edge = self._removed_diag[
                    self._removed_diag != 0].min()
                self._max_edge = self._removed_diag.max()
                self._path = []
                self._cost = None
        else:
            self._name = ""
            self._coords = _content_to_coords(content)
            self._nodes = self._coords.keys()
            self._g = self.get_distance()
            self._no_nodes = len(self._nodes)
            self._no_edges = len(self._nodes) * (len(self._nodes) - 1) / 2
            self._perturbed_g = self.get_perturbed(True)
            self._removed_diag = self.get_removed_diag()
            self._min_edge = self._removed_diag[self._removed_diag != 0].min()
            self._max_edge = self._removed_diag.max()

    def calculate_features(self):
        # self._min_edge = self._removed_diag[self._removed_diag != 0].min() uncomment for self-trained model
        self._min_edge = np.min(self._perturbed_g)  # for ratio_nn, this is 0
        self._mean_edge = np.mean(self._removed_diag) / 2
        self._median_edge = np.median(self._removed_diag)
        # self._var_edge = self._removed_diag.var() uncomment for self-trained model
        self._var_edge = TSP.remove_diag_(np.triu(self._perturbed_g)).var()
        self._no_edges = (np.prod(self._perturbed_g.shape) -
                          len(self._perturbed_g[self._perturbed_g <= 0])) / 2
        self._greedy, self._greedy_route = self.get_greedy_heuristic()
        self._two_opt, _ = self.get_two_opt(self._perturbed_g,
                                            self._greedy_route)
        self._mst = mst(self._perturbed_g).toarray().astype(np.int64).sum()
        self._spect_rad, self._cond_num = self.get_adj()
        self._algebra_con = self.get_algbra_con()
        return self._min_edge, self._max_edge, self._mean_edge, self._median_edge, self._var_edge, self._no_nodes, self._no_edges, self._greedy, self._two_opt, self._mst, self._spect_rad, self._cond_num, self._algebra_con

    def get_distance(self, edge_weight_format=None, edge_weights=None):
        if self._coords:
            return np.array([[
                int(
                    math.sqrt(
                        (self._coords[start][0] - self._coords[end][0])**2 +
                        (self._coords[start][1] - self._coords[end][1])**2) +
                    .5) for start in self._nodes
            ] for end in self._nodes])
        elif edge_weight_format == "UPPER_ROW":
            upper_weights = edge_weights
            upper_weights.append([])
            weights = [[0] * (i + 1) + lst
                       for i, lst in enumerate(upper_weights)]
            weights = np.array(weights)
            weights = weights + weights.T
            return weights
        elif edge_weight_format == "LOWER_DIAG_ROW":
            lower_weights = sum(edge_weights, [])
            k = sum(1 for x in lower_weights if x > 0)
            n = int((1 + math.sqrt(1 + 4 * (2 * k))) / 2)
            weights = []
            for i in range(n):
                weights.append(lower_weights[0:i + 1] + [0] * (n - i - 1))
                lower_weights = lower_weights[i + 1:]
            weights = np.array(weights)
            weights = weights + weights.T
            return weights
        else:  # full matrix
            return np.array(edge_weights)

    def get_perturbed(self, flag=True):
        if flag:
            self._perturbed_g = TSP.perturb(self._g)
        else:
            self._perturbed_g = self._g
        return self._perturbed_g

    def get_removed_diag(self):
        self._removed_diag = TSP.remove_diag(self._perturbed_g)
        return self._removed_diag

    def get_greedy_heuristic(self):
        graph = self._perturbed_g
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
            while TSP.check_unvisited_node(unvisited) and iteration < node_no:
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
        [shortest_min_distance,
         shortest_travel_route] = TSP.find_best_route(node_no, travel_route,
                                                      min_distance)
        return shortest_min_distance, shortest_travel_route

    def get_two_opt(self, graph, route):
        # graph = self._perturbed_g
        best = route
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for j in range(i + 1, len(route)):
                    if j - i == 1: continue
                    if TSP.cost_change(graph, best[i - 1], best[i],
                                       best[j - 1], best[j]) < 0:
                        best[i:j] = best[j - 1:i - 1:-1]
                        improved = True
            route = best
        dist = 0
        for i in range(len(route)):
            dist += graph[route[i]][route[(i + 1) % len(route)]]
        return dist, best

    def get_adj(self):
        eigenvalues = sp.linalg.eigvals(self._perturbed_g)
        return sp.absolute(sp.real(eigenvalues.max())), sp.absolute(
            sp.real(eigenvalues.max()) / sp.real(eigenvalues.min()))

    def get_algbra_con(self):
        L = TSP.laplacian(self._perturbed_g, False, self._no_nodes)
        leigenvalues = np.sort(sp.linalg.eigvals(L))
        return sp.real(leigenvalues[1])

    def k_part(self, verbose=True):
        '''
        partition large Graph instance into smaller ones as per max_size = 30
        hard limit max. iterations to 10
        Constrained kmeans := min_size and max_size variant
        Code from here:=https://github.com/Behrouz-Babaki/MinSizeKmeans
        Credits/Authors"=Behrouz Babaki
                         James Vogel
                         Ofir Reich
                         Paul Gölz
        :return iteratable {Graph} * k:
        '''

        iter_num = 1
        max_size = min(30, math.ceil(self._nodes.__len__() / 2))
        k = math.ceil(self._nodes.__len__() / max_size)
        best = None
        best_clusters = None
        data = list(self._coords.values())
        for i in range(iter_num):
            clusters, centers = minsize_kmeans(data, k, 7, max_size)
            if clusters:
                quality = compute_quality(data, clusters)
                if not best or (quality < best):
                    best = quality
                    best_clusters = clusters
        if best:
            content_dict = {}

            for node in range(self._nodes.__len__()):
                if clusters[node] not in content_dict.keys():
                    content_dict[clusters[node]] = []
                content_dict[clusters[node]].append(
                    [node, self._coords[node][0], self._coords[node][1]])
            if verbose:
                print(
                    "Sum of squared distances for constrained k-means: {:.4f}".
                    format(best))

            tsp_dict = {}
            for centroid in range(k):
                if centroid not in tsp_dict.keys():
                    tsp_dict[centroid] = None
                tsp_dict[centroid] = TSP(None, content=content_dict[centroid])

            return tsp_dict, centers

        else:
            print('No clustering found')

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
    def upper(g, flag=True):
        '''Get the upper triangular matrix of G
        default: filter out the zero elements
        '''
        if flag:
            return np.triu(g)[np.triu(g) != 0]
        return np.triu(g)

    @staticmethod
    def avg_edge_weight(g):
        '''Calculate the average weight of the graph
        '''
        return np.sum(g) / (g.shape[0]**2 - g.shape[0])

    @staticmethod
    def std_edge_weight(g):
        '''Calculate the std of the weight of the graph
        '''
        g_temp = g[g != 0]  # .reshape(g.shape[0], g.shape[0] - 1)
        return np.std(g_temp)

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
    def laplacian(g, is_geo, n):
        if is_geo:
            A = np.array([[1 if g[i, j] > 1 else 0 for j in range(n)]
                          for i in range(n)])
        else:
            A = np.array([[1 if g[i, j] > 0 else 0 for j in range(n)]
                          for i in range(n)])
        return np.subtract(np.diag(A.sum(axis=1)), A)

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


def load_graph(fileName):
    ''' Load graph
    '''
    try:
        return np.loadtxt(fileName)
    except:
        return np.load(fileName, allow_pickle=True)