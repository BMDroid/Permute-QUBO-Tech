'''
Description: Sticher class for TSP
Author: BO Jianyuan
Date: 2020-12-14 13:58:58
LastEditors: BO Jianyuan
LastEditTime: 2021-01-10 16:41:01
'''

from builder import *
import solver
from solver import *


class Stitcher(object):
    _set_enum_size = 0
    _set_comp_size = 0
    _allowed_comp_size = 0

    def __init__(self, clusters, g):
        self._clusters = clusters
        self._cost = 0
        self._path = []
        self._g = g
        Stitcher._allowed_comp_size = clusters.__len__() // 2

    def generate_join_scheme(self):
        '''
        Join clusters
        Using Hamiltonian path QUBO - Andrew Lucas
        :return: scheme, flip_cost,flip_set
        '''
        def _least_cost_flip(self):
            '''
            Complexity ~ O((k^2)(|E|^2))
            :param self: k-part object
            :return: least_cost_flip format {(0,1):2500,(3,2):-500,...}
                     least_cost_flip_set format {(0,1):{((15,16),(7,8)):2},...}
            '''
            least_cost_flip_set = {}
            least_cost_flip = {}
            for c1 in self._clusters.keys():
                for c2 in self._clusters.keys():
                    if c1 != c2:
                        if (c1, c2) not in least_cost_flip_set.keys() and (
                                c2, c1) not in least_cost_flip_set.keys():
                            #                             print(self._clusters[c1], self._clusters[c2], least_cost_flip_set, c1, c2)
                            least_cost_flip_set[(c1, c2)], least_cost_flip[(c1, c2)] = \
                                compute_least_cost_flip_unique(self._clusters[c1], self._clusters[c2], least_cost_flip_set, c1, c2)
            return least_cost_flip, least_cost_flip_set

        def _arrange_order(least_cost, self):

            n = self._clusters.__len__()
            g = np.array([[
                least_cost[(j,
                            i)] if i > j else least_cost[(i,
                                                          j)] if i < j else 0
                for j in range(n)
            ] for i in range(n)])
            g += np.abs(g.min())

            # Replace infinite paths with large numbers
            M = g[g != np.inf].sum()
            g[g == np.inf] = M
            del (M)

            # solve
            result = solver.stitch_tsp_cluster(n, g)

            soln_path, soln_cost = compute_path_distance(n, result[0], g)
            if soln_cost < 0:
                print("No feasible solution found")
                # print([], -1)
                # exit()
                return {}
            print("delta: {}".format(soln_cost))
            # print(soln_path)
            return {
                soln_path[i - 1]:
                (soln_path[i - 1],
                 soln_path[i]) if soln_path[i - 1] < soln_path[i] else
                (soln_path[i], soln_path[i - 1])
                for i in range(1, len(soln_path))
            }

        least_cost_flip, least_cost_flip_set = _least_cost_flip(self)
        join = _arrange_order(least_cost_flip, self)

        return join, least_cost_flip, least_cost_flip_set

    def stitch(self, join, least_cost_flip, least_cost_flip_set):
        '''
        Stitch everything together
        Instead of relying on reversed indicator, use vertex information
        '''
        def _decide_order(join, key, least_cost_flip_set):
            '''
            Key function that provides the correct order!
            :param join:
            :param key:
            :param least_cost_flip_set:
            :return: host, entre, order_host, order_entre, reverse
            '''

            if key == join[key][0]:
                return key, join[key][1], list(least_cost_flip_set[join[key]].keys())[0][0], \
                       list(least_cost_flip_set[join[key]].keys())[0][1], \
                       list(least_cost_flip_set[join[key]].values())[0]
            else:
                return key, join[key][0], list(least_cost_flip_set[join[key]].keys())[0][1], \
                       list(least_cost_flip_set[join[key]].keys())[0][0], \
                       list(least_cost_flip_set[join[key]].values())[0]

        def _get_c1_pair(nodes, order, base_path):
            '''
            Once the decision on host and entre is made,
            get vertex entre points
            :param nodes: vertices of host
            :param order: tuple to consider
            :param order: to check if order in base_path flipped?
            :return:
            '''

            h_order = (list(nodes)[order[0]], list(nodes)[order[1]])
            if base_path.index(h_order[0]) > base_path.index(h_order[1]):
                return h_order, True
            else:
                return h_order, False

        def _prepare_c2(nodes, path_entre, order, reverse, base_path_reversed):
            '''
            Once we know entre, we prepare it for
            penetration
            :param nodes: nodes in entre
            :param path: shortest path in entre
            :param order: tuple to consider
            :param reverse: 1 => =, 2=> x
            :return:
            '''

            nec_1, nec_2 = list(nodes)[order[0]], list(nodes)[order[1]]
            path_entre = [list(nodes)[p] for p in path_entre]
            st1 = path_entre.index(nec_1)
            st2 = path_entre.index(nec_2)
            if abs(st1 - st2) == path_entre.__len__(
            ) - 1:  # I know st2 > st1 but FLUFFS: Allow chance to fluff out errors!
                if st1 < st2:
                    path_entre.reverse()  # Bring back to regular format
            else:
                path_entre = path_entre[max(st1, st2):] + path_entre[:max(
                    st1, st2
                )]  # I know st2 > st1 but FLUFFS: Allow chance to fluff out errors!
            if reverse < 2:  # Keeping it simple for Gods sake!!!!!!!!!!
                path_entre.reverse()
            if base_path_reversed:
                path_entre.reverse()
            return path_entre

        def _join_c1_c2(path, c1_pair, c2_path):
            '''
            Simple stitch logic
            Scan through path of cluster 1 (path) until
            you encounter either of c1_pair.
            Once you do, check next if other.
            If so append here.
            Else, ignore everything and continue and
            append when you find other at path (end)
            :param path: modified for proper label
            :param c1_pair: c1 pairs to consider
            :param c2_path: c2 path to append modified for proper label
            :return:
            '''

            for node_idx in range(path.__len__()):
                if path[node_idx] in c1_pair:
                    if path[(node_idx + 1) % path.__len__()] in c1_pair:
                        break
            path.insert(node_idx + 1, c2_path)
            return list(Stitcher.flatten(path))

        path = []
        path_l = []
        cost = np.inf
        for idx, key in enumerate(join.keys()):
            host, entre, host_order, entre_order, reverse = \
                _decide_order(join, key, least_cost_flip_set)
            g1 = self._clusters[host]
            g2 = self._clusters[entre]

            if idx == 0:
                cost = g1._cost
                path = path_l = [list(g1._nodes)[p] for p in g1._path]

            host_pair, bp_rev = _get_c1_pair(g1._nodes, host_order, path_l)

            entre_path = _prepare_c2(g2._nodes, g2._path[:], entre_order,
                                     reverse, bp_rev)

            path_l = _join_c1_c2(path[:], host_pair, entre_path)

            cost += g2._cost + least_cost_flip[join[key]]

            if cost < sum([
                    self._g[path_l[i], path_l[(i + 1) % path_l.__len__()]]
                    for i in range(path_l.__len__())
            ]):
                entre_path.reverse()
                path = _join_c1_c2(path[:], host_pair, entre_path)
            else:
                path = path_l

        self._path = path
        self._cost = cost

    @staticmethod
    def rotate(lst, x):
        return lst[-x:] + lst[:-x]

    @staticmethod
    def flatten(iterable):
        for item in iterable:
            try:
                yield from Stitcher.flatten(item)
            except TypeError:
                yield item


def compute_least_cost_flip_unique(tsp1, tsp2, r_flip_set, c1, c2):
    def _get_edge_set(tsp, g):
        dict = {}
        for i in range(tsp._path.__len__()):
            if i + 1 < tsp._path.__len__():
                dict[(tsp._path[i],
                      tsp._path[i + 1])] = g[tsp._path[i]][tsp._path[i + 1]]
            else:
                dict[(tsp._path[i],
                      tsp._path[0])] = g[tsp._path[i]][tsp._path[0]]
        return dict

    def _permit(e1, e2, r_flip_set, c1, c2):

        keys = [k for k in list(r_flip_set.keys()) if c1 in k or c2 in k]
        for k in keys:
            try:
                if e1 == list(r_flip_set[k].keys())[0][list(k).index(c1)]:
                    return False
            except:
                try:
                    if e2 == list(r_flip_set[k].keys())[0][list(k).index(c1)]:
                        return False
                except:
                    return True
        return True

    g1 = tsp1._g
    edge_g1 = _get_edge_set(tsp1, tsp1._g)
    g2 = tsp2._g
    edge_g2 = _get_edge_set(tsp2, tsp2._g)

    min_local_cost = np.inf
    min_flip_set = {}
    for e1 in edge_g1.keys():
        for e2 in edge_g2.keys():
            if _permit(e1, e2, r_flip_set, c1, c2):
                local_cost = g1[e1[0]][e1[1]] + g2[e2[0]][e2[1]]

                flip_cost1 = int(math.sqrt((tsp1._coords[list(tsp1._nodes)[e1[0]]][0] -
                                            tsp2._coords[list(tsp2._nodes)[e2[0]]][0]) ** 2 + \
                                        (tsp1._coords[list(tsp1._nodes)[e1[0]]][1] -
                                            tsp2._coords[list(tsp2._nodes)[e2[0]]][1]) ** 2) + 0.5) + \
                            int(math.sqrt((tsp1._coords[list(tsp1._nodes)[e1[1]]][0] -
                                            tsp2._coords[list(tsp2._nodes)[e2[1]]][0]) ** 2 + \
                                        (tsp1._coords[list(tsp1._nodes)[e1[1]]][1] -
                                            tsp2._coords[list(tsp2._nodes)[e2[1]]][1]) ** 2) + 0.5)

                flip_cost2 = int(math.sqrt((tsp1._coords[list(tsp1._nodes)[e1[0]]][0] -
                                            tsp2._coords[list(tsp2._nodes)[e2[1]]][0]) ** 2 + \
                                        (tsp1._coords[list(tsp1._nodes)[e1[0]]][1] -
                                            tsp2._coords[list(tsp2._nodes)[e2[1]]][1]) ** 2) + 0.5) + \
                            int(math.sqrt((tsp1._coords[list(tsp1._nodes)[e1[1]]][0] -
                                            tsp2._coords[list(tsp2._nodes)[e2[0]]][0]) ** 2 + \
                                        (tsp1._coords[list(tsp1._nodes)[e1[1]]][1] -
                                            tsp2._coords[list(tsp2._nodes)[e2[0]]][1]) ** 2) + 0.5)

                if flip_cost1 < flip_cost2:
                    if flip_cost1 - local_cost < min_local_cost:
                        min_local_cost = flip_cost1 - local_cost
                        min_flip_set.clear()
                        min_flip_set[(e1, e2)] = 1

                else:
                    if flip_cost2 - local_cost < min_local_cost:
                        min_local_cost = flip_cost2 - local_cost
                        min_flip_set.clear()
                        min_flip_set[(e1, e2)] = 2
    return min_flip_set, min_local_cost