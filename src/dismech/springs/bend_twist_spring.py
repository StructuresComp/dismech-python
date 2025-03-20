import typing
import numpy as np


class BendTwistSpring:
    def __init__(self,
                 nodes_edges_index: np.ndarray,
                 signs: np.ndarray,
                 ref_len_arr: np.ndarray,
                 EI: np.ndarray,
                 GJ: float,
                 map_node_to_dof: typing.Callable[[np.ndarray], np.ndarray],
                 map_edge_to_dof: typing.Callable[[np.ndarray], np.ndarray]):
        self.stiff_EI = EI
        self.stiff_GJ = GJ

        # N e N e N
        self.nodes_ind = [int(nodes_edges_index[0]),
                          int(nodes_edges_index[2]),
                          int(nodes_edges_index[4])]
        self.edges_ind = [int(nodes_edges_index[1]),
                          int(nodes_edges_index[3])]
        self.sgn = signs

        self.ind = np.concatenate((
            map_node_to_dof(self.nodes_ind[0]),
            map_node_to_dof(self.nodes_ind[1]),
            map_node_to_dof(self.nodes_ind[2]),
            np.array([map_edge_to_dof(self.edges_ind[0])]),
            np.array([map_edge_to_dof(self.edges_ind[1])])
        ), axis=0)

        self.voronoi_len = 0.5 * (ref_len_arr[self.edges_ind[0]] +
                                  ref_len_arr[self.edges_ind[1]])
