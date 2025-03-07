import typing
import numpy as np


class TriangleSpring:

    def __init__(self,
                 nodes_ind: np.ndarray,
                 edges_ind: np.ndarray,
                 face_edges: np.ndarray,
                 signs: np.ndarray,
                 ref_len_arr: np.ndarray,
                 A: np.ndarray,
                 init_ts: np.ndarray,
                 init_fs: np.ndarray,
                 init_cs: np.ndarray,
                 init_xis: np.ndarray,
                 kb: float,
                 nu: float,
                 map_node_to_dof: typing.Callable[[np.ndarray], np.ndarray],
                 map_face_edge_to_dof: typing.Callable[[np.ndarray], np.ndarray]):
        self.kb = kb
        self.nu = nu
        self.nodes_ind = nodes_ind
        self.edges_ind = edges_ind
        self.sgn = signs
        self.face_edges = face_edges

        # Select necessary initial values
        self.ref_len = ref_len_arr[face_edges]

        self.A = A
        self.init_ts = init_ts
        self.init_fs = init_fs
        self.init_cs = init_cs
        self.init_xis = init_xis

        self.ind = np.concat([map_node_to_dof(nodes_ind[0]),
                              map_node_to_dof(nodes_ind[1]),
                              map_node_to_dof(nodes_ind[2]),
                              np.array([map_face_edge_to_dof(edges_ind[0])]),
                              np.array([map_face_edge_to_dof(edges_ind[0])]),
                              np.array([map_face_edge_to_dof(edges_ind[0])])
                              ])
