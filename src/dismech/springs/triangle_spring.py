import typing
import numpy as np


class TriangleSpring:

    def __init__(self,
                 nodes_ind: np.ndarray,
                 edges_ind: np.ndarray,
                 face_edges: np.ndarray,
                 signs: np.ndarray,
                 ref_len: np.ndarray,
                 kb: float,
                 map_node_to_dof: typing.Callable[[np.ndarray], np.ndarray],
                 map_face_edge_to_dof: typing.Callable[[np.ndarray], np.ndarray]):
        self.ref_len = ref_len
        self.kb = kb
        self.nodes_ind = nodes_ind
        self.edges_ind = edges_ind
        self.sgn = signs
        self.face_edges = face_edges

        self.ind = np.concat([map_node_to_dof(nodes_ind[0]),
                              map_node_to_dof(nodes_ind[1]),
                              map_node_to_dof(nodes_ind[2]),
                              np.ndarray([map_face_edge_to_dof(edges_ind[0])]),
                              np.ndarray([map_face_edge_to_dof(edges_ind[1])]),
                              np.ndarray([map_face_edge_to_dof(edges_ind[2])])
                              ])
