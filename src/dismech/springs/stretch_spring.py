import typing
import numpy as np


class StretchSpring:

    def __init__(self,
                 nodes_ind: np.ndarray,
                 ref_len: float,
                 EA: float,
                 map_node_to_dof: typing.Callable[[np.ndarray], np.ndarray]):
        self.EA = EA
        self.ref_len = ref_len
        self.nodes_ind = nodes_ind
        self.ind = np.concat([map_node_to_dof(nodes_ind[0]),
                              map_node_to_dof(nodes_ind[1])])
