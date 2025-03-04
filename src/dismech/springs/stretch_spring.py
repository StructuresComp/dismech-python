import numpy as np


class StretchSpring:

    def __init__(self, nodes_ind: np.ndarray, ref_len: float, EA: float, robot):
        self.EA = EA
        self.ref_len = ref_len
        self.nodes_ind = nodes_ind
        self.ind = np.concat([robot.map_node_to_dof(
            nodes_ind[0]), robot.map_node_to_dof(nodes_ind[1])])
