import numpy as np



class StretchSpring:

    def __init__(self, nodes_ind: np.ndarray, length: float, robot, EA=0):
        self.EA = EA or robot.EA
        self.ref_len = length
        self.nodes_ind = nodes_ind
        self.ind = np.concat([robot.map_node_to_dof(
            nodes_ind[0]), robot.map_node_to_dof(nodes_ind[1])])
