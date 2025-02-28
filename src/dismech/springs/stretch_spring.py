import numpy as np


class StretchSpring:

    def __init__(self, length, nodes_ind, robot, EA=0):
        self.EA = EA or robot.EA
        self.ref_len = length
        self.nodes_ind = nodes_ind
        self.ind = np.concat([robot.map_node_to_dof(
            nodes_ind[0]), robot.map_node_to_dof(nodes_ind[1])])
