import numpy as np


class HingeSpring:
    def __init__(self, nodes_ind: np.ndarray, kb: float, robot):
        self.kb = kb
        self.nodes_ind = nodes_ind
        self.ind = np.concatenate([robot.map_node_to_dof(i)
                                  for i in nodes_ind], axis=0)
