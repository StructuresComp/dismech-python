import numpy as np

class ContactPair:

    def __init__(self, nodes_ind: np.ndarray, map_node_to_dof, mu = 0):
        self.ind = map_node_to_dof(nodes_ind).reshape(-1) # (12,)
        self.mu = 0