import numpy as np

class ContactPair:

    def __init__(self, nodes_ind: np.ndarray, map_node_to_dof):
        self.ind = map_node_to_dof(nodes_ind) # (12,)