import numpy as np

class StretchingSpring:
    def __init__(self, undef_len: float, nodes_index: np.ndarray, SoftRobot, optional_stiffness: float = None):
        """
        Constructor to initialize a StretchingSpring object.

        :param undef_len: Reference (undeformed) length of the spring.
        :param nodes_index: Indices of the nodes connected by the spring.
        :param SoftRobot: Object containing material properties.
        :param optional_stiffness: Optional stiffness value; if not provided, uses SoftRobot.__EA.
        """
        self.stiff = optional_stiffness if optional_stiffness is not None else SoftRobot.__EA
        self.ref_len = undef_len
        self.nodes_ind = nodes_index
        self.ind = np.concatenate((SoftRobot.map_node_to_dof(self.nodes_ind[0]), 
                                   SoftRobot.map_node_to_dof(self.nodes_ind[1])), axis=0)
        
        # Initialize dF and dJ
        self.dF = np.zeros(6)
        self.dJ = np.zeros((6, 6))
