import numpy as np

class BendTwistSpring:
    def __init__(self, nodes_edges_index: np.ndarray, signs: np.ndarray, kappa_bar: np.ndarray, ref_twist: float, robot, optional_stiffnesses_EI: np.ndarray = None, optional_stiffnesses_GJ: float = None):
        """
        Constructor to initialize a BendTwistSpring object.

        :param nodes_edges_index: Indices of nodes and edges involved in bending/twisting.
        :param signs: Signs associated with the bending/twisting.
        :param kappa_bar: Reference curvature.
        :param ref_twist: Reference twist.
        :param robot: Object containing material properties.
        :param optional_stiffnesses_EI: Optional bending stiffness values [EI1, EI2].
        :param optional_stiffnesses_GJ: Optional torsional stiffness value.
        """
        if optional_stiffnesses_EI is not None and optional_stiffnesses_GJ is not None:
            self.stiff_EI = [optional_stiffnesses_EI[0], optional_stiffnesses_EI[1]]
            self.stiff_GJ = optional_stiffnesses_GJ
        else:
            self.stiff_EI = [robot.EI1, robot.EI2]
            self.stiff_GJ = robot.GJ
        
        self.nodes_ind = [int(nodes_edges_index[0]), int(nodes_edges_index[2]), int(nodes_edges_index[4])]
        self.edges_ind = [int(nodes_edges_index[1]), int(nodes_edges_index[3])]
        self.sgn = signs
        
        # Set DOF indices for nodes and edges
        self.ind = np.concatenate((
            robot.map_node_to_dof(self.nodes_ind[0]),
            robot.map_node_to_dof(self.nodes_ind[1]),
            robot.map_node_to_dof(self.nodes_ind[2]),
            np.array([robot.map_edge_to_dof(self.edges_ind[0])]),
            np.array([robot.map_edge_to_dof(self.edges_ind[1])])
        ), axis=0)
        
        # Compute Voronoi length
        self.voronoi_len = 0.5 * (robot.ref_len[self.edges_ind[0]] + robot.ref_len[self.edges_ind[1]])
        
        self.kappa_bar = kappa_bar
        self.ref_twist = ref_twist
        self.ref_twist_init = ref_twist
        
        # Initialize force and Jacobian matrices
        self.dFb = np.zeros(11)
        self.dJb = np.zeros((11, 11))
        self.dFt = np.zeros(11)
        self.dJt = np.zeros((11, 11))
