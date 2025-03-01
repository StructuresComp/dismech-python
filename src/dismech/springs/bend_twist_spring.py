import numpy as np


class BendTwistSpring:
    def __init__(self,
                 nodes_edges_index: np.ndarray,
                 signs: np.ndarray,
                 kappa_bar: np.ndarray,
                 ref_twist: float,
                 robot,
                 EI: np.ndarray = None,
                 GJ: float = None):
        self.stiff_EI = EI or [robot.EI1, robot.EI2]
        self.stiff_GJ = GJ or robot.GJ

        # N e N e N
        self.nodes_ind = [int(nodes_edges_index[0]), int(
            nodes_edges_index[2]), int(nodes_edges_index[4])]
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
        self.voronoi_len = 0.5 * \
            (robot.ref_len[self.edges_ind[0]] +
             robot.ref_len[self.edges_ind[1]])

        self.kappa_bar = kappa_bar
        self.ref_twist = ref_twist
        self.ref_twist_init = ref_twist
