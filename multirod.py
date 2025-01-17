import dataclasses

import numpy as np


@dataclasses.dataclass
class Geometery:
    rod_r0: float
    shell_h: float


@dataclasses.dataclass
class Material:
    density: float
    youngs_rod: float
    youngs_shell: float
    poisson_rod: float
    poisson_shell: float


class MultiRod:

    def __init__(self, geom: Geometery, material: Material, twist_angles: np.ndarray,
                 nodes: np.ndarray, edges, rod_nodes, shell_nodes,
                 rod_edges, shell_edges, rod_shell_joint_edges,
                 rod_shell_joint_total_edges, face_nodes, sign_faces,
                 face_edges, sim_params, environment):
        self.__r0 = geom.rod_r0
        self.__h = geom.shell_h
        self.__rho = material.density
        self.__Y_rod = material.youngs_rod
        self.__Y_shell = material.youngs_shell
        self.__nu_rod = material.poisson_rod
        self.__nu_shell = material.poisson_shell

        self.__n_rod_nodes = np.size(rod_nodes, 0)
        self.__n_shell_nodes = np.size(shell_nodes, 0)
        self.__n_nodes = self.__n_rod_nodes + self.__n_shell_nodes
        self.__n_edges_rod_only = np.size(rod_edges, 0)
        self.__n_edges_shell_only = np.size(shell_edges, 0)
        n_edges_rod_shell_joint_total = np.size(rod_shell_joint_total_edges, 0)
        self.__n_edges = np.size(edges, 0)
        self.__n_edges_dof = self.__n_edges_rod_only + n_edges_rod_shell_joint_total
        self.__n_faces = np.size(face_nodes, 0)

        self.__nodes = nodes
        self.__edges = edges
        self.__face_nodes_shell = face_nodes

        self.__n_dof = 3 * self.__n_nodes + \
            self.__n_edges_rod_only + n_edges_rod_shell_joint_total
        self.__q0 = np.zeros(self.__n_dof)
        self.__q_nodes = np.reshape(nodes, [nodes.size, 1])
        self.__q0[1:3 * self.__n_nodes] = self.__q_nodes
        self.__q0[3 * self.__n_nodes + 1:3 *
                  self.__n_nodes + self.__n_edges_dof] = twist_angles
        
        # still more

    @property
    def ref_len(self):
        pass

    @property
    def voronoi_ref_len(self):
        pass

    @property
    def voronoi_area(self):
        pass

    @property
    def face_area(self):
        pass

    def get_mass_matrix(self, geom):
        pass

    def init_curvature_midedge(self):
        pass

    @staticmethod
    def init_t_f_c_midedge(p_s, tau0_s, s_s):
        pass

    @staticmethod
    def construct_possible_edge_combos(edges):
        pass

    """
    Implied functions
    """

    def update_tangent(self):
        pass

    def derfun(self):
        pass
