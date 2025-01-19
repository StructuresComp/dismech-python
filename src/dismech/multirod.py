import dataclasses
import functools

import numpy as np

from . import environment


@dataclasses.dataclass
class Material:
    density: float
    youngs_rod: float
    youngs_shell: float
    poisson_rod: float
    poisson_shell: float


@dataclasses.dataclass
class SimParams:
    static_sim: bool
    two_d_sim: bool
    use_mid_edge: bool
    use_line_search: bool
    log_data: bool
    log_step: int
    show_floor: bool
    dt: float
    max_iter: int
    total_time: float
    plot_step: int
    tol: float
    ftol: float
    dtol: float


@dataclasses.dataclass
class GeomParams:
    rod_r0: float
    shell_h: float

    axs: float = None
    jxs: float = None
    ixs1: float = None
    ixs2: float = None


class MultiRod:

    def __init__(self, geom: GeomParams, material: Material,
                 twist_angles: np.ndarray, nodes: np.ndarray, edges,
                 rod_nodes, shell_nodes,
                 rod_edges, shell_edges, rod_shell_joint_edges,
                 rod_shell_joint_total_edges, face_nodes, sign_faces,
                 face_edges, sim_params: SimParams, environment: environment.Environment):
        # Store important parameters as local vars for symmetry
        self.__r0 = geom.rod_r0
        self.__h = geom.shell_h
        self.__rho = material.density
        self.__nu_shell = material.poisson_shell

        # Node and edge counts
        self.__n_rod_nodes = np.size(rod_nodes, 0)
        self.__n_shell_nodes = np.size(shell_nodes, 0)
        self.__n_nodes = self.__n_rod_nodes + self.__n_shell_nodes
        self.__n_edges_rod_only = np.size(rod_edges, 0)
        self.__n_edges_shell_only = np.size(shell_edges, 0)
        n_edges_rod_shell_joint_total = np.size(rod_shell_joint_total_edges, 0)
        self.__n_edges = np.size(edges, 0)
        self.__n_edges_dof = self.__n_edges_rod_only + n_edges_rod_shell_joint_total
        self.__n_faces = np.size(face_nodes, 0)

        # Store node and edges
        self.__nodes = nodes
        self.__edges = edges
        self.__face_nodes_shell = face_nodes

        # DOF vector
        self.__n_dof = 3 * self.__n_nodes + \
            self.__n_edges_rod_only + n_edges_rod_shell_joint_total
        self.__q0 = np.zeros(self.__n_dof)
        self.__q_nodes = np.reshape(nodes, [nodes.size, 1])
        self.__q0[1:3 * self.__n_nodes] = self.__q_nodes
        self.__q0[3 * self.__n_nodes + 1:3 *
                  self.__n_nodes + self.__n_edges_dof] = twist_angles

        if sim_params.use_mid_edge:
            self.__n_dof += self.__n_edges_shell_only
            self.__q0 = np.concat(
                self.__q0, np.zeros(self.__n_edges_shell_only))

        # References handled by cached properties

        # Mass matrix is a cached property

        # Weight
        self.__fg = self.get_gravity(environment)

        # Stiffnesses
        G_rod = material.youngs_rod / (2 * (1 + material.poisson_rod))

        if geom.axs is not None:
            self.__EA = material.youngs_rod * geom.axs
        else:
            self.__EA = material.youngs_rod * np.pi * self.__r0 ** 2

        if geom.ixs1 is not None and geom.ixs2 is not None:
            self.__EI1 = material.youngs_rod * geom.ixs1
            self.__EI2 = material.youngs_rod * geom.ixs2
        else:
            self.__EI1 = material.youngs_rod * np.pi * self.__r0 ** 4 / 4
            self.__EI2 = self.__EI1

        if geom.jxs is not None:
            self.__gj = G_rod * geom.jxs
        else:
            self.__gj = G_rod * np.pi * self.__r0 ** 4 / 2

        self.__ks = 3 ** (1/2) / 2 * material.youngs_shell * \
            self.__h * self.ref_len
        self.__kb = 2 / (3 ** (1/2)) * material.youngs_shell * \
            (self.__h ** 3) / 12

        if sim_params.use_mid_edge:
            self.__kb = material.youngs_shell * \
                self.__h ** 3 / (24 * 1 - self.__nu_shell ** 2)

            # trial debug
            self.__ks = 2 * material.youngs_shell * self.__h / \
                (1 - self.__nu_shell ** 2) * self.ref_len

        # other properties
        self.__edge_combos = self.construct_possible_edge_combos(
            np.concat(rod_edges, rod_shell_joint_edges))
        self.__u = np.zeros(self.__q0.size)
        self.__a1 = np.zeros(self.__n_edges_dof, 3)
        self.__a2 = np.zeros(self.__n_edges_dof, 3)
        self.__m1 = np.zeros(self.__n_edges_dof, 3)
        self.__m2 = np.zeros(self.__n_edges_dof, 3)

        # Store additional shell face info if using midedge
        if sim_params.use_mid_edge:
            self.__face_edges = face_edges
            self.__sign_faces = sign_faces
            # TODO: init stuff
        else:
            self.__face_edges = np.empty(0)
            self.__sign_faces = np.empty(0)
            self.__init_ts = np.empty(0)
            self.__init_cs = np.empty(0)
            self.__init_fs = np.empty(0)
            self.__init_xis = np.empty(0)

    @functools.cached_property
    def ref_len(self):
        # TODO: Make it a list comprehension
        ret = np.zero(self.__n_edges)
        for i in range(self.__n_edges):
            n1 = self.__edges[i, 1]
            n2 = self.edges[i, 2]
            ret[i] = np.linalg.norm(self.__nodes[n2, :] - self.__nodes[n1, :])
        return ret

    @functools.cached_property
    def voronoi_ref_len(self):
        pass

    @functools.cached_property
    def voronoi_area(self):
        pass

    @functools.cached_property
    def face_area(self):
        pass

    @functools.cached_property
    def mass_matrix(self):
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

    TODO: Figure out what matlab function should be incorporated
    """

    def get_gravity(self, env):
        g_adjusted = env.g
        if "bouyancy" in env.ext_force_list:
            g_adjusted *= (1 - env.rho / self.__rho)

        fg = np.zeros(self.mass_matrix.shape[0])
        for i in range(self.__n_nodes):
            ind = self.map_node_to_dof(i)
            # TODO: add Fg(ind)
        return fg

    @staticmethod
    def map_node_to_dof(node_num: int) -> np.ndarray:
        # FIXME: Column vector??
        return np.array([node_num * 3 - 2, node_num * 3 - 1, node_num * 3])

    def update_tangent(self):
        pass

    def derfun(self):
        pass
