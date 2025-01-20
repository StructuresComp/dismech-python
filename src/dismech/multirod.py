import dataclasses
import functools

import numpy as np

from . import environment, geometry


@dataclasses.dataclass
class GeomParams:
    rod_r0: float
    shell_h: float

    axs: float = None
    jxs: float = None
    ixs1: float = None
    ixs2: float = None


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


class MultiRod:

    def __init__(self, geom: GeomParams, material: Material,
                 geo: geometry.Geometry, sim_params: SimParams,
                 environment: environment.Environment):
        # Store important parameters as local vars for symmetry
        self.__r0 = geom.rod_r0
        self.__h = geom.shell_h
        self.__rho = material.density
        self.__nu_shell = material.poisson_shell

        # Node and edge counts
        self.__n_rod_nodes = np.size(geo.rod_nodes, 0)
        self.__n_shell_nodes = np.size(geo.shell_nodes, 0)
        self.__n_nodes = self.__n_rod_nodes + self.__n_shell_nodes
        self.__n_edges_rod_only = np.size(geo.rod_edges, 0)
        self.__n_edges_shell_only = np.size(geo.shell_edges, 0)
        n_edges_rod_shell_joint_total = np.size(
            geo.rod_shell_joint_edges_total, 0)
        self.__n_edges = np.size(geo.edges, 0)
        self.__n_edges_dof = self.__n_edges_rod_only + n_edges_rod_shell_joint_total
        self.__n_faces = np.size(geo.face_nodes, 0)

        # Store node and edges
        self.__nodes = geo.nodes
        self.__edges = geo.edges
        self.__face_nodes_shell = geo.face_nodes

        # Twist angle can be defined inside __init__
        self.__twist_angles = np.zeros(
            self.__n_edges_rod_only + self.__n_edges_shell_only)

        # DOF vector
        self.__n_dof = 3 * self.__n_nodes + \
            self.__n_edges_rod_only + n_edges_rod_shell_joint_total
        self.__q0 = np.zeros(self.__n_dof)
        self.__q_nodes = geo.nodes.flatten()
        self.__q0[:3 * self.__n_nodes] = self.__q_nodes
        self.__q0[3 * self.__n_nodes:3 *
                  self.__n_nodes + self.__n_edges_dof] = self.__twist_angles

        if sim_params.use_mid_edge:
            self.__n_dof += self.__n_edges_shell_only
            self.__q0 = np.concat(
                self.__q0, np.zeros(self.__n_edges_shell_only))

        # References and mass matrix handled by cached properties

        self.__mass_matrix = self.get_mass_matrix(geom)

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
            np.concat(geo.rod_edges, geo.rod_shell_joint_edges) if geo.rod_shell_joint_edges.size else geo.rod_edges)
        self.__u = np.zeros(self.__q0.size)
        self.__a1 = np.zeros((self.__n_edges_dof, 3))
        self.__a2 = np.zeros((self.__n_edges_dof, 3))
        self.__m1 = np.zeros((self.__n_edges_dof, 3))
        self.__m2 = np.zeros((self.__n_edges_dof, 3))

        # Store additional shell face info if using midedge
        if sim_params.use_mid_edge:
            self.__face_edges = geo.face_edges
            self.__sign_faces = geo.sign_faces
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
        temp = []
        for i in range(self.__n_edges):
            n1, n2 = self.__edges[i]
            temp.append(np.linalg.norm(self.__nodes[n1] - self.__nodes[n2]))
        return np.stack(temp, axis=-1)

    @functools.cached_property
    def voronoi_ref_len(self):
        ret = np.zeros(self.__n_nodes)
        for i in range(self.__n_edges_dof):
            n1, n2 = self.__edges[i]
            ret[n1] += 0.5 * self.ref_len[i]
            ret[n2] += 0.5 * self.ref_len[i]
        return ret

    @functools.cached_property
    def voronoi_area(self):
        if self.__face_nodes_shell.size:
            ret = np.zeros(self.__n_nodes)
            for n1, n2, n3 in range(self.__face_nodes_shell):
                face_A = 0.5 * np.linalg.norm(np.linalg.cross(
                    self.__nodes[n2] - self.__nodes[n1], self.__nodes[n3] - self.__nodes[n2]))
                ret[n1] += face_A / 3
                ret[n2] += face_A / 3
                ret[n3] += face_A / 3
            return ret
        return np.empty(0)

    @functools.cached_property
    def face_area(self):
        temp = []
        for i in range(self.__n_faces):
            n1, n2, n3 = self.__face_nodes_shell[i]
            temp.append(0.5 * np.linalg.norm(np.linalg.cross(
                self.__nodes[n2] - self.__nodes[n1], self.__nodes[n3] - self.__nodes[n2])))
        return np.stack(temp, axis=-1) if len(temp) else np.empty(0)

    @property
    def mass_matrix(self):
        """
        mass matrix associated with initial geom
        """
        return self.__mass_matrix

    def get_mass_matrix(self, geom: GeomParams):
        """
        generate a mass matrix for specified parameters
        """
        m = np.zeros(self.__n_dof)

        # shell faces
        for i in range(self.__n_faces):
            n1, n2, n3 = self.__face_nodes_shell[i]
            face_A = 0.5 * np.linalg.norm(np.linalg.cross(
                self.__nodes[n2] - self.__nodes[n1], self.__nodes[n3] - self.__nodes[n2]))
            mface = self.__rho * face_A * self.__h

            m[self.map_node_to_dof(n1)] += mface / 3 * np.ones(3)
            m[self.map_node_to_dof(n2)] += mface / 3 * np.ones(3)
            m[self.map_node_to_dof(n3)] += mface / 3 * np.ones(3)

        # rod nodes
        for i in range(self.__n_nodes):
            if geom.axs is not None:
                dm = self.voronoi_ref_len[i] * geom.axs * self.__rho
            else:
                dm = self.voronoi_ref_len[i] * \
                    np.pi * self.__r0 ** 2 * self.__rho
            m[self.map_node_to_dof(i)] += dm * np.ones(3)

        # Rod edges
        for i in range(self.__n_edges_dof):
            if geom.axs is not None:
                dm = self.ref_len[i] * geom.axs * self.__rho
            else:
                dm = self.refLen[i] * np.pi * self.__r0 ** 2 * self.__rho
            m[self.map_edge_to_dof(i)] = dm / 2 * self.__r0 ** 2
        return np.diag(m)

    # TODO:
    def init_curvature_midedge(self):
        pass

    @staticmethod
    def init_t_f_c_midedge(p_s, tau0_s, s_s):
        pass

    @staticmethod
    def construct_possible_edge_combos(edges: np.ndarray):
        idx = [np.array([0, 0])]  # jugaad

        for i in range(n_edges := np.size(edges, 0)):
            for j in range(n_edges):
                temp_combo = np.array([i, j])
                if not (edges[i, 0] == edges[j, 0] or edges[i, 0] == edges[j, 1] or edges[i, 1] == edges[j, 0] or edges[i, 1] == edges[j, 1]) and \
                        not any([(n1 == i and n2 == j) or (n1 == j and n2 == i) for (n1, n2) in idx]):
                    idx.append(np.array([i, j]))

        edge_combos = np.zeros((idx_count := (np.size(idx, 1)-1), 4))
        for i in range(np.size(idx_count)):
            edge_combos[i] = np.concat((edges[idx[i][0]], edges[idx[i][1]]))

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
            fg[ind] = np.diag(self.mass_matrix[np.ix_(ind, ind)]) * g_adjusted
        return fg

    @staticmethod
    def map_node_to_dof(node_num: int) -> np.ndarray:
        return np.array([node_num * 3, node_num * 3 + 1, node_num * 3 + 2])
    
    def map_edge_to_dof(self, edge_num: int) -> np.ndarray:
        """
        skip nodes
        """
        return self.__n_nodes * 3 + edge_num

    def update_tangent(self):
        pass

    def derfun(self):
        pass
