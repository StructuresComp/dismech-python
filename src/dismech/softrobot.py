import copy
import dataclasses
import typing

import numpy as np

from . import environment, geometry, bendingtwistingspring, stretch_spring


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


_TANGENT_THRESHOLD = 1e-10


class SoftRobot:

    def __init__(self, geom: GeomParams, material: Material,
                 geo: geometry.Geometry, sim_params: SimParams,
                 environment: environment.Environment):
        self.__sim_params = sim_params

        # Store important parameters as local vars for symmetry
        self.__r0 = geom.rod_r0
        self.__h = geom.shell_h
        self.__rho = material.density
        self.__nu_shell = material.poisson_shell

        # Node and edge counts
        self.__n_nodes = np.size(geo.nodes, 0)
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

        # Twist angles from geometry
        self.__twist_angles = geo.twist_angles

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

        self.__q = self.__q0

        # References and mass matrix
        self.__ref_len = self.__get_ref_len()
        self.__voronoi_ref_len = self.__get_voronoi_ref_len()
        self.__voronoi_area = self.__get_voronoi_area()
        self.__faec_area = self.__get_face_area()

        self.__mass_matrix = self.__get_mass_matrix(geom)

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
            self.__GJ = G_rod * geom.jxs
        else:
            self.__GJ = G_rod * np.pi * self.__r0 ** 4 / 2

        self.__ks = 3 ** (1/2) / 2 * material.youngs_shell * \
            self.__h * self.__ref_len
        self.__kb = 2 / (3 ** (1/2)) * material.youngs_shell * \
            (self.__h ** 3) / 12

        if sim_params.use_mid_edge:
            self.__kb = material.youngs_shell * \
                self.__h ** 3 / (24 * 1 - self.__nu_shell ** 2)

            # trial debug
            self.__ks = 2 * material.youngs_shell * self.__h / \
                (1 - self.__nu_shell ** 2) * self.__ref_len

        # other properties
        # FIXME: Calculate only if self contact force is used
        self.__edge_combos = self.__construct_possible_edge_combos(
            np.concat((geo.rod_edges, geo.rod_shell_joint_edges)) if geo.rod_shell_joint_edges.size else geo.rod_edges)
        self.__u = np.zeros(self.__q0.size)
        self.__a1 = np.zeros((self.__n_edges_dof, 3))
        self.__a2 = np.zeros((self.__n_edges_dof, 3))
        self.__m1 = np.zeros((self.__n_edges_dof, 3))
        self.__m2 = np.zeros((self.__n_edges_dof, 3))

        # Store additional shell face info if using midedge
        if sim_params.use_mid_edge:
            self.__face_edges = geo.face_edges
            self.__sign_faces = geo.sign_faces
            self.__init_ts, self.__init_cs, self.__init_fs, self.__init_xis = self.__init_curvature_midedge()
        else:
            self.__face_edges = np.empty(0)
            self.__sign_faces = np.empty(0)
            self.__init_ts = np.empty(0)
            self.__init_cs = np.empty(0)
            self.__init_fs = np.empty(0)
            self.__init_xis = np.empty(0)

        # Springs
        n_rod_springs = np.size(geo.rod_stretch_springs, 0)
        self.__stretch_springs = [stretch_spring.StretchSpring(
            self.ref_len[i], spring, self) for i, spring in enumerate(geo.rod_stretch_springs)]
        self.__stretch_springs += [stretch_spring.StretchSpring(
            self.ref_len[i + n_rod_springs], spring, self, self.__ks[i + n_rod_springs]) for i, spring in enumerate(geo.shell_stretch_springs)]

        self.__bend_twist_springs = [
            bendingtwistingspring.BendingTwistingSpring(spring, sign, np.array([0, 0]), 0, self) for spring, sign in zip(geo.bend_twist_springs, geo.bend_twist_signs)]

    def __get_ref_len(self):
        temp = []
        for i in range(self.__n_edges):
            n1, n2 = self.__edges[i]
            temp.append(np.linalg.norm(self.__nodes[n1] - self.__nodes[n2]))
        return np.stack(temp, axis=-1)

    def __get_voronoi_ref_len(self):
        ret = np.zeros(self.__n_nodes)
        for i in range(self.__n_edges_dof):
            n1, n2 = self.__edges[i]
            ret[n1] += 0.5 * self.__ref_len[i]
            ret[n2] += 0.5 * self.__ref_len[i]
        return ret

    def __get_voronoi_area(self):
        if self.__face_nodes_shell.size:
            ret = np.zeros(self.__n_nodes)
            for n1, n2, n3 in self.__face_nodes_shell:
                face_A = 0.5 * np.linalg.norm(np.linalg.cross(
                    self.__nodes[n2] - self.__nodes[n1], self.__nodes[n3] - self.__nodes[n2]))
                ret[n1] += face_A / 3
                ret[n2] += face_A / 3
                ret[n3] += face_A / 3
            return ret
        return np.empty(0)

    def __get_face_area(self):
        temp = []
        for i in range(self.__n_faces):
            n1, n2, n3 = self.__face_nodes_shell[i]
            temp.append(0.5 * np.linalg.norm(np.linalg.cross(
                self.__nodes[n2] - self.__nodes[n1], self.__nodes[n3] - self.__nodes[n2])))
        return np.stack(temp, axis=-1) if len(temp) else np.empty(0)

    def __get_mass_matrix(self, geom: GeomParams):
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
                dm = self.__voronoi_ref_len[i] * geom.axs * self.__rho
            else:
                dm = self.__voronoi_ref_len[i] * \
                    np.pi * self.__r0 ** 2 * self.__rho
            m[self.map_node_to_dof(i)] += dm * np.ones(3)

        # Rod edges
        for i in range(self.__n_edges_dof):
            if geom.axs is not None:
                dm = self.__ref_len[i] * geom.axs * self.__rho
            else:
                dm = self.__ref_len[i] * np.pi * self.__r0 ** 2 * self.__rho
            m[self.map_edge_to_dof(i)] = dm / 2 * self.__r0 ** 2
        return np.diag(m)

    def __init_curvature_midedge(self):
        ts = np.zeros((3, 3, self.__n_faces))
        fs = np.zeros((3, self.__n_faces))
        cs = np.zeros((3, self.__n_faces))
        xis = np.zeros((3, self.__n_faces))

        tau_0 = self.update_pre_comp_shell(self.__q)

        for i in range(self.__n_faces):
            face_i_nodes = self.__face_nodes_shell[i]
            face_i_edges = self.__face_edges[i]

            p_is = np.zeros((3, 3))
            xi_is = np.zeros((3, 3))
            tau_0_is = np.zeros((3, 3))

            for j in range(3):
                p_is[: j] = self.__q[3 * face_i_nodes[j] - 3: 3 * face_i_nodes[j]]
                xi_is[j] = self.__q[3 * self.__n_nodes + face_i_edges[j]]
                tau_0_is[:, j] = tau_0[:, face_i_edges[j]]

            s_is = self.__sign_faces[i]

            xis[:, i] = xi_is

            t, f, c = self.__init_t_f_c_midedge(p_is, tau_0_is, s_is)
            ts[:, :, i] = t
            fs[:, i] = f.T
            cs[:, i] = c.T

        return ts, fs, cs, xis

    @staticmethod
    def __init_t_f_c_midedge(p_s: np.ndarray, tau0_s: np.ndarray, s_s: np.ndarray):
        pi, pj, pk = p_s

        tau0_i, tau0_j, tau0_k = s_s * tau0_s

        # edges
        vi = pk - pj
        vj = pi - pk
        vk = pj - pi

        li = np.linalg.norm(vi)
        lj = np.linalg.norm(vj)
        lk = np.linalg.norm(vk)

        # triangle face normal
        normal = np.linalg.cross(vk, vi)
        A = np.linalg.norm(normal) / 2
        unit_norm = normal / np.linalg.norm(normal)

        # t_i
        t_i = np.linalg.norm(vi, unit_norm)
        t_j = np.linalg.norm(vj, unit_norm)
        t_k = np.linalg.norm(vk, unit_norm)

        # c_i
        c_i = 1 / (A * li * np.dot((t_i / np.linalg.norm(t_i)), tau0_i))
        c_j = 1 / (A * lj * np.dot((t_j / np.linalg.norm(t_j)), tau0_j))
        c_k = 1 / (A * lk * np.dot((t_k / np.linalg.norm(t_k)), tau0_k))

        # f_i
        f_i = np.dot(unit_norm, tau0_i)
        f_j = np.dot(unit_norm, tau0_j)
        f_k = np.dot(unit_norm, tau0_k)

        return np.concat((t_i, t_j, t_k)), np.concat((f_i, f_j, f_k)), np.concat((c_i, c_j, c_k))

    @staticmethod
    def __construct_possible_edge_combos(edges: np.ndarray):
        idx = [np.array([0, 0])]  # jugaad

        for i in range(n_edges := np.size(edges, 0)):
            for j in range(n_edges):
                if not (edges[i, 0] == edges[j, 0] or edges[i, 0] == edges[j, 1] or edges[i, 1] == edges[j, 0] or edges[i, 1] == edges[j, 1]) and \
                        not any([(n1 == i and n2 == j) or (n1 == j and n2 == i) for (n1, n2) in idx]):
                    idx.append(np.array([i, j]))

        edge_combos = np.zeros((idx_count := (np.size(idx, 1)-1), 4))
        for i in range(np.size(idx_count)):
            edge_combos[i] = np.concat((edges[idx[i][0]], edges[idx[i][1]]))

    @staticmethod
    def __parallel_transport(u, t1, t2) -> np.ndarray:
        b = np.cross(t1, t2)
        if (b_norm := np.linalg.norm(b)) == 0:
            return u

        # for numerical stability
        b /= b_norm
        b -= np.dot(b, t1) * t1
        b /= np.linalg.norm(b)
        b -= np.dot(b, t2) * t2
        b /= np.linalg.norm(b)

        n1 = np.cross(t1, b)
        n2 = np.cross(t2, b)
        return np.dot(u, t1) * t2 + np.dot(u, n1) * n2 + np.dot(u, b) * b

    def __set_kappa(self):
        for spring in self.__bend_twist_springs:
            n0, n1, n2 = spring.nodes_ind
            e0, e1 = spring.edges_ind

            n0p = self.q[self.map_node_to_dof(n0)]
            n1p = self.q[self.map_node_to_dof(n1)]
            n2p = self.q[self.map_node_to_dof(n2)]

            m1e = self.__m1[e0]
            m2e = spring.sgn[0] * self.__m2[e0]
            m1f = self.__m1[e1]
            m2f = spring.sgn[1] * self.__m2[e1]

            spring.kappa_bar = self.compute_kappa(
                n0p, n1p, n2p, m1e, m2e, m1f, m2f)

    """
    Public Interface
    """

    def initialize(self, fixed_nodes) -> "SoftRobot":
        # reference frame + material frame
        ret = self.compute_space_parallel()
        ret.__m1, ret.__m2 = ret.compute_material_directors(
            ret.a1, ret.a2, ret.get_theta(ret.q0))

        # natural curvature
        ret.__set_kappa()

        # reference twist
        ret.__undef_ref_twist = ret.compute_reference_twist(
            ret.bend_twist_springs, ret.a1, ret.tangent, np.zeros(len(self.__bend_twist_springs)))
        ret.__ref_twist = ret.compute_reference_twist(
            ret.bend_twist_springs, ret.a1, ret.tangent,  ret.__undef_ref_twist)

        # boundary conditions
        ret.__fixed_nodes = fixed_nodes
        fixed_edge_indices = []
        for i, edge in enumerate(ret.__edges):
            if edge[0] in fixed_nodes and edge[1] in fixed_nodes:
                fixed_edge_indices.append(i)
        if ret.sim_params.two_d_sim:
            fixed_edge_indices += range(ret.__n_edges_dof)
        ret.__fixed_edges = np.array(fixed_edge_indices)
        ret.__fixed_dof, ret.__free_dof = ret.find_fixed_free_dof(
            ret.__fixed_nodes, ret.__fixed_edges)

        return ret

    def find_fixed_free_dof(self, fixed_nodes, fixed_edges):
        # Initialize fixed DOF arrays
        fixedDOF_nodes = np.zeros((3, len(fixed_nodes)), dtype=int)
        fixedDOF_edges = np.zeros(len(fixed_edges), dtype=int)

        # Map fixed nodes to DOFs
        for i in range(len(fixed_nodes)):
            fixedDOF_nodes[:, i] = self.map_node_to_dof(fixed_nodes[i])

        # Flatten fixedDOF_nodes into a 1D array
        fixedDOF_nodes_vec = fixedDOF_nodes.reshape(-1)

        # Map fixed edges to DOFs
        for i in range(len(fixed_edges)):
            fixedDOF_edges[i] = self.map_edge_to_dof(
                fixed_edges[i])

        # Combine fixed DOFs from nodes and edges
        fixedDOF = np.concatenate((fixedDOF_nodes_vec, fixedDOF_edges))

        # Find free DOFs
        dummy = np.ones(self.n_dof, dtype=int)
        dummy[fixedDOF] = 0
        freeDOF = np.where(dummy == 1)[0]

        return fixedDOF, freeDOF

    @staticmethod
    def map_node_to_dof(node_num: int) -> np.ndarray:
        return np.array([node_num * 3, node_num * 3 + 1, node_num * 3 + 2])

    def map_edge_to_dof(self, edge_num: int) -> np.ndarray:
        return self.__n_nodes * 3 + edge_num

    def get_theta(self, q):
        return q[3 * self.__n_nodes:3 * self.__n_nodes + self.__n_edges_dof + 1]

    @staticmethod
    def rotate_axis_angle(v, z, theta):
        if theta == 0:
            return v
        return np.cos(theta) * v + np.sin(theta) * np.cross(z, v) + np.dot(z, v) * (1 - np.cos(theta)) * z

    @staticmethod
    def signed_angle(u, v, n):
        w = np.cross(u, v)
        angle = np.atan2(np.linalg.norm(w), np.dot(u, v))
        if (np.dot(n, w) < 0):
            angle = -angle
        return angle

    @staticmethod
    def compute_material_directors(a1, a2, theta):
        n_edges_dof = np.size(theta)

        m1 = np.zeros((n_edges_dof, 3))
        m2 = np.zeros((n_edges_dof, 3))

        for i in range(n_edges_dof):
            m1[i] = np.cos(theta[i]) * a1[i] + np.sin(theta[i]) * a2[i]
            m2[i] = -np.sin(theta[i]) * a1[i] + np.cos(theta[i]) * a2[i]

        return m1, m2

    def compute_time_parallel(self, a1_old, q0, q) -> typing.Tuple[np.ndarray, np.ndarray]:
        # Should we change the interface?
        tangent0 = self.compute_tangent(q0)
        tangent = self.compute_tangent(q)

        a1 = np.zeros((self.__n_edges_dof, 3))
        a2 = np.zeros((self.__n_edges_dof, 3))

        for i in range(self.__n_edges_dof):
            t0 = tangent0[i]
            t = tangent[i]

            a1_local = self.__parallel_transport(a1_old[i], t0, t)

            a1_local -= np.dot(a1_local, t) * t
            a1_local /= np.linalg.norm(a1_local)

            a1[i] = a1_local
            a2[i] = np.cross(t, a1_local)

        return a1, a2

    def compute_space_parallel(self) -> "SoftRobot":
        """
        Return reference frame
        """
        ret = copy.deepcopy(self)   # preserve original

        ret.__tangent = ret.compute_tangent(ret.__q0)

        # Set initial entry
        t0 = ret.__tangent[1]
        t1 = np.array([0, 1, 0])
        a1Tmp = np.cross(t0, t1)

        if (abs(a1Tmp) < 1e-6).all():  # ==0
            t1 = np.array([0, 0, -1])
            a1Tmp = np.cross(t0, t1)

        ret.__a1[0] = a1Tmp / np.linalg.norm(a1Tmp)
        ret.__a2[0] = np.cross(ret.__tangent[0], ret.__a1[0])

        # Space parallel transport to construct the reference frame
        for i in range(1, ret.__n_edges_dof):
            t0 = ret.__tangent[i-1]
            t1 = ret.__tangent[i]
            a1_0 = ret.__a1[i-1]
            a1_1 = ret.__parallel_transport(a1_0, t0, t1)
            ret.__a1[i] = a1_1 / np.linalg.norm(a1_1)
            ret.__a2[i] = np.cross(t1, ret.__a1[i])

        return ret

    def compute_tangent(self, q: np.ndarray) -> np.ndarray:
        tangent = np.zeros((self.__n_edges_dof, 3))

        for i in range(self.__n_edges_dof):
            n0, n1 = self.__edges[i]
            n0_pos = q[self.map_node_to_dof(n0)]
            n1_pos = q[self.map_node_to_dof(n1)]
            de = (n1_pos - n0_pos).T
            tangent_vec = de / np.linalg.norm(de)
            # Remove small non-zero terms
            tangent_vec[np.abs(tangent_vec) < _TANGENT_THRESHOLD] = 0

            tangent[i] = tangent_vec

        return tangent

    @staticmethod
    def compute_kappa(n0, n1, n2, m1e, m2e, m1f, m2f):
        t0 = n1 - n0 / np.linalg.norm(n1 - n0)
        t1 = n2 - n1 / np.linalg.norm(n2 - n1)
        kb = 2.0 * np.cross(t0, t1) / (1.0 + np.dot(t0, t1))

        kappa1 = 0.5 * np.dot(kb, m2e + m2f)
        kappa2 = -0.5 * np.dot(kb, m1e + m1f)

        return np.stack((kappa1, kappa2), axis=0)

    def compute_reference_twist(self, bend_twist_springs, a1, tangent, ref_twist):
        n_twist = np.size(ref_twist)

        for i in range(n_twist):

            e0, e1 = bend_twist_springs[i].edges_ind
            u0 = a1[e0]
            u1 = a1[e1]

            t0 = bend_twist_springs[i].sgn[0] * tangent[e0]
            t1 = bend_twist_springs[i].sgn[1] * tangent[e1]

        ut = self.__parallel_transport(u0, t0, t1)
        ut = self.rotate_axis_angle(ut, t1, ref_twist[i])

        ref_twist[i] += self.signed_angle(ut, u1, t1)

        return ref_twist

    # UNTESTED: Need shell model
    def update_pre_comp_shell(self, q):
        edge_common_to = np.zeros(self.__n_edges)
        n_avg = np.zeros((3, self.__n_edges))
        tau_0 = np.zeros((3, self.__n_edges))
        e = np.zeros((3, self.__n_edges))

        for i in range(self.__n_faces):
            n1, n2, n3 = self.__face_nodes_shell[i]
            n1p = self.__q[3 * n1 - 3:3 * n1]
            n2p = self.__q[3 * n2 - 3:3 * n2]
            n3p = self.__q[3 * n3 - 3:3 * n3]

            # face normal
            face_normal = np.linalg.cross(n2p - n1p, n3p - n1p)
            face_unit_normal = face_normal * 1 / np.linalg.norm(face_normal)

            # face edge map
            for edge in self.__face_edges[i]:
                edge_common_to[edge] = edge_common_to[edge] + 1

                n_avg[:, edge] += face_unit_normal
                n_avg[:, edge] /= np.linalg.norm(n_avg[:, edge])

                assert (edge_common_to[edge] < 3)

        for i in range(self.__n_edges):
            e[:, i] = q[3 * self.__edges[i, 1] - 3: 3 * self.__edges[i, 1]
                        ] - self.__q[3 * self.__edges[i, 0] - 3: 3 * self.__edges[i, 0]]
            tau_0 = np.linalg.cross(e[:, i], n_avg[:, i])
            tau_0 /= np.linalg.norm(tau_0)

        return tau_0

    @property
    def ref_len(self):
        return self.__ref_len

    @property
    def q(self):
        return self.__q

    @property
    def q0(self):
        return self.__q0

    @property
    def a1(self):
        return self.__a1

    @property
    def a2(self):
        return self.__a2

    @property
    def bend_twist_springs(self):
        return self.__bend_twist_springs

    @property
    def stretch_springs(self):
        return self.__stretch_springs

    @property
    def EA(self):
        return self.__EA

    @property
    def EI1(self):
        return self.__EI1

    @property
    def EI2(self):
        return self.__EI2

    @property
    def GJ(self):
        return self.__GJ

    @property
    def n_dof(self):
        return self.__n_dof

    @property
    def sim_params(self):
        return self.__sim_params

    # Post initialize variables
    # TODO: Make it harder to use improperly

    @property
    def tangent(self):
        return self.__tangent

    @property
    def ref_twist(self):
        return self.__ref_twist
