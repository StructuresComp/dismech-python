import copy
import dataclasses
import typing

import numpy as np

from .state import RobotState
from .stiffness import compute_rod_stiffness, compute_shell_stiffness
from .frame_util import compute_reference_twist, parallel_transport
from .environment import Environment
from .geometry import Geometry
from .params import GeomParams, Material, SimParams
from .springs import BendTwistSpring, StretchSpring, HingeSpring


_TANGENT_THRESHOLD = 1e-10


class SoftRobot:
    def __init__(self, geom: GeomParams, material: Material,
                 geo: Geometry, sim_params: SimParams,
                 env: Environment):
        # store mutable
        self.__sim_params = sim_params
        self.__env = env

        self._init_geometry(geo)
        self._init_stiffness(geom, material)
        self._init_state(geo)
        self._init_fixed_dof()
        self._init_directors_and_springs(geo)
        self.__mass_matrix = self._get_mass_matrix(geom, material)

        # self.__edge_combos = self._construct_edge_combinations(
        #    np.concatenate((geo.rod_edges, geo.rod_shell_joint_edges)
        #                   ) if geo.rod_shell_joint_edges.size else geo.rod_edges
        # )

    def _init_geometry(self, geo: Geometry):
        """Initialize geometry properties"""
        self.__n_nodes = geo.nodes.shape[0]
        self.__n_edges_rod_only = geo.rod_edges.shape[0]
        self.__n_edges_shell_only = geo.shell_edges.shape[0]
        n_edges_joint = geo.rod_shell_joint_edges_total.shape[0]
        self.__n_edges = geo.edges.shape[0]
        self.__n_edges_dof = self.__n_edges_rod_only + n_edges_joint
        self.__n_faces = geo.face_nodes.shape[0] if geo.face_nodes.size else 0

        self.__nodes = geo.nodes
        self.__edges = geo.edges
        self.__face_nodes_shell = geo.face_nodes
        self.__twist_angles = geo.twist_angles

        # Initialize DOF vector
        self.__n_dof = 3 * self.__n_nodes + self.__n_edges_dof
        self.__q0 = np.zeros(self.__n_dof)
        self.__q0[:3 * self.__n_nodes] = self.__nodes.flatten()
        self.__q0[3 * self.__n_nodes:3 * self.__n_nodes +
                  self.__n_edges_dof] = self.__twist_angles

        # Midedge bending has more DOF
        if self.__sim_params.use_mid_edge:
            self.__n_dof += self.__n_edges_shell_only
            self.__q0 = np.concatenate(
                (self.__q0, np.zeros(self.__n_edges_shell_only)))

        self.__ref_len = self._get_ref_len()
        self.__voronoi_ref_len = self._get_voronoi_ref_len()
        self.__voronoi_area = self._get_voronoi_area()
        self.__face_area = self._get_face_area()

    def _get_ref_len(self) -> np.ndarray:
        vectors = self.__nodes[self.__edges[:, 1]] - \
            self.__nodes[self.__edges[:, 0]]
        return np.linalg.norm(vectors, axis=1)

    def _get_voronoi_ref_len(self) -> np.ndarray:
        edges = self.__edges[:self.__n_edges_dof]
        n_nodes = self.__n_nodes
        weights = 0.5 * self.__ref_len[:self.__n_edges_dof]
        contributions = np.zeros(n_nodes)
        np.add.at(contributions, edges[:, 0], weights)
        np.add.at(contributions, edges[:, 1], weights)
        return contributions

    def _get_voronoi_area(self) -> np.ndarray:
        if not self.__face_nodes_shell.size:
            return np.empty(0)
        faces = self.__face_nodes_shell
        v1 = self.__nodes[faces[:, 1]] - self.__nodes[faces[:, 0]]
        v2 = self.__nodes[faces[:, 2]] - self.__nodes[faces[:, 1]]
        cross = np.cross(v1, v2)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        node_areas = np.bincount(faces.ravel(), weights=np.repeat(
            areas / 3, 3), minlength=self.__n_nodes)
        return node_areas

    def _get_face_area(self) -> np.ndarray:
        if not self.__n_faces:
            return np.empty(0)
        faces = self.__face_nodes_shell
        v1 = self.__nodes[faces[:, 1]] - self.__nodes[faces[:, 0]]
        v2 = self.__nodes[faces[:, 2]] - self.__nodes[faces[:, 1]]
        cross = np.cross(v1, v2)
        return 0.5 * np.linalg.norm(cross, axis=1)

    def _init_stiffness(self, geom: GeomParams, material: Material):
        """Initialize global stiffness properties"""
        self.__EA, self.__EI1, self.__EI2, self.__GJ = compute_rod_stiffness(
            geom, material)
        self.__ks, self.__kb = compute_shell_stiffness(
            geom, material, self.ref_len, self.sim_params.use_mid_edge)

    def _get_mass_matrix(self, geom: GeomParams, material: Material) -> np.ndarray:
        mass = np.zeros(self.__n_dof)

        # Shell face contributions
        if self.__n_faces:
            faces = self.__face_nodes_shell
            v1 = self.__nodes[faces[:, 1]] - self.__nodes[faces[:, 0]]
            v2 = self.__nodes[faces[:, 2]] - self.__nodes[faces[:, 1]]
            areas = 0.5 * np.linalg.norm(np.cross(v1, v2), axis=1)
            m_shell = material.density * areas * geom.shell_h
            dof_indices = (3 * faces[:, :, None] + np.arange(3)).reshape(-1)
            np.add.at(mass, dof_indices, np.repeat(m_shell / 3, 9))

        # Node contributions
        if self.__n_nodes:
            if geom.axs is not None:
                dm_nodes = self.__voronoi_ref_len * geom.axs * material.density
            else:
                dm_nodes = self.__voronoi_ref_len * \
                    np.pi * (geom.rod_r0 ** 2) * material.density
            node_dofs = np.arange(3 * self.__n_nodes).reshape(-1, 3)
            mass[node_dofs] += dm_nodes[:, None]

        # Edge contributions
        if self.__n_edges_dof:
            if geom.axs is not None:
                dm_edges = self.__ref_len[:self.__n_edges_dof] * \
                    geom.axs * material.density
            else:
                dm_edges = self.__ref_len[:self.__n_edges_dof] * \
                    np.pi * (geom.rod_r0 ** 2) * material.density
            edge_mass = dm_edges / 2 * (geom.rod_r0 ** 2)
            edge_dofs = 3 * self.__n_nodes + np.arange(self.__n_edges_dof)
            mass[edge_dofs] = edge_mass

        return np.diag(mass)

    @staticmethod
    def map_node_to_dof(node_nums: typing.Union[int, np.ndarray]) -> np.ndarray:
        return (3 * np.asarray(node_nums))[..., None] + np.array([0, 1, 2])

    def map_edge_to_dof(self, edge_nums: typing.Union[int, np.ndarray]) -> np.ndarray:
        return 3 * self.__n_nodes + np.asarray(edge_nums)

    def scale_mass_matrix(self, nodes: int | np.ndarray, scale: float):
        self.__mass_matrix[np.ix_(self.map_node_to_dof(
            nodes), self.map_node_to_dof(nodes))] *= scale

    def _init_state(self, geo: Geometry):
        """Initialize RobotState state for q0"""
        a1, a2 = self.compute_space_parallel()
        m1, m2 = self.compute_material_directors(self.q0, a1, a2)

        edges = np.array([(s[1], s[3])
                         for s in geo.bend_twist_springs], dtype=np.int64)
        sign = np.array([sign for sign in geo.bend_twist_signs])

        if edges.size:
            self.__undef_ref_twist = compute_reference_twist(
                edges, sign, a1, self.__tangent, np.zeros(sign.shape[0]))
            ref_twist = compute_reference_twist(
                edges, sign, a1, self.__tangent, self.__undef_ref_twist
            )
        else:
            self.__undef_ref_twist = np.array([])
            ref_twist = np.array([])

        self.__state = RobotState.init(self.__q0, a1, a2, m1, m2, ref_twist)

    def _init_fixed_dof(self):
        """Initialize all DOF as free"""
        self.__fixed_nodes = np.array([], dtype=np.int64)
        self.__fixed_edges = np.array([], dtype=np.int64)

        self.__fixed_dof, self.__free_dof = self._find_fixed_free_dof(
            self.__fixed_nodes, self.__fixed_edges)

    def _init_directors_and_springs(self, geo: Geometry):
        """Initialize spring list objects"""
        n_rod = geo.rod_stretch_springs.shape[0]

        # Stretch springs
        rod_springs = [StretchSpring(spring, self.ref_len[i], self)
                       for i, spring in enumerate(geo.rod_stretch_springs)]
        shell_springs = [StretchSpring(spring, self.ref_len[i + n_rod], self, self.__ks[i + n_rod])
                         for i, spring in enumerate(geo.shell_stretch_springs)]
        self.__stretch_springs = rod_springs + shell_springs

        self.__bend_twist_springs = [
            BendTwistSpring(
                spring, sign, np.array([0, 0]), undef_ref_twist, self)
            for spring, sign, undef_ref_twist in zip(geo.bend_twist_springs, geo.bend_twist_signs, self.__undef_ref_twist)
        ]

        # Hinge springs
        self.__shell_hinge_springs = [
            HingeSpring(spring, self) for spring in geo.hinges
        ]

        # TODO: Move this functionality into bend_twist_spring init
        # Set initial spring parameters
        self._set_kappa(self.state.m1, self.state.m2)
        # Mid-edge?

    @staticmethod
    def _construct_edge_combinations(edges: np.ndarray) -> np.ndarray:
        n = edges.shape[0]
        if n == 0:
            return np.array([])

        i, j = np.triu_indices(n, 1)
        mask = ~np.any((edges[i, None] == edges[j][:, None, :]) | (
            edges[i, None] == edges[j][:, None, ::-1]), axis=(1, 2))
        valid = np.column_stack((i[mask], j[mask]))
        return np.hstack((edges[valid[:, 0]], edges[valid[:, 1]]))

    def __init_curvature_midedge(self) -> typing.Tuple[np.ndarray, ...]:
        faces = self.__face_nodes_shell
        face_edges = self.__face_edges
        signs = self.__sign_faces

        # Vectorize face processing
        all_p_is = self.__q0[3 * faces].reshape(-1, 3, 3)
        all_xi_is = self.__q0[3*self.__n_nodes + face_edges]
        tau_0 = self._update_pre_comp_shell(self.__q0)
        all_tau0_is = tau_0[:, face_edges].transpose(1, 0, 2)

        # Compute t, f, c for all faces simultaneously
        t, f, c = self._batch_init_tfc_midedge(all_p_is, all_tau0_is, signs)

        return t.transpose(1, 2, 0), f.T, c.T, all_xi_is.T

    @staticmethod
    def _batch_init_tfc_midedge(p_s: np.ndarray, tau0_s: np.ndarray, s_s: np.ndarray) -> typing.Tuple[np.ndarray, ...]:
        # Vectorized computation for all faces
        vi = p_s[:, 2] - p_s[:, 1]  # pk - pj
        vj = p_s[:, 0] - p_s[:, 2]  # pi - pk
        vk = p_s[:, 1] - p_s[:, 0]  # pj - pi

        norms = np.linalg.norm(np.cross(vk, vi), axis=1, keepdims=True)
        unit_norm = np.cross(vk, vi) / (norms + 1e-10)

        # Compute t vectors
        t_i = np.cross(vi, unit_norm)
        t_j = np.cross(vj, unit_norm)
        t_k = np.cross(vk, unit_norm)
        t_norms = np.linalg.norm([t_i, t_j, t_k], axis=2, keepdims=True)
        t_i, t_j, t_k = (x / (t_norms + 1e-10) for x in (t_i, t_j, t_k))

        # Compute c values
        li = np.linalg.norm(vi, axis=1, keepdims=True)
        lj = np.linalg.norm(vj, axis=1, keepdims=True)
        lk = np.linalg.norm(vk, axis=1, keepdims=True)

        dot_prods = np.einsum('ij,ijk->ik', t_i.reshape(-1, 3), tau0_s[:, 0])
        c_i = 1 / (norms * li * dot_prods + 1e-10)
        c_j = 1 / (norms * lj * dot_prods + 1e-10)
        c_k = 1 / (norms * lk * dot_prods + 1e-10)

        # Compute f values
        f_vals = np.einsum('ij,ijk->ik', unit_norm,
                           tau0_s * s_s[:, None, None])

        return np.stack((t_i, t_j, t_k), f_vals, np.stack((c_i, c_j, c_k)))

    def _set_kappa(self, m1: np.ndarray, m2: np.ndarray) -> None:
        """sets natural curvature of all internal bend twist springs"""
        springs = self.__bend_twist_springs

        if len(springs) == 0:
            return

        # Precompute all spring data
        nodes_ind = np.array([s.nodes_ind for s in springs])
        edges_ind = np.array([s.edges_ind for s in springs])
        sgn = np.array([s.sgn for s in springs])

        # Get positions for all springs
        n0_dofs = 3 * nodes_ind[:, 0]
        n1_dofs = 3 * nodes_ind[:, 1]
        n2_dofs = 3 * nodes_ind[:, 2]
        n0_pos = self.__q0[n0_dofs[:, None] + np.arange(3)]
        n1_pos = self.__q0[n1_dofs[:, None] + np.arange(3)]
        n2_pos = self.__q0[n2_dofs[:, None] + np.arange(3)]

        # Compute tangents and curvature
        t0 = (n1_pos - n0_pos) / \
            np.linalg.norm(n1_pos - n0_pos, axis=1, keepdims=True)
        t1 = (n2_pos - n1_pos) / \
            np.linalg.norm(n2_pos - n1_pos, axis=1, keepdims=True)
        kb = 2.0 * np.cross(t0, t1) / \
            (1.0 + np.einsum('ij,ij->i', t0, t1))[:, None]

        # Compute kappa values
        m1e = m1[edges_ind[:, 0]]
        m2e = sgn[:, 0][:, None] * m2[edges_ind[:, 0]]
        m1f = m1[edges_ind[:, 1]]
        m2f = sgn[:, 1][:, None] * m2[edges_ind[:, 1]]

        kappa1 = 0.5 * np.einsum('ij,ij->i', kb, m2e + m2f)
        kappa2 = -0.5 * np.einsum('ij,ij->i', kb, m1e + m1f)

        # Update springs in bulk
        for i, spring in enumerate(self.__bend_twist_springs):
            spring.kappa_bar = np.array([kappa1[i], kappa2[i]])

    def compute_material_directors(self, q: np.ndarray, a1: np.ndarray, a2: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        theta = self.get_theta(q)
        cos_theta = np.cos(theta)[:, None]
        sin_theta = np.sin(theta)[:, None]
        m1 = cos_theta * a1 + sin_theta * a2
        m2 = -sin_theta * a1 + cos_theta * a2
        return m1, m2

    @staticmethod
    def compute_reference_twist(springs: typing.List[BendTwistSpring],
                                a1: np.ndarray, tangent: np.ndarray,
                                ref_twist: np.ndarray) -> np.ndarray:
        if len(springs) == 0:
            return np.array([])

        edges = np.array([s.edges_ind for s in springs])
        sgn = np.array([s.sgn for s in springs])

        return compute_reference_twist(edges, sgn, a1, tangent, ref_twist)

    def _update_pre_comp_shell(self, q: np.ndarray) -> np.ndarray:
        faces = self.__face_nodes_shell
        face_edges = self.__face_edges

        # Compute face normals
        v1 = q[3*faces[:, 1]] - q[3*faces[:, 0]]
        v2 = q[3*faces[:, 2]] - q[3*faces[:, 1]]
        face_normals = np.cross(v1, v2)
        face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)

        # Accumulate edge normals
        edge_normals = np.zeros((self.__n_edges, 3))
        np.add.at(edge_normals, face_edges.ravel(),
                  face_normals.repeat(3, axis=0))
        edge_counts = np.bincount(face_edges.ravel(), minlength=self.__n_edges)
        edge_normals /= edge_counts[:, None] + 1e-10

        # Compute edge vectors and tau_0
        edge_vecs = q[3*self.__edges[:, 1]] - q[3*self.__edges[:, 0]]
        tau_0 = np.cross(edge_vecs, edge_normals)
        tau_0 /= np.linalg.norm(tau_0, axis=1, keepdims=True)
        return tau_0.T

    def compute_space_parallel(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        a1 = np.zeros((self.__n_edges_dof, 3))
        a2 = np.zeros((self.__n_edges_dof, 3))

        self.__tangent = self.compute_tangent(self.__q0)

        if self.__tangent.size:
            # Initialize first a1
            a1_init = np.cross(self.__tangent[0], np.array([0, 1, 0]))
            if np.linalg.norm(a1_init) < 1e-6:
                a1_init = np.cross(self.__tangent[0], np.array([0, 0, -1]))
            a1[0] = a1_init / np.linalg.norm(a1_init)
            a2[0] = np.cross(self.__tangent[0], a1[0])

            # Iterative parallel transport (depends on previous a1)
            for i in range(1, self.__n_edges_dof):
                t_prev = self.__tangent[i-1]
                t_curr = self.__tangent[i]
                a1_prev = a1[i-1]

                a1[i] = parallel_transport(a1_prev, t_prev, t_curr)
                a1[i] -= np.dot(a1[i], t_curr) * t_curr
                a1[i] /= np.linalg.norm(a1[i])
                a2[i] = np.cross(t_curr, a1[i])
        return a1, a2

    def compute_tangent(self, q: np.ndarray) -> np.ndarray:
        edges = self.__edges[:self.__n_edges_dof]
        n0 = edges[:, 0]
        n1 = edges[:, 1]
        pos0 = q.flatten()[self.map_node_to_dof(n0)]
        pos1 = q.flatten()[self.map_node_to_dof(n1)]
        vecs = pos1 - pos0
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        tangent = vecs / norms
        tangent[np.abs(tangent) < _TANGENT_THRESHOLD] = 0

        return tangent

    def compute_time_parallel(self, a1_old: np.ndarray, q0: np.ndarray, q: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        tangent0 = self.compute_tangent(q0)
        tangent = self.compute_tangent(q)

        a1_transported = parallel_transport(
            a1_old, tangent0, tangent)

        # Orthonormalization
        t_dot = np.einsum('ij,ij->i', a1_transported, tangent)
        a1 = a1_transported - tangent * t_dot[:, None]
        a1 /= np.linalg.norm(a1, axis=1, keepdims=True)
        a2 = np.cross(tangent, a1)

        return a1, a2

    def _find_fixed_free_dof(self, fixed_nodes: typing.List[int], fixed_edges: typing.List[int]) -> typing.Tuple[np.ndarray, np.ndarray]:
        # Vectorized node DOF mapping
        node_dofs = 3 * np.array(fixed_nodes)[:, None] + np.arange(3)
        fixed_node_dofs = node_dofs.ravel()

        # Edge DOF mapping
        fixed_edge_dofs = 3 * self.__n_nodes + np.array(fixed_edges)

        # Combine fixed DOFs
        fixed_dof = np.unique(np.concatenate(
            [fixed_node_dofs, fixed_edge_dofs]))

        # Find free DOFs using set operations
        all_dofs = np.arange(self.__n_dof)
        free_dof = np.setdiff1d(all_dofs, fixed_dof, assume_unique=True)

        return fixed_dof, free_dof

    def _get_fixed_edges(self, fixed_nodes: np.ndarray) -> np.ndarray:
        # Add edges in between fixed nodes
        edge_mask = np.isin(self.__edges[:, 0], fixed_nodes) & np.isin(
            self.__edges[:, 1], fixed_nodes)
        fixed_edge_indices = np.where(edge_mask)[0]

        # if 2d, edges cannot twist
        if self.sim_params.two_d_sim:
            fixed_edge_indices = np.union1d(
                fixed_edge_indices, np.arange(self.__n_edges_dof))

        return fixed_edge_indices

    def free_nodes(self, nodes: np.ndarray, fix_edges: bool = True) -> "SoftRobot":
        """ Return a SoftRobot object with freed nodes. If nodes is None, all nodes and edges are freed """
        # remove all nodes, or specified ones
        if nodes is None:
            new_fixed_nodes = np.empty(0)
        else:
            mask = np.isin(self.__fixed_nodes, nodes, invert=True)
            new_fixed_nodes = self.__fixed_nodes[mask]

        return self.fix_nodes(new_fixed_nodes, fix_edges)

    def fix_nodes(self, nodes: np.ndarray, fix_edges: bool = True) -> "SoftRobot":
        """ Return a SoftRobot object with new fixed nodes """
        ret = copy.copy(self)
        ret.__fixed_nodes = np.union1d(nodes, ret.__fixed_nodes)
        if fix_edges:
            ret.__fixed_edges = ret._get_fixed_edges(ret.__fixed_nodes)
        ret.__fixed_dof, ret.__free_dof = ret._find_fixed_free_dof(
            ret.__fixed_nodes, ret.__fixed_edges)
        return ret

    def update(self, **kwargs) -> "SoftRobot":
        """Return a new SoftRobot with updated state"""
        new_robot = copy.copy(self)
        new_robot.__state = dataclasses.replace(
            self.__state, **{k: v.copy() for k, v in kwargs.items()})
        return new_robot

    def get_theta(self, q: np.ndarray) -> np.ndarray:
        """ All DOF for edges (self.__n_edges_dof,)"""
        return q[3*self.__n_nodes: 3*self.__n_nodes + self.__n_edges_dof]

    @property
    def n_dof(self) -> int:
        """Total number of degrees of freedom"""
        return self.__n_dof

    @property
    def node_dof_indices(self) -> np.ndarray:
        """Node DOF indices matrix (n_nodes, 3)"""
        return np.arange(3 * self.__n_nodes).reshape(-1, 3)

    @property
    def end_node_dof_index(self) -> int:
        """First edge DOF index after node DOFs"""
        return 3 * self.__n_nodes

    @property
    def q0(self) -> np.ndarray:
        """Initial state vector (n_dof,)"""
        return self.__q0.view()

    @property
    def state(self) -> RobotState:
        """Return current state"""
        return self.__state

    @property
    def bend_twist_springs(self) -> typing.List[BendTwistSpring]:
        """List of bend-twist spring elements"""
        return self.__bend_twist_springs

    @property
    def stretch_springs(self) -> typing.List[StretchSpring]:
        """List of stretch spring elements"""
        return self.__stretch_springs

    @property
    def hinge_springs(self) -> typing.List[HingeSpring]:
        """List of hinge spring elements"""
        return self.__shell_hinge_springs

    @property
    def edges(self) -> np.ndarray:
        """Edges (n_edges, 2)"""
        return self.__edges.view()

    @property
    def face_nodes_shell(self) -> np.ndarray:
        """Shell face node indices (n_faces, 3)"""
        return self.__face_nodes_shell.view()

    @property
    def EA(self) -> float:
        """Axial stiffness"""
        return self.__EA

    @property
    def EI1(self) -> float:
        """First bending stiffness"""
        return self.__EI1

    @property
    def EI2(self) -> float:
        """Second bending stiffness"""
        return self.__EI2

    @property
    def GJ(self) -> float:
        """Torsional stiffness"""
        return self.__GJ

    @property
    def kb(self) -> float:
        """Hinge stiffness"""
        return self.__kb

    @property
    def sim_params(self) -> SimParams:
        """Simulation parameters object"""
        return self.__sim_params

    @property
    def env(self) -> Environment:
        """Environment parameters object"""
        return self.__env

    @property
    def mass_matrix(self) -> np.ndarray:
        """Diagonal mass matrix (n_dof, n_dof)"""
        return self.__mass_matrix.view()

    @property
    def ref_len(self) -> np.ndarray:
        """Reference lengths for all edges (n_edges,)"""
        return self.__ref_len.view()

    @property
    def voronoi_area(self) -> np.ndarray:
        """Voronoi area per node (n_nodes,)"""
        return self.__voronoi_area.view()

    @property
    def face_area(self) -> np.ndarray:
        """Face areas for shell elements (n_faces,)"""
        return self.__face_area.view()

    @property
    def undef_ref_twist(self) -> np.ndarray:
        """Initial reference twist values (n_bend_twist_springs,)"""
        return self.__undef_ref_twist.view()

    @property
    def fixed_nodes(self) -> np.ndarray:
        """Fixed node ids"""
        return self.__fixed_nodes.view()

    @property
    def fixed_edges(self) -> np.ndarray:
        """Fixed edge ids"""
        return self.__fixed_edges.view()

    @property
    def fixed_dof(self) -> np.ndarray:
        """Indices of constrained degrees of freedom"""
        return self.__fixed_dof.view()

    @property
    def free_dof(self) -> np.ndarray:
        """Indices of free degrees of freedom"""
        return self.__free_dof.view()
