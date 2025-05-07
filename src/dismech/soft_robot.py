import copy
import dataclasses
import typing

import numpy as np

from .state import RobotState
from .stiffness import compute_rod_stiffness, compute_shell_stiffness
from .frame_util import compute_reference_twist, compute_tfc_midedge, parallel_transport, construct_edge_combinations, construct_triangle_combinations
from .environment import Environment
from .geometry import Geometry
from .params import GeomParams, Material, SimParams
from .springs import StretchSprings, BendSprings, TwistSprings, HingeSprings, TriangleSpring
from .contact import ContactPair


_TANGENT_THRESHOLD = 1e-10


class SoftRobot:
    def __init__(self, geom: GeomParams, material: Material,
                 geo: Geometry, sim_params: SimParams,
                 env: Environment):
        # Store mutable parameters
        self.__sim_params = sim_params
        self.__env = env

        # Multipart initialization
        self._init_geometry(geo)
        self._init_stiffness(geom, material)
        self._init_state(geo)
        self._init_springs(geo)
        self.__mass_matrix = self._get_mass_matrix(geom, material)

        # Contact
        self.__edge_combos = construct_edge_combinations(
            np.concatenate((geo.rod_edges, geo.rod_shell_joint_edges)
                           ) if geo.rod_shell_joint_edges.size else geo.rod_edges
        )
        self.__triangle_combos = construct_triangle_combinations(geo.face_edges)

        self.__contact_pairs = [ContactPair(
            e, self.map_node_to_dof) for e in self.__edge_combos]
        
        self.__triangle_contact_pairs = [ContactPair(
            t, self.map_node_to_dof) for t in self.__triangle_combos]

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

            self.__init_ts, self.__init_fs, self.__init_cs, self.__init_xis = self._init_curvature_midedge(
                geo)
        else:
            self.__tau0 = np.empty(0)
            self.__init_ts = np.empty(0)
            self.__init_fs = np.empty(0)
            self.__init_cs = np.empty(0)
            self.__init_xis = np.empty(0)

        self.__ref_len = self._get_ref_len()
        self.__voronoi_ref_len = self._get_voronoi_ref_len()
        self.__voronoi_area = self._get_voronoi_area()
        self.__face_area = self._get_face_area()

    def _init_stiffness(self, geom: GeomParams, material: Material):
        """Initialize global stiffness properties"""
        self.__EA, self.__EI1, self.__EI2, self.__GJ = compute_rod_stiffness(
            geom, material)
        self.__ks, self.__kb = compute_shell_stiffness(
            geom, material, self.ref_len, self.sim_params.use_mid_edge)
        self.__nu = material.poisson_shell

    def _init_state(self, geo: Geometry):
        """Initialize RobotState state for q0"""
        a1, a2 = self.compute_space_parallel()
        m1, m2 = self.compute_material_directors(self.q0, a1, a2)

        edges = np.array([(s[1], s[3])
                         for s in geo.bend_twist_springs], dtype=np.int64)
        sign = np.array([sign for sign in geo.bend_twist_signs])

        if edges.size:
            undef_ref_twist = compute_reference_twist(
                edges, sign, a1, self.__tangent, np.zeros(sign.shape[0]))
        else:
            undef_ref_twist = np.array([])

        self.__state = RobotState.init(
            self.__q0, a1, a2, m1, m2, undef_ref_twist, self.__tau0)

    def _init_springs(self, geo: Geometry):
        """Initialize spring list objects"""
        n_rod = geo.rod_stretch_springs.shape[0]
        n_shell = geo.shell_stretch_springs.shape[0]

        nodes_ind = np.concat([geo.rod_stretch_springs,
                               geo.shell_stretch_springs], axis=0)
        ref_len = self.__ref_len[:n_rod + n_shell]
        EA = np.concat([np.full(n_rod, self.__EA),
                       self.__ks[n_rod:n_rod+n_shell]], axis=0)

        self.__stretch_springs = StretchSprings.from_arrays(
            nodes_ind, ref_len, EA, self.map_node_to_dof)

        # Bend/twist spring
        n_bt_springs = geo.bend_twist_springs.shape[0]
        EI = np.tile(np.array([self.__EI1, self.__EI2]),
                     n_bt_springs).reshape(-1, 2)
        GJ = np.full(n_bt_springs, self.__GJ)

        self.__bend_springs = BendSprings.from_arrays(
            geo.bend_twist_springs, geo.bend_twist_signs, EI,
            self.ref_len, self.map_node_to_dof, self.map_edge_to_dof)

        self.__twist_springs = TwistSprings.from_arrays(
            geo.bend_twist_springs, geo.bend_twist_signs, GJ,
            self.ref_len, self.map_node_to_dof, self.map_edge_to_dof)

        if self.__sim_params.use_mid_edge:
            # Triangle springs
            self.__triangle_springs = [
                TriangleSpring(spring,
                               shell_edges,
                               face_edge,
                               sign,
                               self.__ref_len,
                               a,
                               ts,
                               fs,
                               cs,
                               xis,
                               self.__kb,
                               self.__nu,
                               self.map_node_to_dof,
                               self.map_face_edge_to_dof)
                for spring, face_edge, shell_edges, sign, a, ts, fs, cs, xis in zip(geo.face_nodes,
                                                                                    geo.face_edges,
                                                                                    geo.face_shell_edges,
                                                                                    geo.sign_faces,
                                                                                    self.__face_area,
                                                                                    self.__init_ts,
                                                                                    self.__init_fs,
                                                                                    self.__init_cs,
                                                                                    self.__init_xis)
            ]
            self.__shell_hinge_springs = []
        else:
            # Hinge springs
            kb = np.full(geo.hinges.shape[0], self.__kb)
            self.__shell_hinge_springs = HingeSprings.from_arrays(
                geo.hinges, kb, self.map_node_to_dof)
            self.__triangle_springs = []

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
                edge_mass = dm_edges * geom.jxs/geom.axs
            else:
                dm_edges = self.__ref_len[:self.__n_edges_dof] * \
                    np.pi * (geom.rod_r0 ** 2) * material.density
                edge_mass = dm_edges * (geom.rod_r0 ** 2)/2
            edge_dofs = 3 * self.__n_nodes + np.arange(self.__n_edges_dof)
            mass[edge_dofs] = edge_mass

        return mass

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

    def scale_mass_matrix(self, nodes: int | np.ndarray, scale: float):
        self.__mass_matrix[self.map_node_to_dof(nodes)] *= scale

    def _init_curvature_midedge(self, geo: Geometry) -> typing.Tuple[np.ndarray, ...]:
        self.__face_edges = geo.face_edges
        self.__sign_faces = geo.sign_faces

        # Vectorize face processing
        all_p_is = self.__q0[self.map_node_to_dof(self.__face_nodes_shell)]
        all_xi_is = self.__q0[self.map_edge_to_dof(self.__face_edges)]
        self.__tau0 = self.update_pre_comp_shell(self.__q0)
        all_tau0_is = self.__tau0[:, self.__face_edges].transpose(1, 2, 0)

        # Compute t, f, c for all faces simultaneously
        t, f, c = compute_tfc_midedge(all_p_is, all_tau0_is, self.__sign_faces)

        return t, f, c, all_xi_is

    def update_pre_comp_shell(self, q: np.ndarray) -> np.ndarray:
        if not self.sim_params.use_mid_edge:
            return np.empty(0)

        # Compute face normals
        v1 = q[self.map_node_to_dof(self.__face_nodes_shell[:, 1])] - \
            q[self.map_node_to_dof(self.__face_nodes_shell[:, 0])]
        v2 = q[self.map_node_to_dof(self.__face_nodes_shell[:, 2])] - \
            q[self.map_node_to_dof(self.__face_nodes_shell[:, 1])]
        face_normals = np.cross(v1, v2)
        face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)

        # Accumulate edge normals
        edge_normals = np.zeros((self.__n_edges, 3))
        np.add.at(edge_normals, self.__face_edges.ravel(),
                  face_normals.repeat(3, axis=0))

        # Normalize edge normals
        edge_counts = np.bincount(
            self.__face_edges.ravel(), minlength=self.__n_edges)
        edge_normals /= edge_counts[:, None] + 1e-10

        # Compute edge vectors and tau_0
        edge_vecs = q[self.map_node_to_dof(
            self.__edges[:, 1])] - q[self.map_node_to_dof(self.__edges[:, 0])]
        tau_0 = np.cross(edge_vecs, edge_normals)
        tau_0 /= np.linalg.norm(tau_0, axis=1, keepdims=True) + 1e-10

        return tau_0.T

    def compute_space_parallel(self) -> typing.Tuple[np.ndarray, np.ndarray]:
        a1 = np.zeros((self.__n_edges_dof, 3))
        a2 = np.zeros((self.__n_edges_dof, 3))

        self.__tangent = self._compute_tangent(self.__q0)

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

    def compute_time_parallel(self, a1_old: np.ndarray, q0: np.ndarray, q: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        tangent0 = self._compute_tangent(q0)
        tangent = self._compute_tangent(q)

        a1_transported = parallel_transport(
            a1_old, tangent0, tangent)

        # Orthonormalization
        t_dot = np.einsum('ij,ij->i', a1_transported, tangent)
        a1 = a1_transported - tangent * t_dot[:, None]
        a1 /= np.linalg.norm(a1, axis=1, keepdims=True)
        a2 = np.cross(tangent, a1)

        return a1, a2

    def compute_material_directors(self, q: np.ndarray, a1: np.ndarray, a2: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        theta = q[3*self.__n_nodes: 3*self.__n_nodes + self.__n_edges_dof]
        cos_theta = np.cos(theta)[:, None]
        sin_theta = np.sin(theta)[:, None]
        m1 = cos_theta * a1 + sin_theta * a2
        m2 = -sin_theta * a1 + cos_theta * a2
        return m1, m2

    def compute_reference_twist(self,
                                springs: TwistSprings,
                                q: np.ndarray,
                                a1: np.ndarray,
                                ref_twist: np.ndarray) -> np.ndarray:
        if springs.N == 0:
            return np.array([])

        return compute_reference_twist(springs.edges_ind, springs.sgn, a1, self._compute_tangent(q), ref_twist)

    def _compute_tangent(self, q: np.ndarray) -> np.ndarray:
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

    # Fix/free nodes and edges

    def free_nodes(self, nodes: typing.List[int] | np.ndarray, axis: int | None = None, fix_edges: bool = True) -> "SoftRobot":
        new_dof = np.setdiff1d(self.fixed_dof, self._get_node_dof_mask(
            nodes, axis), assume_unique=True)
        if fix_edges:
            new_dof = np.setdiff1d(
                new_dof, self._get_intermediate_edge_dof(nodes), assume_unique=True)
        return self._fix_dof(new_dof)

    def fix_nodes(self, nodes: typing.List[int] | np.ndarray, axis: int | None = None, fix_edges: bool = True) -> "SoftRobot":
        new_dof = np.union1d(
            self.fixed_dof, self._get_node_dof_mask(nodes, axis))
        if fix_edges:
            new_dof = np.union1d(
                new_dof, self._get_intermediate_edge_dof(nodes))
        return self._fix_dof(new_dof)

    def free_edges(self, edges: typing.List[int] | np.ndarray) -> "SoftRobot":
        new_dof = np.setdiff1d(
            self.fixed_dof, self.map_edge_to_dof(edges), assume_unique=True)
        return self._fix_dof(new_dof)

    def fix_edges(self, edges: np.ndarray) -> "SoftRobot":
        new_dof = np.union1d(self.fixed_dof, self.map_edge_to_dof(edges))
        return self._fix_dof(new_dof)

    def _fix_dof(self, new_fixed_dof):
        return copy.copy(self).update(free_dof=np.setdiff1d(np.arange(self.__n_dof), new_fixed_dof, assume_unique=True))

    def _get_intermediate_edge_dof(self, nodes: np.ndarray) -> np.ndarray:
        # Add edges in between fixed nodes
        edge_mask = np.isin(self.__edges[:, 0], nodes) & np.isin(
            self.__edges[:, 1], nodes)
        edges = np.where(edge_mask)[0]

        # if 2d, edges cannot twist
        if self.sim_params.two_d_sim:
            edges = np.union1d(
                edges, np.arange(self.__n_edges_dof))

        return self.map_edge_to_dof(edges)

    # Perturb system

    def move_nodes(self, nodes: typing.List[int] | np.ndarray, perturbation: np.ndarray, axis: int | None = None):
        q = np.copy(self.state.q)
        perturbation = np.asarray(perturbation)
        if perturbation.size == 3:
            perturbation = np.tile(perturbation, len(nodes))
        q[self._get_node_dof_mask(nodes, axis)] += perturbation
        return self.update(q)

    def twist_edges(self, edges: typing.List[int] | np.ndarray, perturbation: np.ndarray):
        q = np.copy(self.state.q)
        q[self.map_edge_to_dof(edges)] += np.asarray(perturbation)
        return self.update(q)

    # Utility

    @staticmethod
    def _get_node_dof_mask(nodes: typing.List[int] | np.ndarray, axis: int | None = None):
        """Masked get_node_dof for fixing specific axes """
        node_dof = SoftRobot.map_node_to_dof(nodes)
        return (node_dof if axis is None else node_dof[:, axis]).ravel()

    @staticmethod
    def map_node_to_dof(node_nums: int | np.ndarray) -> np.ndarray:
        return (3 * np.asarray(node_nums))[..., None] + np.array([0, 1, 2])

    def map_edge_to_dof(self, edge_nums: int | np.ndarray) -> np.ndarray:
        return 3 * self.__n_nodes + np.asarray(edge_nums)

    def map_face_edge_to_dof(self, edge_nums: int | np.ndarray) -> np.ndarray:
        return 3 * self.__n_nodes + self.__n_edges_dof + np.asarray(edge_nums)

    def update(
        self,
        q: np.ndarray = None,
        u: np.ndarray = None,
        a: np.ndarray = None,
        a1: np.ndarray = None,
        a2: np.ndarray = None,
        m1: np.ndarray = None,
        m2: np.ndarray = None,
        ref_twist: np.ndarray = None,
        free_dof: np.ndarray = None
    ) -> "SoftRobot":
        """Return a new SoftRobot with updated state"""
        state_updates = {
            k: v.copy() for k, v in locals().items()
            if k != "self" and v is not None
        }
        new_robot = copy.copy(self)
        new_robot.__state = dataclasses.replace(self.__state, **state_updates)
        return new_robot

    # Parameters

    @property
    def sim_params(self) -> SimParams:
        """Simulation parameters"""
        return self.__sim_params

    @property
    def env(self) -> Environment:
        """Environment parameters"""
        return self.__env

    # Indexing Constants

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

    # States

    @property
    def q0(self) -> np.ndarray:
        """Initial state vector (n_dof,)"""
        return self.__q0.view()

    @property
    def state(self) -> RobotState:
        """Return current state"""
        return self.__state

    # Springs

    @property
    def bend_springs(self) -> BendSprings:
        """List of bend-twist spring elements"""
        return self.__bend_springs
    
    @property
    def twist_springs(self) -> TwistSprings:
        """List of bend-twist spring elements"""
        return self.__twist_springs

    @property
    def stretch_springs(self) -> StretchSprings:
        """List of stretch spring elements"""
        return self.__stretch_springs

    @property
    def hinge_springs(self) -> HingeSprings:
        """List of hinge spring elements"""
        return self.__shell_hinge_springs

    @property
    def triangle_springs(self) -> typing.List[TriangleSpring]:
        """List of triangle spring elements"""
        return self.__triangle_springs

    @property
    def contact_pairs(self):
        return self.__contact_pairs
    
    @property
    def tri_contact_pairs(self):
        return self.__triangle_contact_pairs

    # Visualization properties

    @property
    def nodes(self) -> np.ndarray:
        """Nodes (n_nodes,)"""
        return self.__nodes.view()

    @property
    def edges(self) -> np.ndarray:
        """Edges (n_edges, 2)"""
        return self.__edges.view()

    @property
    def face_nodes_shell(self) -> np.ndarray:
        """Shell face node indices (n_faces, 3)"""
        return self.__face_nodes_shell.view()

    @property
    def fixed_dof(self) -> np.ndarray:
        """Indices of constrained degrees of freedom"""
        return np.setdiff1d(np.arange(self.n_dof), self.state.free_dof, assume_unique=True)

    # Geometric constants

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
    def mass_matrix(self) -> np.ndarray:
        """Diagonal mass matrix (n_dof, n_dof)"""
        return self.__mass_matrix.view()
