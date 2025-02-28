import copy
import dataclasses
import typing

import numpy as np

from .environment import Environment
from .geometry import Geometry
from .springs import BendTwistSpring, StretchSpring, HingeSpring


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
                 geo: Geometry, sim_params: SimParams,
                 env: Environment):
        self.__sim_params = sim_params
        self.__env = env

        # Store parameters as instance variables
        self.__r0 = geom.rod_r0
        self.__h = geom.shell_h
        self.__rho = material.density
        self.__nu_shell = material.poisson_shell

        # Node and edge counts
        self.__n_nodes = geo.nodes.shape[0]
        self.__n_edges_rod_only = geo.rod_edges.shape[0]
        self.__n_edges_shell_only = geo.shell_edges.shape[0]
        n_edges_joint = geo.rod_shell_joint_edges_total.shape[0]
        self.__n_edges = geo.edges.shape[0]
        self.__n_edges_dof = self.__n_edges_rod_only + n_edges_joint
        self.__n_faces = geo.face_nodes.shape[0] if geo.face_nodes.size else 0

        # Geometry data
        self.__nodes = geo.nodes
        self.__edges = geo.edges
        self.__face_nodes_shell = geo.face_nodes
        self.__twist_angles = geo.twist_angles

        # Initialize DOF vector
        self.__n_dof = 3 * self.__n_nodes + self.__n_edges_dof
        self.__q0 = np.zeros(self.__n_dof)
        self.__q0[:3 * self.__n_nodes] = geo.nodes.flatten()
        self.__q0[3 * self.__n_nodes:3 * self.__n_nodes +
                  self.__n_edges_dof] = self.__twist_angles

        if sim_params.use_mid_edge:
            self.__n_dof += self.__n_edges_shell_only
            self.__q0 = np.concatenate(
                (self.__q0, np.zeros(self.__n_edges_shell_only)))

        self.__q = self.__q0.copy()
        self.__u = np.zeros(self.__q0.size)  # velocity vector

        # Precompute reference metrics
        self.__ref_len = self._get_ref_len()
        self.__voronoi_ref_len = self._get_voronoi_ref_len()
        self.__voronoi_area = self._get_voronoi_area()
        self.__face_area = self._get_face_area()

        # Mass matrix and stiffness properties
        self.__mass_matrix = self._get_mass_matrix(geom)
        G_rod = material.youngs_rod / (2 * (1 + material.poisson_rod))
        self.__EA, self.__EI1, self.__EI2, self.__GJ = self._compute_stiffness(
            geom, material, G_rod)
        self.__ks, self.__kb = self._compute_shell_stiffness(material)

        # Initialize directors and springs
        self._initialize_directors_and_springs(geo, sim_params)

    @staticmethod
    def map_node_to_dof(node_nums: typing.Union[int, np.ndarray]) -> np.ndarray:
        base = 3 * np.asarray(node_nums)
        return base[..., None] + np.array([0, 1, 2])

    def map_edge_to_dof(self, edge_nums: typing.Union[int, np.ndarray]) -> np.ndarray:
        return 3 * self.__n_nodes + np.asarray(edge_nums)

    def _get_ref_len(self) -> np.ndarray:
        edge_pairs = self.__edges
        vectors = self.__nodes[edge_pairs[:, 1]] - \
            self.__nodes[edge_pairs[:, 0]]
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

    def _get_mass_matrix(self, geom: GeomParams) -> np.ndarray:
        mass = np.zeros(self.__n_dof)

        # Shell face contributions
        if self.__n_faces:
            faces = self.__face_nodes_shell
            v1 = self.__nodes[faces[:, 1]] - self.__nodes[faces[:, 0]]
            v2 = self.__nodes[faces[:, 2]] - self.__nodes[faces[:, 1]]
            areas = 0.5 * np.linalg.norm(np.cross(v1, v2), axis=1)
            m_shell = self.__rho * areas * self.__h
            dof_indices = (3 * faces[:, :, None] + np.arange(3)).reshape(-1)
            np.add.at(mass, dof_indices, np.repeat(m_shell / 3, 9))

        if self.__n_nodes:
            # Rod node contributions
            if geom.axs is not None:
                dm_nodes = self.__voronoi_ref_len * geom.axs * self.__rho
            else:
                dm_nodes = self.__voronoi_ref_len * \
                    np.pi * (self.__r0 ** 2) * self.__rho
            node_dofs = np.arange(3 * self.__n_nodes).reshape(-1, 3)
            mass[node_dofs] += dm_nodes[:, None]

        # Edge contributions
        if self.__n_edges_dof:
            if geom.axs is not None:
                dm_edges = self.__ref_len[:self.__n_edges_dof] * \
                    geom.axs * self.__rho
            else:
                dm_edges = self.__ref_len[:self.__n_edges_dof] * \
                    np.pi * (self.__r0 ** 2) * self.__rho
            edge_mass = dm_edges / 2 * (self.__r0 ** 2)
            edge_dofs = 3 * self.__n_nodes + np.arange(self.__n_edges_dof)
            mass[edge_dofs] = edge_mass

        return np.diag(mass)

    def scale_mass_matrix(self, nodes: int | np.ndarray, scale: float):
        self.__mass_matrix[np.ix_(self.map_node_to_dof(
            nodes), self.map_node_to_dof(nodes))] *= scale

    def _compute_stiffness(self, geom: GeomParams, material: Material, G_rod: float) -> typing.Tuple[float, ...]:
        axs = geom.axs if geom.axs is not None else np.pi * self.__r0 ** 2
        EA = material.youngs_rod * axs

        if geom.ixs1 and geom.ixs2:
            EI1 = material.youngs_rod * geom.ixs1
            EI2 = material.youngs_rod * geom.ixs2
        else:
            EI1 = EI2 = material.youngs_rod * np.pi * self.__r0 ** 4 / 4

        if geom.jxs:
            GJ = G_rod * geom.jxs
        else:
            GJ = G_rod * np.pi * self.__r0 ** 4 / 2

        return EA, EI1, EI2, GJ

    def _compute_shell_stiffness(self, material: Material) -> typing.Tuple[float, float]:
        if self.__sim_params.use_mid_edge:
            kb = material.youngs_shell * self.__h ** 3 / \
                (24 * (1 - self.__nu_shell ** 2))
            ks = 2 * material.youngs_shell * self.__h / \
                (1 - self.__nu_shell ** 2) * self.__ref_len
        else:
            ks = (3 ** 0.5 / 2) * material.youngs_shell * \
                self.__h * self.__ref_len
            kb = (2 / (3 ** 0.5)) * material.youngs_shell * \
                (self.__h ** 3) / 12
        return ks, kb

    def _initialize_directors_and_springs(self, geo: Geometry, sim_params: SimParams):
        # Initialize edge combinations for contact (simplified)
        self.__edge_combos = self._construct_edge_combinations(
            np.concatenate((geo.rod_edges, geo.rod_shell_joint_edges)
                           ) if geo.rod_shell_joint_edges.size else geo.rod_edges
        )

        # Initialize directors and reference frames
        self.__a1 = np.zeros((self.__n_edges_dof, 3))
        self.__a2 = np.zeros((self.__n_edges_dof, 3))
        self.__m1 = np.zeros((self.__n_edges_dof, 3))
        self.__m2 = np.zeros((self.__n_edges_dof, 3))

        # Initialize springs
        n_rod = geo.rod_stretch_springs.shape[0]
        rod_springs = [StretchSpring(self.ref_len[i], spring, self)
                       for i, spring in enumerate(geo.rod_stretch_springs)]
        shell_springs = [StretchSpring(self.ref_len[i + n_rod], spring, self, self.__ks[i + n_rod])
                         for i, spring in enumerate(geo.shell_stretch_springs)]
        self.__stretch_springs = rod_springs + shell_springs

        self.__bend_twist_springs = [
            BendTwistSpring(
                spring, sign, np.array([0, 0]), 0, self)
            for spring, sign in zip(geo.bend_twist_springs, geo.bend_twist_signs)
        ]

        self.__hinge_springs = [
            HingeSpring(spring, self) for spring in geo.hinges
        ]

    @staticmethod
    def _construct_edge_combinations(edges: np.ndarray) -> np.ndarray:
        n = edges.shape[0]
        # FIXME: not sure what default should be
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
        all_p_is = self.__q[3 * faces].reshape(-1, 3, 3)
        all_xi_is = self.__q[3*self.__n_nodes + face_edges]
        tau_0 = self.update_pre_comp_shell(self.__q)
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
        n0_pos = self.__q[n0_dofs[:, None] + np.arange(3)]
        n1_pos = self.__q[n1_dofs[:, None] + np.arange(3)]
        n2_pos = self.__q[n2_dofs[:, None] + np.arange(3)]

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

    def compute_reference_twist(self, springs: typing.List[BendTwistSpring],
                                a1: np.ndarray, tangent: np.ndarray,
                                ref_twist: np.ndarray) -> np.ndarray:
        if len(springs) == 0:
            return np.array([])

        edges = np.array([s.edges_ind for s in springs])
        sgn = np.array([s.sgn for s in springs])

        e0 = edges[:, 0]
        e1 = edges[:, 1]
        t0 = tangent[e0] * sgn[:, 0][:, None]
        t1 = tangent[e1] * sgn[:, 1][:, None]
        u0 = a1[e0]
        u1 = a1[e1]

        # Batch parallel transport
        ut = self._batch_parallel_transport(u0, t0, t1)
        ut = self._batch_rotate_axis_angle(ut, t1, ref_twist)

        # Compute signed angles in bulk
        angles = self._batch_signed_angle(ut, u1, t1)
        return ref_twist + angles

    def _batch_parallel_transport(self, u: np.ndarray, t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
        # Compute initial cross product and norms
        b = np.cross(t0, t1)
        b_norm = np.linalg.norm(b, axis=1, keepdims=True)
        mask = b_norm.squeeze() < 1e-10

        # Safe normalization with epsilon guard
        safe_b_norm = np.where(b_norm < 1e-10, 1.0, b_norm)
        b_valid = b / safe_b_norm

        # Orthogonalize against t0 with safe dot product
        dot_prod = np.einsum('ij,ij->i', b_valid, t0)
        b_valid = b_valid - dot_prod[:, None] * t0

        # Safe normalization of orthogonalized vector
        b_valid_norm = np.linalg.norm(b_valid, axis=1, keepdims=True)
        safe_b_valid_norm = np.where(b_valid_norm < 1e-10, 1.0, b_valid_norm)
        b_valid = b_valid / safe_b_valid_norm

        # Compute basis vectors
        n1 = np.cross(t0, b_valid)
        n2 = np.cross(t1, b_valid)

        # Calculate transport components using einsum
        components = (
            np.einsum('ij,ij->i', u, t0)[:, None] * t1 +
            np.einsum('ij,ij->i', u, n1)[:, None] * n2 +
            np.einsum('ij,ij->i', u, b_valid)[:, None] * b_valid
        )

        # Use original vectors where directions are parallel
        return np.where(mask[:, None], u, components)

    @staticmethod
    def _parallel_transport(u, t1, t2) -> np.ndarray:
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

    def _batch_rotate_axis_angle(self, v: np.ndarray, axis: np.ndarray, theta: np.ndarray) -> np.ndarray:
        cos_theta = np.cos(theta)[:, None]
        sin_theta = np.sin(theta)[:, None]
        return (cos_theta * v +
                sin_theta * np.cross(axis, v) +
                (1 - cos_theta) * np.einsum('ij,ij->i', axis, v)[:, None] * axis)

    @staticmethod
    def _batch_signed_angle(u: np.ndarray, v: np.ndarray, n: np.ndarray) -> np.ndarray:
        w = np.cross(u, v)
        norm_w = np.linalg.norm(w, axis=-1, keepdims=False)
        dot_uv = np.einsum('...i,...i', u, v)

        # Handle near-zero denominators
        safe_denominator = np.where(np.abs(dot_uv) < 1e-10, 1.0, dot_uv)
        angle = np.arctan2(norm_w, safe_denominator)

        # Compute sign with safe cross product check
        sign = np.sign(np.einsum('...i,...i', n, w))
        return angle * sign

    def update_pre_comp_shell(self, q: np.ndarray) -> np.ndarray:
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

    def compute_space_parallel(self) -> "SoftRobot":
        ret = copy.deepcopy(self)
        ret.__tangent = ret.compute_tangent(ret.__q0)

        if not ret.__tangent.size:
            return ret

        # Initialize first a1
        t0 = ret.__tangent[0]
        rand_vec = np.array([0, 1, 0])
        a1_init = np.cross(t0, rand_vec)
        if np.linalg.norm(a1_init) < 1e-6:
            rand_vec = np.array([0, 0, -1])
            a1_init = np.cross(t0, rand_vec)
        ret.__a1[0] = a1_init / np.linalg.norm(a1_init)
        ret.__a2[0] = np.cross(ret.__tangent[0], ret.__a1[0])

        # Iterative parallel transport (depends on previous a1)
        for i in range(1, ret.__n_edges_dof):
            t_prev = ret.__tangent[i-1]
            t_curr = ret.__tangent[i]
            a1_prev = ret.__a1[i-1]

            ret.__a1[i] = self._parallel_transport(a1_prev, t_prev, t_curr)
            ret.__a1[i] -= np.dot(ret.__a1[i], t_curr) * t_curr
            ret.__a1[i] /= np.linalg.norm(ret.__a1[i])
            ret.__a2[i] = np.cross(t_curr, ret.__a1[i])

        return ret

    def compute_tangent(self, q: np.ndarray) -> np.ndarray:
        edges = self.__edges[:self.__n_edges_dof]
        n0 = edges[:, 0]
        n1 = edges[:, 1]
        pos0 = q.flatten()[self.map_node_to_dof(n0)]  # shape (n_edges_dof, 3)
        pos1 = q.flatten()[self.map_node_to_dof(n1)]  # shape (n_edges_dof, 3)
        vecs = pos1 - pos0
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)  # prevent division by zero
        tangent = vecs / norms
        tangent[np.abs(tangent) < _TANGENT_THRESHOLD] = 0

        return tangent

    @staticmethod
    def compute_kappa(n0: np.ndarray, n1: np.ndarray, n2: np.ndarray,
                      m1e: np.ndarray, m2e: np.ndarray,
                      m1f: np.ndarray, m2f: np.ndarray) -> np.ndarray:
        t0 = (n1 - n0) / np.linalg.norm(n1 - n0)
        t1 = (n2 - n1) / np.linalg.norm(n2 - n1)
        kb = 2.0 * np.cross(t0, t1) / (1.0 + np.dot(t0, t1))
        return np.array([0.5 * np.dot(kb, m2e + m2f),
                        -0.5 * np.dot(kb, m1e + m1f)])

    def compute_material_directors(self, a1: np.ndarray, a2: np.ndarray, theta: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        cos_theta = np.cos(theta)[:, None]
        sin_theta = np.sin(theta)[:, None]
        m1 = cos_theta * a1 + sin_theta * a2
        m2 = -sin_theta * a1 + cos_theta * a2
        return m1, m2

    def compute_time_parallel(self, a1_old: np.ndarray, q0: np.ndarray, q: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        tangent0 = self.compute_tangent(q0)
        tangent = self.compute_tangent(q)

        a1_transported = self._batch_parallel_transport(
            a1_old, tangent0, tangent)

        # Orthonormalization
        t_dot = np.einsum('ij,ij->i', a1_transported, tangent)
        a1 = a1_transported - tangent * t_dot[:, None]
        a1 /= np.linalg.norm(a1, axis=1, keepdims=True)
        a2 = np.cross(tangent, a1)

        return a1, a2

    def find_fixed_free_dof(self, fixed_nodes: typing.List[int], fixed_edges: typing.List[int]) -> typing.Tuple[np.ndarray, np.ndarray]:
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

    def initialize(self, fixed_nodes: typing.List[int]) -> "SoftRobot":
        """ Return a SoftRobot object prepared for timestepping """
        ret = self.compute_space_parallel()

        # Material frame computation
        theta = self.get_theta(ret.q0)
        ret.__m1, ret.__m2 = self.compute_material_directors(
            ret.a1, ret.a2, theta)

        # Curvature computation
        ret._set_kappa(ret.__m1, ret.__m2)

        # Reference twist computation
        ret.__undef_ref_twist = self.compute_reference_twist(
            ret.bend_twist_springs, ret.a1, ret.tangent, np.zeros(
                len(self.__bend_twist_springs))
        )
        ret.__ref_twist = self.compute_reference_twist(
            ret.bend_twist_springs, ret.a1, ret.tangent, ret.__undef_ref_twist
        )

        # Boundary condition setup
        edge_mask = np.isin(ret.__edges[:, 0], fixed_nodes) & np.isin(
            ret.__edges[:, 1], fixed_nodes)
        fixed_edge_indices = np.where(edge_mask)[0]
        if ret.sim_params.two_d_sim:
            fixed_edge_indices = np.union1d(
                fixed_edge_indices, np.arange(ret.__n_edges_dof))

        ret.__fixed_edges = fixed_edge_indices
        ret.__fixed_dof, ret.__free_dof = ret.find_fixed_free_dof(
            fixed_nodes, ret.__fixed_edges)

        return ret

    def update(self, q: np.ndarray,
               u: np.ndarray = None,
               a1: np.ndarray = None,
               a2: np.ndarray = None,
               m1: np.ndarray = None,
               m2: np.ndarray = None,
               ref_twist: np.ndarray = None) -> "SoftRobot":
        ret = copy.copy(self)
        ret.__q = q.copy()
        if u is not None:
            ret.__u = u.copy()
        if a1 is not None:
            ret.__a1 = a1.copy()
        if a2 is not None:
            ret.__a2 = a2.copy()
        if m1 is not None:
            ret.__m1 = m1.copy()
        if m2 is not None:
            ret.__m2 = m2.copy()
        if ref_twist is not None:
            ret.__ref_twist = ref_twist.copy()
        return ret

    @staticmethod
    def rotate_axis_angle(v: np.ndarray, z: np.ndarray, theta: np.ndarray) -> np.ndarray:
        cos_t = np.cos(theta)[:, None]
        sin_t = np.sin(theta)[:, None]
        dot_prod = np.einsum('ij,ij->i', v, z)[:, None]
        return cos_t * v + sin_t * np.cross(z, v) + (1 - cos_t) * dot_prod * z

    @staticmethod
    def signed_angle(u: np.ndarray, v: np.ndarray, n: np.ndarray) -> np.ndarray:
        cross_prod = np.cross(u, v)
        angles = np.arctan2(np.linalg.norm(
            cross_prod, axis=1), np.einsum('ij,ij->i', u, v))
        signs = np.sign(np.einsum('ij,ij->i', n, cross_prod))
        return angles * signs

    def get_theta(self, q: np.ndarray) -> np.ndarray:
        """ All DOF for edges (self.__n_edges_dof,)"""
        return q[3*self.__n_nodes: 3*self.__n_nodes + self.__n_edges_dof]

    @property
    def ref_len(self) -> np.ndarray:
        """Reference lengths for all edges (n_edges,)"""
        return self.__ref_len.view()

    @property
    def q(self) -> np.ndarray:
        """Current state vector (n_dof,)"""
        return self.__q.view()

    @property
    def u(self) -> np.ndarray:
        """Current velocity vector (n_dof,)"""
        return self.__u.view()

    @property
    def q0(self) -> np.ndarray:
        """Initial state vector (n_dof,)"""
        return self.__q0.view()

    @property
    def a1(self) -> np.ndarray:
        """First reference frame directors (n_edges_dof, 3)"""
        return self.__a1.view()

    @property
    def a2(self) -> np.ndarray:
        """Second reference frame directors (n_edges_dof, 3)"""
        return self.__a2.view()

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
        return self.__hinge_springs

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
    def n_dof(self) -> int:
        """Total number of degrees of freedom"""
        return self.__n_dof

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
    def node_dof_indices(self) -> np.ndarray:
        """Node DOF indices matrix (n_nodes, 3)"""
        return np.arange(3 * self.__n_nodes).reshape(-1, 3)

    @property
    def end_node_dof_index(self) -> int:
        """First edge DOF index after node DOFs"""
        return 3 * self.__n_nodes

    @property
    def edges(self):
        return self.__edges

    @property
    def tangent(self) -> np.ndarray:
        """Current edge tangents (n_edges_dof, 3)"""
        return self.__tangent.view()

    @property
    def undef_ref_twist(self) -> np.ndarray:
        """Initial reference twist values (n_bend_twist_springs,)"""
        return self.__undef_ref_twist.view()

    @property
    def ref_twist(self) -> np.ndarray:
        """Reference twist values (n_bend_twist_springs,)"""
        return self.__ref_twist.view()

    @property
    def fixed_dof(self) -> np.ndarray:
        """Indices of constrained degrees of freedom"""
        return self.__fixed_dof.view()

    @property
    def free_dof(self) -> np.ndarray:
        """Indices of free degrees of freedom"""
        return self.__free_dof.view()

    @property
    def voronoi_area(self) -> np.ndarray:
        """Voronoi area per node (n_nodes,)"""
        return self.__voronoi_area.view()

    @property
    def face_area(self) -> np.ndarray:
        """Face areas for shell elements (n_faces,)"""
        return self.__face_area.view()
