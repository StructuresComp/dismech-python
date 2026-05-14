import dataclasses
import typing

import numpy as np

from ..elastics import HingeEnergy, StretchEnergy, TriangleEnergy
from ..frame_util import compute_tfc_midedge
from ..soft_robot import SoftRobot
from ..springs import HingeSprings, StretchSprings, TriangleSpring
from ..state import RobotState
from .mesh import VisuoShellMesh, build_mesh


@dataclasses.dataclass
class VisuoShellForceEstimator:
    """Estimate nodal shell forces from tracked marker positions."""

    mesh: VisuoShellMesh
    springs: HingeSprings | list[TriangleSpring]
    energy: HingeEnergy | TriangleEnergy
    reference_state: RobotState
    use_midedge: bool = False
    midedge: "VisuoShellMidedgeData | None" = None
    reference_n_nodes: int = 0
    stretch_springs: StretchSprings | None = None
    stretch_energy: StretchEnergy | None = None

    @classmethod
    def from_reference_points(
        cls,
        reference_points: np.ndarray,
        kb: float | np.ndarray = 1.0e9,
        ks: float | np.ndarray = 1.0e12,
        nu: float = 0.5,
        mesh_method: str = "convex_hull",
        use_midedge: bool = False,
    ) -> "VisuoShellForceEstimator":
        reference_points = _validate_nodes(reference_points)
        mesh = build_mesh(reference_points, method=mesh_method)

        shell_edges, face_edges, signs = _face_edge_connectivity(mesh.triangles)
        edge_ref_len = np.linalg.norm(
            reference_points[shell_edges[:, 1]] - reference_points[shell_edges[:, 0]], axis=1
        )

        if use_midedge:
            midedge = _build_midedge_data(
                reference_points, mesh.triangles, shell_edges, face_edges, signs, edge_ref_len
            )
            kb_values = _stiffness_values(kb, mesh.triangles.shape[0], "n_triangles")
            springs = _build_triangle_springs(midedge, kb_values, nu)
            reference_state = _midedge_state_from_nodes(reference_points, midedge)
            energy = TriangleEnergy(springs, reference_state, zero_reference=True)
        else:
            midedge = None
            kb_values = _stiffness_values(kb, mesh.hinge_nodes.shape[0], "n_hinges")
            springs = HingeSprings.from_arrays(
                mesh.hinge_nodes,
                kb_values,
                SoftRobot.map_node_to_dof,
            )
            springs.nat_strain = mesh.rest_angles
            reference_state = _state_from_nodes(reference_points)
            energy = HingeEnergy(springs, reference_state)

        ks_values = _stiffness_values(ks, shell_edges.shape[0], "n_edges")
        stretch_springs = StretchSprings.from_arrays(
            shell_edges, edge_ref_len, ks_values, SoftRobot.map_node_to_dof
        )
        stretch_energy = StretchEnergy(stretch_springs, reference_state)

        return cls(
            mesh=mesh,
            springs=springs,
            energy=energy,
            reference_state=reference_state,
            use_midedge=use_midedge,
            midedge=midedge,
            reference_n_nodes=reference_points.shape[0],
            stretch_springs=stretch_springs,
            stretch_energy=stretch_energy,
        )

    def elastic_force(self, nodes: np.ndarray) -> np.ndarray:
        """Return the reference-calibrated elastic force at each tracked node."""
        nodes = _validate_nodes(nodes, self.n_nodes)
        state = self._state_from_nodes(nodes)
        bend_force = self.energy.grad_energy_linear_elastic(state)
        stretch_force = self.stretch_energy.grad_energy_linear_elastic(state)
        force = bend_force + stretch_force
        return force[: 3 * self.n_nodes].reshape(-1, 3)

    def external_balance_force(self, nodes: np.ndarray) -> np.ndarray:
        """Return the external force that would balance the elastic force."""
        return -self.elastic_force(nodes)

    def energy_value(self, nodes: np.ndarray) -> float:
        nodes = _validate_nodes(nodes, self.n_nodes)
        state = self._state_from_nodes(nodes)
        bend_energy = float(np.sum(self.energy.get_energy_linear_elastic(state)))
        stretch_energy = float(np.sum(self.stretch_energy.get_energy_linear_elastic(state)))
        return bend_energy + stretch_energy

    @property
    def reference_force(self) -> np.ndarray:
        """Per-DOF reference gradient (midedge path only) used internally by the energy.

        For the hinge path this returns an empty array because HingeEnergy
        already vanishes at the reference state and no calibration is needed.
        """
        if self.use_midedge:
            ref = self.energy.reference_force
            return np.asarray(ref) if isinstance(ref, np.ndarray) else np.empty(0)
        return np.empty(0)

    @property
    def reference_energy(self) -> float:
        """Total reference bending energy (midedge path only)."""
        if self.use_midedge:
            ref = self.energy.reference_energy
            if isinstance(ref, np.ndarray):
                return float(np.sum(ref))
            return float(ref)
        return 0.0

    def _state_from_nodes(self, nodes: np.ndarray) -> RobotState:
        if self.use_midedge:
            if self.midedge is None:
                raise ValueError("midedge data is required when use_midedge is True")
            return _midedge_state_from_nodes(nodes, self.midedge)
        return _state_from_nodes(nodes)

    @property
    def triangles(self) -> np.ndarray:
        return self.mesh.triangles

    @property
    def n_nodes(self) -> int:
        return self.reference_n_nodes


@dataclasses.dataclass(frozen=True)
class VisuoShellMidedgeData:
    """Connectivity and reference data needed by TriangleEnergy."""

    triangles: np.ndarray
    shell_edges: np.ndarray
    face_edges: np.ndarray
    signs: np.ndarray
    ref_len: np.ndarray
    face_area: np.ndarray
    init_ts: np.ndarray
    init_fs: np.ndarray
    init_cs: np.ndarray
    init_xis: np.ndarray
    n_reference_nodes: int

    @property
    def n_nodes(self) -> int:
        return self.n_reference_nodes

    @property
    def n_edges(self) -> int:
        return self.shell_edges.shape[0]


def _state_from_nodes(nodes: np.ndarray) -> RobotState:
    q = np.asarray(nodes, dtype=np.float64).reshape(-1)
    return RobotState.init(
        q,
        np.empty((0, 3)),
        np.empty((0, 3)),
        np.empty((0, 3)),
        np.empty((0, 3)),
        np.empty(0),
        np.empty(0),
    )


def _midedge_state_from_nodes(nodes: np.ndarray, midedge: VisuoShellMidedgeData) -> RobotState:
    q = np.zeros(3 * midedge.n_nodes + midedge.n_edges, dtype=np.float64)
    q[: 3 * midedge.n_nodes] = np.asarray(nodes, dtype=np.float64).reshape(-1)
    return RobotState.init(
        q,
        np.empty((0, 3)),
        np.empty((0, 3)),
        np.empty((0, 3)),
        np.empty((0, 3)),
        np.empty(0),
        _compute_midedge_tau(nodes, midedge),
    )


def _validate_nodes(nodes: np.ndarray, expected_n: int | None = None) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("nodes must have shape (N, 3)")
    if expected_n is not None and nodes.shape[0] != expected_n:
        raise ValueError(f"expected {expected_n} nodes, got {nodes.shape[0]}")
    return nodes


def _stiffness_values(kb: float | np.ndarray, size: int, size_name: str) -> np.ndarray:
    if np.isscalar(kb):
        return np.full(size, kb, dtype=np.float64)

    kb_values = np.asarray(kb, dtype=np.float64)
    if kb_values.shape != (size,):
        raise ValueError(f"kb array must have shape ({size_name},)")
    return kb_values


def _build_midedge_data(
    nodes: np.ndarray,
    triangles: np.ndarray,
    shell_edges: np.ndarray,
    face_edges: np.ndarray,
    signs: np.ndarray,
    ref_len: np.ndarray,
) -> VisuoShellMidedgeData:
    face_area = _triangle_areas(nodes, triangles)
    tau = _compute_tau(nodes, triangles, shell_edges, face_edges)
    all_p = nodes[triangles]
    all_tau = tau[:, face_edges].transpose(1, 2, 0)
    init_ts, init_fs, init_cs = compute_tfc_midedge(all_p, all_tau, signs)
    init_xis = np.zeros((triangles.shape[0], 3), dtype=np.float64)

    return VisuoShellMidedgeData(
        triangles=triangles,
        shell_edges=shell_edges,
        face_edges=face_edges,
        signs=signs,
        ref_len=ref_len,
        face_area=face_area,
        init_ts=init_ts,
        init_fs=init_fs,
        init_cs=init_cs,
        init_xis=init_xis,
        n_reference_nodes=nodes.shape[0],
    )


def _face_edge_connectivity(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge_lookup: dict[tuple[int, int], int] = {}
    shell_edges: list[tuple[int, int]] = []
    face_edges = np.zeros((triangles.shape[0], 3), dtype=np.int64)
    signs = np.zeros((triangles.shape[0], 3), dtype=np.int64)

    for face_idx, (n0, n1, n2) in enumerate(triangles):
        for edge_pos, edge in enumerate(((n1, n2), (n2, n0), (n0, n1))):
            edge = typing.cast(tuple[int, int], tuple(int(n) for n in edge))
            reverse = (edge[1], edge[0])
            if edge in edge_lookup:
                edge_idx = edge_lookup[edge]
                sign = 1
            elif reverse in edge_lookup:
                edge_idx = edge_lookup[reverse]
                sign = -1
            else:
                edge_idx = len(shell_edges)
                shell_edges.append(edge)
                edge_lookup[edge] = edge_idx
                sign = 1

            face_edges[face_idx, edge_pos] = edge_idx
            signs[face_idx, edge_pos] = sign

    return np.asarray(shell_edges, dtype=np.int64), face_edges, signs


def _build_triangle_springs(
    midedge: VisuoShellMidedgeData,
    kb_values: np.ndarray,
    nu: float,
) -> list[TriangleSpring]:
    return [
        TriangleSpring(
            triangle,
            face_edges,
            face_edges,
            signs,
            midedge.ref_len,
            area,
            init_ts,
            init_fs,
            init_cs,
            init_xis,
            kb,
            nu,
            SoftRobot.map_node_to_dof,
            _map_midedge_to_dof(midedge.n_nodes),
        )
        for triangle, face_edges, signs, area, init_ts, init_fs, init_cs, init_xis, kb in zip(
            midedge.triangles,
            midedge.face_edges,
            midedge.signs,
            midedge.face_area,
            midedge.init_ts,
            midedge.init_fs,
            midedge.init_cs,
            midedge.init_xis,
            kb_values,
        )
    ]


def _map_midedge_to_dof(n_nodes: int) -> typing.Callable[[int | np.ndarray], np.ndarray]:
    def map_face_edge_to_dof(edge_nums: int | np.ndarray) -> np.ndarray:
        return 3 * n_nodes + np.asarray(edge_nums)

    return map_face_edge_to_dof


def _compute_midedge_tau(nodes: np.ndarray, midedge: VisuoShellMidedgeData) -> np.ndarray:
    return _compute_tau(nodes, midedge.triangles, midedge.shell_edges, midedge.face_edges)


def _compute_tau(
    nodes: np.ndarray,
    triangles: np.ndarray,
    shell_edges: np.ndarray,
    face_edges: np.ndarray,
) -> np.ndarray:
    face_normals = _face_normals(nodes, triangles)
    edge_normals = np.zeros((shell_edges.shape[0], 3), dtype=np.float64)
    np.add.at(edge_normals, face_edges.ravel(), face_normals.repeat(3, axis=0))

    edge_counts = np.bincount(face_edges.ravel(), minlength=shell_edges.shape[0])
    edge_normals /= edge_counts[:, None] + 1.0e-10

    edge_vectors = nodes[shell_edges[:, 1]] - nodes[shell_edges[:, 0]]
    tau = np.cross(edge_vectors, edge_normals)
    tau /= np.linalg.norm(tau, axis=1, keepdims=True) + 1.0e-10
    return tau.T


def _face_normals(nodes: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    p0 = nodes[triangles[:, 0]]
    p1 = nodes[triangles[:, 1]]
    p2 = nodes[triangles[:, 2]]
    normals = np.cross(p1 - p0, p2 - p1)
    return normals / np.linalg.norm(normals, axis=1, keepdims=True)


def _triangle_areas(nodes: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    p0 = nodes[triangles[:, 0]]
    p1 = nodes[triangles[:, 1]]
    p2 = nodes[triangles[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p1), axis=1)
