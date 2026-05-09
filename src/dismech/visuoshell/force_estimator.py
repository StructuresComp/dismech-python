import dataclasses

import numpy as np

from ..elastics import HingeEnergy
from ..soft_robot import SoftRobot
from ..springs import HingeSprings
from ..state import RobotState
from .mesh import VisuoShellMesh, build_mesh


@dataclasses.dataclass
class VisuoShellForceEstimator:
    """Estimate nodal shell forces from tracked marker positions."""

    mesh: VisuoShellMesh
    springs: HingeSprings
    energy: HingeEnergy
    reference_state: RobotState

    @classmethod
    def from_reference_points(
        cls,
        reference_points: np.ndarray,
        kb: float | np.ndarray = 1.0e9,
        mesh_method: str = "convex_hull",
    ) -> "VisuoShellForceEstimator":
        reference_points = _validate_nodes(reference_points)
        mesh = build_mesh(reference_points, method=mesh_method)

        if np.isscalar(kb):
            kb_values = np.full(mesh.hinge_nodes.shape[0], kb, dtype=np.float64)
        else:
            kb_values = np.asarray(kb, dtype=np.float64)
            if kb_values.shape != (mesh.hinge_nodes.shape[0],):
                raise ValueError("kb array must have shape (n_hinges,)")

        springs = HingeSprings.from_arrays(
            mesh.hinge_nodes,
            kb_values,
            SoftRobot.map_node_to_dof,
        )
        springs.nat_strain = mesh.rest_angles

        reference_state = _state_from_nodes(reference_points)
        energy = HingeEnergy(springs, reference_state)

        return cls(
            mesh=mesh,
            springs=springs,
            energy=energy,
            reference_state=reference_state,
        )

    def elastic_force(self, nodes: np.ndarray) -> np.ndarray:
        """Return elastic force, equal to negative gradient of hinge energy."""
        nodes = _validate_nodes(nodes, self.n_nodes)
        force, _ = self.energy.grad_hess_energy_linear_elastic(_state_from_nodes(nodes))
        return force.reshape(-1, 3)

    def external_balance_force(self, nodes: np.ndarray) -> np.ndarray:
        """Return the external force that would balance the elastic hinge force."""
        return -self.elastic_force(nodes)

    def energy_value(self, nodes: np.ndarray) -> float:
        nodes = _validate_nodes(nodes, self.n_nodes)
        return float(self.energy.get_energy_linear_elastic(_state_from_nodes(nodes)))

    @property
    def triangles(self) -> np.ndarray:
        return self.mesh.triangles

    @property
    def n_nodes(self) -> int:
        return self.reference_state.q.size // 3


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


def _validate_nodes(nodes: np.ndarray, expected_n: int | None = None) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=np.float64)
    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("nodes must have shape (N, 3)")
    if expected_n is not None and nodes.shape[0] != expected_n:
        raise ValueError(f"expected {expected_n} nodes, got {nodes.shape[0]}")
    return nodes
