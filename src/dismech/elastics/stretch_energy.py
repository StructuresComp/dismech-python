import typing
import numpy as np

from .elastic_energyv2 import ElasticEnergy2
from ..state import RobotState
from ..springs import StretchSprings


class StretchEnergy(ElasticEnergy2):
    def __init__(self, springs: StretchSprings, initial_state: RobotState):
        super().__init__(springs, initial_state)
        self.inv_l_k = 1.0 / self._springs.ref_len

    @property
    def K(self):
        return self._springs.ref_len * self._springs.EA

    def get_strain(self, state: RobotState) -> np.ndarray:
        node0, node1 = self._get_node_pos(state.q)
        edge = node1 - node0
        edge_len = np.linalg.norm(edge, axis=1)
        return edge_len * self.inv_l_k - 1.0

    def grad_hess_strain(self, state: RobotState) -> typing.Tuple[np.ndarray, np.ndarray]:
        node0, node1 = self._get_node_pos(state.q)
        N = node0.shape[0]

        edge = node1 - node0
        edge_len = np.linalg.norm(edge, axis=1)
        tangent = edge / edge_len[:, None]
        eps = edge_len * self.inv_l_k - 1.0

        # Gradient computation
        eps_unit = tangent * self.inv_l_k[:, None]
        grad_eps = np.zeros((N, 6))
        grad_eps[:, :3] = -eps_unit
        grad_eps[:, 3:] = eps_unit

        # Hessian computation
        edge_outer = np.einsum('...i,...j->...ij', edge, edge)
        edge_outer /= (edge_len ** 3)[:, None, None]
        M = ((self.inv_l_k - 1.0 / edge_len)
             [:, None, None] * np.eye(3)) + edge_outer
        M *= 2.0 * self.inv_l_k[:, None, None]

        M2 = M - 2.0 * np.einsum('...i,...j->...ij', eps_unit, eps_unit)
        M2 *= 0.5
        mask = eps != 0
        M2 = np.divide(M2, eps[:, None, None], where=mask[:, None, None])
        M2[~mask] = 0  # Explicit zeroing

        # Fill Hessian blocks
        hess_eps = np.zeros((N, 6, 6))
        hess_eps[:, :3, :3] = M2
        hess_eps[:, 3:, 3:] = M2
        hess_eps[:, :3, 3:] = -M2
        hess_eps[:, 3:, :3] = -M2

        return grad_eps, hess_eps
