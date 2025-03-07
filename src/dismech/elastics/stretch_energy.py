import typing
import numpy as np

from .elastic_energy import ElasticEnergy
from ..state import RobotState
from ..springs import StretchSpring


class StretchEnergy(ElasticEnergy):
    def __init__(self, springs: typing.List[StretchSpring], initial_state: RobotState):
        self.l_k = np.array([s.ref_len for s in springs], dtype=np.float64)
        self.inv_l_k = 1.0 / self.l_k

        super().__init__(
            self.l_k * np.array([s.EA for s in springs], dtype=np.float64),
            np.array([s.nodes_ind for s in springs], dtype=np.int64),
            np.array([s.ind for s in springs], dtype=np.int64),
            initial_state
        )

        N = len(springs)

        self.edge = np.empty((N, 3), dtype=np.float64)
        self.edge_len = np.empty(N, dtype=np.float64)
        self.tangent = np.empty((N, 3), dtype=np.float64)
        self.eps = np.empty(N, dtype=np.float64)
        self.eps_unit = np.empty((N, 3), dtype=np.float64)
        self.grad_eps = np.empty((N, 6), dtype=np.float64)
        self.hess_eps = np.empty((N, 6, 6), dtype=np.float64)
        self.edge_outer = np.empty((N, 3, 3), dtype=np.float64)
        self.mask = np.empty(N, dtype=np.bool)
        self.M = np.empty((N, 3, 3), dtype=np.float64)
        self.M2 = np.empty((N, 3, 3), dtype=np.float64)

    def get_strain(self, state: RobotState) -> np.ndarray:
        node0, node1 = self._get_node_pos(state.q)
        np.subtract(node1, node0, out=self.edge)
        self.edge_len[:] = np.linalg.norm(self.edge, axis=1)
        return self.edge_len * self.inv_l_k - 1.0

    def grad_hess_strain(self, state: RobotState) -> typing.Tuple[np.ndarray, np.ndarray]:
        node0, node1 = self._get_node_pos(state.q)
        np.subtract(node1, node0, out=self.edge)
        self.edge_len[:] = np.linalg.norm(self.edge, axis=1)
        np.divide(self.edge, self.edge_len[:, None], out=self.tangent)
        self.eps[:] = self.edge_len * self.inv_l_k - 1.0

        # Gradient computation
        np.multiply(self.tangent, self.inv_l_k[:, None], out=self.eps_unit)
        self.grad_eps[:, :3] = -self.eps_unit
        self.grad_eps[:, 3:] = self.eps_unit

        # Hessian computation
        np.einsum('...i,...j->...ij', self.edge,
                  self.edge, out=self.edge_outer)

        np.divide(self.edge_outer,
                  (self.edge_len ** 3)[:, None, None], out=self.edge_outer)
        np.add((self.inv_l_k - 1.0/self.edge_len)
               [:, None, None] * np.eye(3), self.edge_outer, out=self.M)
        np.multiply(2.0 * self.inv_l_k[:, None, None], self.M, out=self.M)

        np.subtract(self.M, 2.0 * np.einsum('...i,...j->...ij',
                    self.eps_unit, self.eps_unit), out=self.M2)
        np.multiply(self.M2, 0.5, out=self.M2)
        self.mask[:] = self.eps != 0
        np.divide(self.M2, self.eps[:, None, None],
                  where=self.mask[:, None, None], out=self.M2)
        self.M2[~self.mask] = 0  # Explicit zeroing

        # Fill Hessian blocks
        self.hess_eps[:, :3, :3] = self.M2
        self.hess_eps[:, 3:, 3:] = self.M2
        self.hess_eps[:, :3, 3:] = -self.M2
        self.hess_eps[:, 3:, :3] = -self.M2

        return self.grad_eps, self.hess_eps
