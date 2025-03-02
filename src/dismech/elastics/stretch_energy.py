import typing
import numpy as np

from .elastic_energy import ElasticEnergy
from ..springs import StretchSpring


class StretchEnergy(ElasticEnergy):
    def __init__(self, springs: typing.List[StretchSpring]):
        self.l_k = np.array([s.ref_len for s in springs], dtype=np.float64)
        self.inv_l_k = 1.0 / self.l_k

        super().__init__(
            self.l_k * np.array([s.EA for s in springs], dtype=np.float64),
            np.zeros_like(self.l_k, dtype=np.float64),
            np.array([s.nodes_ind for s in springs], dtype=np.int64),
            np.array([s.ind for s in springs], dtype=np.int64)
        )

        N = len(springs)

        # Pre-allocate all arrays
        self.grad_eps = np.empty((N, 6), dtype=np.float64)
        self.hess_eps = np.empty((N, 6, 6), dtype=np.float64)
        self.edge_outer = np.empty((N, 3, 3), dtype=np.float64)
        self.M = np.empty((N, 3, 3), dtype=np.float64)
        self.M2 = np.empty((N, 3, 3), dtype=np.float64)

    def get_strain(self, q: np.ndarray,
                   m1: np.ndarray | None = None,
                   m2: np.ndarray | None = None,
                   ref_twist: np.ndarray | None = None) -> np.ndarray:
        node0, node1 = self._get_node_pos(q)
        edge = node1 - node0
        edge_len = np.linalg.norm(edge, axis=1)
        return edge_len * self.inv_l_k - 1.0

    def grad_hess_strain(self, q: np.ndarray,
                         m1: np.ndarray | None = None,
                         m2: np.ndarray | None = None,
                         ref_twist: np.ndarray | None = None) -> typing.Tuple[np.ndarray, np.ndarray]:
        node0, node1 = self._get_node_pos(q)
        edge = node1 - node0
        edge_len = np.linalg.norm(edge, axis=1)
        tangent = edge / edge_len[:, None]
        eps = edge_len * self.inv_l_k - 1.0

        # Gradient computatio
        dEps_unit = tangent * self.inv_l_k[:, None]
        self.grad_eps[:, :3] = -dEps_unit
        self.grad_eps[:, 3:] = dEps_unit

        # Hessian computation
        # 1. Compute edge outer product
        np.einsum('...i,...j->...ij', edge, edge, out=self.edge_outer)

        # 2. Compute M components
        edge_len_cubed = edge_len ** 3
        np.divide(self.edge_outer,
                  edge_len_cubed[:, None, None], out=self.edge_outer)
        term1 = (self.inv_l_k - 1.0/edge_len)[:, None, None] * np.eye(3)
        np.add(term1, self.edge_outer, out=self.M)
        np.multiply(2.0 * self.inv_l_k[:, None, None], self.M, out=self.M)

        # 3. Compute M2 with safe division
        np.subtract(self.M, 2.0 * np.einsum('...i,...j->...ij',
                    dEps_unit, dEps_unit), out=self.M2)
        np.multiply(self.M2, 0.5, out=self.M2)
        mask = eps != 0
        np.divide(self.M2, eps[:, None, None],
                  where=mask[:, None, None], out=self.M2)
        self.M2[~mask] = 0  # Explicit zeroing

        # Fill Hessian blocks
        self.hess_eps[:, :3, :3] = self.M2
        self.hess_eps[:, 3:, 3:] = self.M2
        self.hess_eps[:, :3, 3:] = -self.M2
        self.hess_eps[:, 3:, :3] = -self.M2

        return self.grad_eps, self.hess_eps
