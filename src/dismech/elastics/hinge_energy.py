import typing
import numpy as np

from .elastic_energy import ElasticEnergy
from ..state import RobotState
from ..springs import HingeSpring


class HingeEnergy(ElasticEnergy):
    def __init__(self, springs: typing.List[HingeSpring]):
        super().__init__(
            np.array([s.kb for s in springs]),
            np.array([s.theta_bar for s in springs], dtype=np.float64),
            np.array([s.nodes_ind for s in springs]),
            np.array([s.ind for s in springs])
        )

        N = len(springs)

        # Edge vectors and norms
        self.m_e0 = np.empty((N, 3), dtype=np.float64)
        self.m_e1 = np.empty((N, 3), dtype=np.float64)
        self.m_e2 = np.empty((N, 3), dtype=np.float64)
        self.m_e3 = np.empty((N, 3), dtype=np.float64)
        self.m_e4 = np.empty((N, 3), dtype=np.float64)

        # Norms
        self.norm_e0 = np.empty((N, 1), dtype=np.float64)
        self.norm_e1 = np.empty((N, 1), dtype=np.float64)
        self.norm_e2 = np.empty((N, 1), dtype=np.float64)
        self.norm_e3 = np.empty((N, 1), dtype=np.float64)
        self.norm_e4 = np.empty((N, 1), dtype=np.float64)

        # Cross products
        self.cross_e0_e1 = np.empty((N, 3), dtype=np.float64)
        self.cross_e0_e2 = np.empty((N, 3), dtype=np.float64)
        self.cross_e0_e3 = np.empty((N, 3), dtype=np.float64)
        self.cross_e0_e4 = np.empty((N, 3), dtype=np.float64)

        # Trigonometric terms
        self.m_cosA1 = np.empty((N, 1), dtype=np.float64)
        self.m_cosA2 = np.empty((N, 1), dtype=np.float64)
        self.m_cosA3 = np.empty((N, 1), dtype=np.float64)
        self.m_cosA4 = np.empty((N, 1), dtype=np.float64)
        self.m_sinA1 = np.empty((N, 1), dtype=np.float64)
        self.m_sinA2 = np.empty((N, 1), dtype=np.float64)
        self.m_sinA3 = np.empty((N, 1), dtype=np.float64)
        self.m_sinA4 = np.empty((N, 1), dtype=np.float64)

        # Normals and helpers
        self.m_nn1 = np.empty((N, 3), dtype=np.float64)
        self.m_nn2 = np.empty((N, 3), dtype=np.float64)
        self.m_h1 = np.empty((N, 1), dtype=np.float64)
        self.m_h2 = np.empty((N, 1), dtype=np.float64)
        self.m_h3 = np.empty((N, 1), dtype=np.float64)
        self.m_h4 = np.empty((N, 1), dtype=np.float64)
        self.m_h01 = np.empty((N, 1), dtype=np.float64)
        self.m_h02 = np.empty((N, 1), dtype=np.float64)

        # Gradient and intermediate vectors
        self.grad_theta = np.zeros((N, 12), dtype=np.float64)
        self.m_m1 = np.empty((N, 3), dtype=np.float64)
        self.m_m2 = np.empty((N, 3), dtype=np.float64)
        self.m_m3 = np.empty((N, 3), dtype=np.float64)
        self.m_m4 = np.empty((N, 3), dtype=np.float64)
        self.m_m01 = np.empty((N, 3), dtype=np.float64)
        self.m_m02 = np.empty((N, 3), dtype=np.float64)

        self.temp_block_2d = np.empty((N, 3), dtype=np.float64)

        # Hessian components
        self.hess_theta = np.zeros((N, 12, 12), dtype=np.float64)
        self.M331 = np.empty((N, 3, 3), dtype=np.float64)
        self.M311 = np.empty((N, 3, 3), dtype=np.float64)
        self.M131 = np.empty((N, 3, 3), dtype=np.float64)
        self.M3011 = np.empty((N, 3, 3), dtype=np.float64)
        self.M111 = np.empty((N, 3, 3), dtype=np.float64)
        self.M1011 = np.empty((N, 3, 3), dtype=np.float64)
        self.M442 = np.empty((N, 3, 3), dtype=np.float64)
        self.M422 = np.empty((N, 3, 3), dtype=np.float64)
        self.M242 = np.empty((N, 3, 3), dtype=np.float64)
        self.M4022 = np.empty((N, 3, 3), dtype=np.float64)
        self.M222 = np.empty((N, 3, 3), dtype=np.float64)
        self.M2022 = np.empty((N, 3, 3), dtype=np.float64)
        self.B1 = np.empty((N, 3, 3), dtype=np.float64)
        self.B2 = np.empty((N, 3, 3), dtype=np.float64)
        self.N13 = np.empty((N, 3, 3), dtype=np.float64)
        self.N24 = np.empty((N, 3, 3), dtype=np.float64)
        self.N11 = np.empty((N, 3, 3), dtype=np.float64)
        self.N22 = np.empty((N, 3, 3), dtype=np.float64)
        self.N101 = np.empty((N, 3, 3), dtype=np.float64)
        self.N202 = np.empty((N, 3, 3), dtype=np.float64)

        self.temp_block_3d = np.empty((N, 3, 3), dtype=np.float64)

    def get_strain(self, state: RobotState) -> np.ndarray:
        n0p, n1p, n2p, n3p = self._get_node_pos(state.q)
        np.subtract(n1p, n0p, out=self.m_e0)
        np.subtract(n2p, n0p, out=self.m_e1)
        np.subtract(n3p, n0p, out=self.m_e2)

        self.cross_e0_e1[:] = np.cross(self.m_e0, self.m_e1)
        self.cross_e0_e2[:] = np.cross(
            self.m_e2, self.m_e0)  # Reusing space, wrong name

        self.temp_block_2d[:] = np.cross(self.cross_e0_e1, self.cross_e0_e2)

        norm_w = np.linalg.norm(self.temp_block_2d, axis=1, keepdims=True)
        dot_uv = np.sum(self.cross_e0_e1 * self.cross_e0_e2,
                        axis=1, keepdims=True)

        angle = np.arctan2(norm_w, dot_uv)
        sign = np.sign(
            np.sum(self.m_e0 * self.temp_block_2d, axis=1, keepdims=True))

        return (angle * sign).squeeze(1)

    def grad_hess_strain(self, state: RobotState) -> typing.Tuple[np.ndarray, np.ndarray]:
        n0p, n1p, n2p, n3p = self._get_node_pos(state.q)

        # Compute edges
        np.subtract(n1p, n0p, out=self.m_e0)
        np.subtract(n2p, n0p, out=self.m_e1)
        np.subtract(n3p, n0p, out=self.m_e2)
        np.subtract(n2p, n1p, out=self.m_e3)
        np.subtract(n3p, n1p, out=self.m_e4)

        # Compute norms
        self.norm_e0[:] = np.linalg.norm(self.m_e0, axis=-1, keepdims=True)
        self.norm_e1[:] = np.linalg.norm(self.m_e1, axis=-1, keepdims=True)
        self.norm_e2[:] = np.linalg.norm(self.m_e2, axis=-1, keepdims=True)
        self.norm_e3[:] = np.linalg.norm(self.m_e3, axis=-1, keepdims=True)
        self.norm_e4[:] = np.linalg.norm(self.m_e4, axis=-1, keepdims=True)

        # Compute cosine terms
        np.sum(self.m_e0 * self.m_e1, axis=-1, keepdims=True, out=self.m_cosA1)
        np.divide(self.m_cosA1, self.norm_e0 * self.norm_e1, out=self.m_cosA1)

        np.sum(self.m_e0 * self.m_e2, axis=-1, keepdims=True, out=self.m_cosA2)
        np.divide(self.m_cosA2, self.norm_e0 * self.norm_e2, out=self.m_cosA2)

        np.sum(-self.m_e0 * self.m_e3, axis=-1,
               keepdims=True, out=self.m_cosA3)
        np.divide(self.m_cosA3, self.norm_e0 * self.norm_e3, out=self.m_cosA3)

        np.sum(-self.m_e0 * self.m_e4, axis=-1,
               keepdims=True, out=self.m_cosA4)
        np.divide(self.m_cosA4, self.norm_e0 * self.norm_e4, out=self.m_cosA4)

        # Compute sine terms
        self.cross_e0_e1[:] = np.cross(self.m_e0, self.m_e1, axis=-1)
        self.cross_e0_e2[:] = np.cross(self.m_e0, self.m_e2, axis=-1)
        self.cross_e0_e3[:] = np.cross(self.m_e0, self.m_e3, axis=-1)
        self.cross_e0_e4[:] = np.cross(self.m_e0, self.m_e4, axis=-1)

        self.m_sinA1[:] = np.linalg.norm(
            self.cross_e0_e1, axis=-1, keepdims=True)
        np.divide(self.m_sinA1, self.norm_e0 * self.norm_e1, out=self.m_sinA1)

        self.m_sinA2[:] = np.linalg.norm(
            self.cross_e0_e2, axis=-1, keepdims=True)
        np.divide(self.m_sinA2, self.norm_e0 * self.norm_e2, out=self.m_sinA2)

        self.m_sinA3[:] = np.linalg.norm(
            self.cross_e0_e3, axis=-1, keepdims=True)
        np.divide(self.m_sinA3, -self.norm_e0 * self.norm_e3, out=self.m_sinA3)

        self.m_sinA4[:] = np.linalg.norm(
            self.cross_e0_e4, axis=-1, keepdims=True)
        np.divide(self.m_sinA4, -self.norm_e0 * self.norm_e4, out=self.m_sinA4)

        # Compute normals
        self.m_nn1[:] = np.cross(self.m_e0, self.m_e3, axis=-1)
        nn1_norm = np.linalg.norm(self.m_nn1, axis=-1, keepdims=True)
        np.divide(self.m_nn1, nn1_norm, out=self.m_nn1)

        self.m_nn2[:] = -np.cross(self.m_e0, self.m_e4, axis=-1)
        nn2_norm = np.linalg.norm(self.m_nn2, axis=-1, keepdims=True)
        np.divide(self.m_nn2, nn2_norm, out=self.m_nn2)

        # Compute h terms
        np.multiply(self.norm_e0, self.m_sinA1, out=self.m_h1)
        np.multiply(self.norm_e0, self.m_sinA2, out=self.m_h2)
        np.multiply(-self.norm_e0, self.m_sinA3, out=self.m_h3)
        np.multiply(-self.norm_e0, self.m_sinA4, out=self.m_h4)
        np.multiply(self.norm_e1, self.m_sinA1, out=self.m_h01)
        np.multiply(self.norm_e2, self.m_sinA2, out=self.m_h02)

        # Gradient computation
        self.grad_theta[:, 0:3] = (
            self.m_cosA3 * self.m_nn1 / self.m_h3) + (self.m_cosA4 * self.m_nn2 / self.m_h4)
        self.grad_theta[:, 3:6] = (
            self.m_cosA1 * self.m_nn1 / self.m_h1) + (self.m_cosA2 * self.m_nn2 / self.m_h2)
        self.grad_theta[:, 6:9] = -self.m_nn1 / self.m_h01
        self.grad_theta[:, 9:12] = -self.m_nn2 / self.m_h02

        # Intermediate vectors for Hessian
        self.m_m1[:] = np.cross(self.m_nn1, self.m_e1, axis=-1)
        np.divide(self.m_m1, self.norm_e1, out=self.m_m1)
        self.m_m2[:] = -np.cross(self.m_nn2, self.m_e2, axis=-1)
        np.divide(self.m_m2, self.norm_e2, out=self.m_m2)
        self.m_m3[:] = -np.cross(self.m_nn1, self.m_e3, axis=-1)
        np.divide(self.m_m3, self.norm_e3, out=self.m_m3)
        self.m_m4[:] = np.cross(self.m_nn2, self.m_e4, axis=-1)
        np.divide(self.m_m4, self.norm_e4, out=self.m_m4)
        self.m_m01[:] = -np.cross(self.m_nn1, self.m_e0, axis=-1)
        np.divide(self.m_m01, self.norm_e0, out=self.m_m01)
        self.m_m02[:] = np.cross(self.m_nn2, self.m_e0, axis=-1)
        np.divide(self.m_m02, self.norm_e0, out=self.m_m02)

        # Hessian computation
        def compute_block(coefficient, vec1, vec2, out):
            np.einsum('ni,nj->nij', vec1, vec2, out=out)
            np.multiply(out, coefficient[..., None], out=out)

        # Compute M blocks
        compute_block(self.m_cosA3/(self.m_h3**2),
                      self.m_m3, self.m_nn1, self.M331)
        compute_block(self.m_cosA3/(self.m_h3*self.m_h1),
                      self.m_m1, self.m_nn1, self.M311)
        compute_block(self.m_cosA1/(self.m_h1*self.m_h3),
                      self.m_m3, self.m_nn1, self.M131)
        compute_block(self.m_cosA3/(self.m_h3*self.m_h01),
                      self.m_m01, self.m_nn1, self.M3011)
        compute_block(self.m_cosA1/(self.m_h1**2),
                      self.m_m1, self.m_nn1, self.M111)
        compute_block(self.m_cosA1/(self.m_h1*self.m_h01),
                      self.m_m01, self.m_nn1, self.M1011)

        compute_block(self.m_cosA4/(self.m_h4**2),
                      self.m_m4, self.m_nn2, self.M442)
        compute_block(self.m_cosA4/(self.m_h4*self.m_h2),
                      self.m_m2, self.m_nn2, self.M422)
        compute_block(self.m_cosA2/(self.m_h2*self.m_h4),
                      self.m_m4, self.m_nn2, self.M242)
        compute_block(self.m_cosA4/(self.m_h4*self.m_h02),
                      self.m_m02, self.m_nn2, self.M4022)
        compute_block(self.m_cosA2/(self.m_h2**2),
                      self.m_m2, self.m_nn2, self.M222)
        compute_block(self.m_cosA2/(self.m_h2*self.m_h02),
                      self.m_m02, self.m_nn2, self.M2022)

        # Compute B blocks
        compute_block(1.0/(self.norm_e0**2), self.m_nn1, self.m_m01, self.B1)
        compute_block(1.0/(self.norm_e0**2), self.m_nn2, self.m_m02, self.B2)

        # Compute N blocks
        compute_block(1.0/(self.m_h01*self.m_h3),
                      self.m_nn1, self.m_m3, self.N13)
        compute_block(1.0/(self.m_h02*self.m_h4),
                      self.m_nn2, self.m_m4, self.N24)
        compute_block(1.0/(self.m_h01*self.m_h1),
                      self.m_nn1, self.m_m1, self.N11)
        compute_block(1.0/(self.m_h02*self.m_h2),
                      self.m_nn2, self.m_m2, self.N22)
        compute_block(1.0/(self.m_h01**2), self.m_nn1, self.m_m01, self.N101)
        compute_block(1.0/(self.m_h02**2), self.m_nn2, self.m_m02, self.N202)

        # Assemble Hessian
        self.hess_theta.fill(0.0)

        np.add(self.M331, np.swapaxes(self.M331, -1, -2), out=self.M331)
        np.subtract(self.M331, self.B1, out=self.hess_theta[:, :3, :3])
        np.add(self.M442, np.swapaxes(self.M442, -1, -2), out=self.M442)
        np.subtract(self.M442, self.B2, out=self.temp_block_3d)
        np.add(self.hess_theta[:, :3, :3], self.temp_block_3d,
               out=self.hess_theta[:, :3, :3])
        np.add(self.M311, np.swapaxes(self.M131, -1, -2), out=self.temp_block_3d)
        np.add(self.temp_block_3d, self.B1, out=self.hess_theta[:, :3, 3:6])
        np.add(self.M422, np.swapaxes(self.M242, -1, -2), out=self.temp_block_3d)
        np.add(self.temp_block_3d, self.B2, out=self.temp_block_3d)
        np.add(self.hess_theta[:, :3, 3:6],
               self.temp_block_3d, out=self.hess_theta[:, :3, 3:6])
        np.subtract(self.M3011, self.N13, out=self.hess_theta[:, :3, 6:9])
        np.subtract(self.M4022, self.N24, out=self.hess_theta[:, :3, 9:12])
        np.add(self.M111, np.swapaxes(self.M111, -1, -2), out=self.M111)
        np.subtract(self.M111, self.B1, out=self.hess_theta[:, 3:6, 3:6])
        np.add(self.M222, np.swapaxes(self.M222, -1, -2), out=self.M222)
        np.subtract(self.M222, self.B2, out=self.temp_block_3d)
        np.add(self.hess_theta[:, 3:6, 3:6],
               self.temp_block_3d, out=self.hess_theta[:, 3:6, 3:6])
        np.subtract(self.M1011, self.N11, out=self.hess_theta[:, 3:6, 6:9])
        np.subtract(self.M2022, self.N22, out=self.hess_theta[:, 3:6, 9:12])
        np.add(self.N101, np.swapaxes(self.N101, -1, -2), out=self.N101)
        np.multiply(self.N101, -1.0, out=self.hess_theta[:, 6:9, 6:9])
        np.add(self.N202, np.swapaxes(self.N202, -1, -2), out=self.N202)
        np.multiply(self.N202, -1.0, out=self.hess_theta[:, 9:12, 9:12])

        # Fill symmetric blocks
        self.hess_theta[:, 3:6, :3] = np.swapaxes(
            self.hess_theta[:, :3, 3:6], -1, -2)
        self.hess_theta[:, 6:9, :3] = np.swapaxes(
            self.hess_theta[:, :3, 6:9], -1, -2)
        self.hess_theta[:, 9:12, :3] = np.swapaxes(
            self.hess_theta[:, :3, 9:12], -1, -2)
        self.hess_theta[:, 6:9, 3:6] = np.swapaxes(
            self.hess_theta[:, 3:6, 6:9], -1, -2)
        self.hess_theta[:, 9:12, 3:6] = np.swapaxes(
            self.hess_theta[:, 3:6, 9:12], -1, -2)

        return self.grad_theta, self.hess_theta
