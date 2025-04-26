import typing
import numpy as np

from .elastic_energy import ElasticEnergy
from ..state import RobotState
from ..springs import HingeSprings


class HingeEnergy(ElasticEnergy):
    def __init__(self, springs: HingeSprings, initial_state: RobotState):
        super().__init__(springs, initial_state)

    @property
    def K(self):
        return self._springs.kb

    def get_strain(self, state: RobotState) -> np.ndarray:
        n0p, n1p, n2p, n3p = self._get_node_pos(state.q)
        
        m_e0 = n1p - n0p
        m_e1 = n2p - n0p
        m_e2 = n3p - n0p

        cross_e0_e1 = np.cross(m_e0, m_e1)
        cross_e0_e2 = np.cross(m_e2, m_e0)

        temp_block_2d = np.cross(cross_e0_e1, cross_e0_e2)

        norm_w = np.linalg.norm(temp_block_2d, axis=1, keepdims=True)
        dot_uv = np.sum(cross_e0_e1 * cross_e0_e2, axis=1, keepdims=True)

        angle = np.arctan2(norm_w, dot_uv)
        sign = np.sign(np.sum(m_e0 * temp_block_2d, axis=1, keepdims=True))

        return (angle * sign).squeeze(1)

    def grad_hess_strain(self, state: RobotState) -> typing.Tuple[np.ndarray, np.ndarray]:
        n0p, n1p, n2p, n3p = self._get_node_pos(state.q)
        N = n0p.shape[0]

        # Compute edges
        m_e0 = n1p - n0p
        m_e1 = n2p - n0p
        m_e2 = n3p - n0p
        m_e3 = n2p - n1p
        m_e4 = n3p - n1p

        # Compute norms
        norm_e0 = np.linalg.norm(m_e0, axis=-1, keepdims=True)
        norm_e1 = np.linalg.norm(m_e1, axis=-1, keepdims=True)
        norm_e2 = np.linalg.norm(m_e2, axis=-1, keepdims=True)
        norm_e3 = np.linalg.norm(m_e3, axis=-1, keepdims=True)
        norm_e4 = np.linalg.norm(m_e4, axis=-1, keepdims=True)

        # Compute cosine terms
        cosA1 = np.sum(m_e0 * m_e1, axis=-1, keepdims=True) / (norm_e0 * norm_e1)
        cosA2 = np.sum(m_e0 * m_e2, axis=-1, keepdims=True) / (norm_e0 * norm_e2)
        cosA3 = np.sum(-m_e0 * m_e3, axis=-1, keepdims=True) / (norm_e0 * norm_e3)
        cosA4 = np.sum(-m_e0 * m_e4, axis=-1, keepdims=True) / (norm_e0 * norm_e4)

        # Compute sine terms
        cross_e0_e1 = np.cross(m_e0, m_e1, axis=-1)
        sinA1 = np.linalg.norm(cross_e0_e1, axis=-1, keepdims=True) / (norm_e0 * norm_e1)
        cross_e0_e2 = np.cross(m_e0, m_e2, axis=-1)
        sinA2 = np.linalg.norm(cross_e0_e2, axis=-1, keepdims=True) / (norm_e0 * norm_e2)
        cross_e0_e3 = np.cross(m_e0, m_e3, axis=-1)
        sinA3 = np.linalg.norm(cross_e0_e3, axis=-1, keepdims=True) / (-norm_e0 * norm_e3)
        cross_e0_e4 = np.cross(m_e0, m_e4, axis=-1)
        sinA4 = np.linalg.norm(cross_e0_e4, axis=-1, keepdims=True) / (-norm_e0 * norm_e4)

        # Compute normals
        nn1 = np.cross(m_e0, m_e3, axis=-1)
        nn1 /= np.linalg.norm(nn1, axis=-1, keepdims=True)
        nn2 = -np.cross(m_e0, m_e4, axis=-1)
        nn2 /= np.linalg.norm(nn2, axis=-1, keepdims=True)

        # Compute h terms
        h1 = norm_e0 * sinA1
        h2 = norm_e0 * sinA2
        h3 = -norm_e0 * sinA3
        h4 = -norm_e0 * sinA4
        h01 = norm_e1 * sinA1
        h02 = norm_e2 * sinA2

        # Gradient computation
        grad_theta = np.zeros((N, 12), dtype=np.float64)
        grad_theta[:, 0:3] = (cosA3 * nn1 / h3) + (cosA4 * nn2 / h4)
        grad_theta[:, 3:6] = (cosA1 * nn1 / h1) + (cosA2 * nn2 / h2)
        grad_theta[:, 6:9] = -nn1 / h01
        grad_theta[:, 9:12] = -nn2 / h02

        # Intermediate vectors for Hessian
        m_m1 = np.cross(nn1, m_e1, axis=-1) / norm_e1
        m_m2 = -np.cross(nn2, m_e2, axis=-1) / norm_e2
        m_m3 = -np.cross(nn1, m_e3, axis=-1) / norm_e3
        m_m4 = np.cross(nn2, m_e4, axis=-1) / norm_e4
        m_m01 = -np.cross(nn1, m_e0, axis=-1) / norm_e0
        m_m02 = np.cross(nn2, m_e0, axis=-1) / norm_e0

        def compute_block(coefficient, vec1, vec2):
            return coefficient[..., None] * np.einsum('ni,nj->nij', vec1, vec2)

        # Compute M blocks
        M331 = compute_block(cosA3/(h3**2), m_m3, nn1)
        M311 = compute_block(cosA3/(h3 * h1), m_m1, nn1)
        M131 = compute_block(cosA1/(h1 * h3), m_m3, nn1)
        M3011 = compute_block(cosA3/(h3 * h01), m_m01, nn1)
        M111 = compute_block(cosA1/(h1**2), m_m1, nn1)
        M1011 = compute_block(cosA1/(h1 * h01), m_m01, nn1)
        M442 = compute_block(cosA4/(h4**2), m_m4, nn2)
        M422 = compute_block(cosA4/(h4 * h2), m_m2, nn2)
        M242 = compute_block(cosA2/(h2 * h4), m_m4, nn2)
        M4022 = compute_block(cosA4/(h4 * h02), m_m02, nn2)
        M222 = compute_block(cosA2/(h2**2), m_m2, nn2)
        M2022 = compute_block(cosA2/(h2 * h02), m_m02, nn2)

        # Compute B blocks
        B1 = compute_block(1.0/(norm_e0**2), nn1, m_m01)
        B2 = compute_block(1.0/(norm_e0**2), nn2, m_m02)

        # Compute N blocks
        N13 = compute_block(1.0/(h01 * h3), nn1, m_m3)
        N24 = compute_block(1.0/(h02 * h4), nn2, m_m4)
        N11 = compute_block(1.0/(h01 * h1), nn1, m_m1)
        N22 = compute_block(1.0/(h02 * h2), nn2, m_m2)
        N101 = compute_block(1.0/(h01**2), nn1, m_m01)
        N202 = compute_block(1.0/(h02**2), nn2, m_m02)

        # Assemble Hessian
        hess_theta = np.zeros((N, 12, 12), dtype=np.float64)
        hess_theta[:, :3, :3] = (M331 + np.swapaxes(M331, -1, -2) - B1) + (M442 + np.swapaxes(M442, -1, -2) - B2)
        hess_theta[:, :3, 3:6] = (M311 + np.swapaxes(M131, -1, -2) + B1) + (M422 + np.swapaxes(M242, -1, -2) + B2)
        hess_theta[:, :3, 6:9] = M3011 - N13
        hess_theta[:, :3, 9:12] = M4022 - N24
        hess_theta[:, 3:6, 3:6] = (M111 + np.swapaxes(M111, -1, -2) - B1) + (M222 + np.swapaxes(M222, -1, -2) - B2)
        hess_theta[:, 3:6, 6:9] = M1011 - N11
        hess_theta[:, 3:6, 9:12] = M2022 - N22
        hess_theta[:, 6:9, 6:9] = -(N101 + np.swapaxes(N101, -1, -2))
        hess_theta[:, 9:12, 9:12] = -(N202 + np.swapaxes(N202, -1, -2))

        # Fill symmetric blocks
        hess_theta[:, 3:6, :3] = np.swapaxes(hess_theta[:, :3, 3:6], -1, -2)
        hess_theta[:, 6:9, :3] = np.swapaxes(hess_theta[:, :3, 6:9], -1, -2)
        hess_theta[:, 9:12, :3] = np.swapaxes(hess_theta[:, :3, 9:12], -1, -2)
        hess_theta[:, 6:9, 3:6] = np.swapaxes(hess_theta[:, 3:6, 6:9], -1, -2)
        hess_theta[:, 9:12, 3:6] = np.swapaxes(hess_theta[:, 3:6, 9:12], -1, -2)

        return grad_theta, hess_theta