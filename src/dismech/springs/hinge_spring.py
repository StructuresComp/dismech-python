import numpy as np


class HingeSpring:
    """
                 x2
                 /\
                /  \
             e1/    \e3
              /  t0  \
             /        \
            /    e0    \
          x0------------x1
            \          /
             \   t1   /
              \      /
             e2\    /e4
                \  /
                 \/
                 x3

        Edge orientation: e0,e1,e2 point away from x0
                             e3,e4 point away from x1
    """

    def __init__(self, indices, robot, kb=None):
        self.kb = kb or robot.kb
        self.nodes_ind = indices
        self.ind = np.concatenate([robot.map_node_to_dof(i)
                                  for i in indices], axis=0)
        self.theta_bar = self.get_theta(*np.split(robot.q[self.ind], 4))

    @staticmethod
    def get_theta(n0, n1, n2, n3):
        m_e0 = n1 - n0
        m_e1 = n2 - n0
        m_e2 = n3 - n0

        c0 = np.cross(m_e0, m_e1)
        c1 = np.cross(m_e2, m_e0)

        w = np.cross(c0, c1)

        if n0.ndim == 2:
            norm_w = np.linalg.norm(w, axis=1, keepdims=True)
            dot_uv = np.sum(c0 * c1, axis=1, keepdims=True)

            angle = np.arctan2(norm_w, dot_uv)
            sign = np.sign(np.sum(m_e0 * w, axis=1, keepdims=True))

            return (angle * sign).squeeze(1)
        elif n0.ndim == 1:
            angle = np.atan2(np.linalg.norm(w), np.dot(c0, c1))
            return -angle if np.dot(m_e0, w) < 0 else angle
        else:
            raise ValueError("{} should be 1 or 2 dimensions".format(n0.ndim))

    @staticmethod
    def get_grad_theta(x0, x1, x2, x3):
        # Compute edge vectors
        m_e0 = x1 - x0
        m_e1 = x2 - x0
        m_e2 = x3 - x0
        m_e3 = x2 - x1
        m_e4 = x3 - x1

        # Precompute norms of edges
        norm_e0 = np.linalg.norm(m_e0, axis=1, keepdims=True)
        norm_e1 = np.linalg.norm(m_e1, axis=1, keepdims=True)
        norm_e2 = np.linalg.norm(m_e2, axis=1, keepdims=True)
        norm_e3 = np.linalg.norm(m_e3, axis=1, keepdims=True)
        norm_e4 = np.linalg.norm(m_e4, axis=1, keepdims=True)

        # Compute cosine terms using vectorized operations
        m_cosA1 = np.sum(m_e0 * m_e1, axis=1, keepdims=True) / \
            (norm_e0 * norm_e1)
        m_cosA2 = np.sum(m_e0 * m_e2, axis=1, keepdims=True) / \
            (norm_e0 * norm_e2)
        m_cosA3 = -np.sum(m_e0 * m_e3, axis=1,
                          keepdims=True) / (norm_e0 * norm_e3)
        m_cosA4 = -np.sum(m_e0 * m_e4, axis=1,
                          keepdims=True) / (norm_e0 * norm_e4)

        # Compute sine terms using cross products
        cross_e0_e1 = np.cross(m_e0, m_e1)
        m_sinA1 = np.linalg.norm(
            cross_e0_e1, axis=1, keepdims=True) / (norm_e0 * norm_e1)

        cross_e0_e2 = np.cross(m_e0, m_e2)
        m_sinA2 = np.linalg.norm(
            cross_e0_e2, axis=1, keepdims=True) / (norm_e0 * norm_e2)

        cross_e0_e3 = np.cross(m_e0, m_e3)
        m_sinA3 = -np.linalg.norm(cross_e0_e3, axis=1,
                                  keepdims=True) / (norm_e0 * norm_e3)

        cross_e0_e4 = np.cross(m_e0, m_e4)
        m_sinA4 = -np.linalg.norm(cross_e0_e4, axis=1,
                                  keepdims=True) / (norm_e0 * norm_e4)

        # Compute height terms
        m_h1 = norm_e0 * m_sinA1
        m_h2 = norm_e0 * m_sinA2
        m_h3 = -norm_e0 * m_sinA3
        m_h4 = -norm_e0 * m_sinA4
        m_h01 = norm_e1 * m_sinA1
        m_h02 = norm_e2 * m_sinA2

        # Compute normal vectors with safe normalization
        m_nn1 = np.cross(m_e0, m_e3)
        norm_nn1 = np.linalg.norm(m_nn1, axis=1, keepdims=True)
        mask_nn1 = norm_nn1 < 1e-6
        m_nn1 = np.where(mask_nn1, 0.0, m_nn1 / norm_nn1)

        m_nn2 = -np.cross(m_e0, m_e4)
        norm_nn2 = np.linalg.norm(m_nn2, axis=1, keepdims=True)
        mask_nn2 = norm_nn2 < 1e-6
        m_nn2 = np.where(mask_nn2, 0.0, m_nn2 / norm_nn2)

        # Prepare error checking
        norm_nn1_sq = norm_nn1.squeeze()
        norm_nn2_sq = norm_nn2.squeeze()
        h_masks = [m_h3.squeeze() == 0, m_h4.squeeze() == 0,
                   m_h1.squeeze() == 0, m_h2.squeeze() == 0,
                   m_h01.squeeze() == 0, m_h02.squeeze() == 0]
        error_conditions = [
            h_masks[0] & (norm_nn1_sq >= 1e-6),
            h_masks[1] & (norm_nn2_sq >= 1e-6),
            h_masks[2] & (norm_nn1_sq >= 1e-6),
            h_masks[3] & (norm_nn2_sq >= 1e-6),
            h_masks[4] & (norm_nn1_sq >= 1e-6),
            h_masks[5] & (norm_nn2_sq >= 1e-6)
        ]

        if any(np.any(cond) for cond in error_conditions):
            raise ValueError("Division by zero in gradient computation")

        # Compute gradient components with safe division
        t11 = np.where(h_masks[0][:, None], 0.0, (m_cosA3 * m_nn1) / m_h3)
        t12 = np.where(h_masks[1][:, None], 0.0, (m_cosA4 * m_nn2) / m_h4)
        t21 = np.where(h_masks[2][:, None], 0.0, (m_cosA1 * m_nn1) / m_h1)
        t22 = np.where(h_masks[3][:, None], 0.0, (m_cosA2 * m_nn2) / m_h2)
        t31 = np.where(h_masks[4][:, None], 0.0, m_nn1 / m_h01)
        t41 = np.where(h_masks[5][:, None], 0.0, m_nn2 / m_h02)

        # Assemble final gradient tensor
        gradTheta = np.zeros((x0.shape[0], 12), dtype=np.float64)
        gradTheta[:, 0:3] = t11 + t12
        gradTheta[:, 3:6] = t21 + t22
        gradTheta[:, 6:9] = -t31
        gradTheta[:, 9:12] = -t41

        return gradTheta

    @staticmethod
    def get_grad_hess_theta(x0, x1, x2, x3):
        # All inputs are (N, 3) numpy arrays
        N = x0.shape[0]

        # Compute edges
        m_e0 = x1 - x0
        m_e1 = x2 - x0
        m_e2 = x3 - x0
        m_e3 = x2 - x1
        m_e4 = x3 - x1

        # Compute norms
        norm_e0 = np.linalg.norm(m_e0, axis=-1, keepdims=True)
        norm_e1 = np.linalg.norm(m_e1, axis=-1, keepdims=True)
        norm_e2 = np.linalg.norm(m_e2, axis=-1, keepdims=True)
        norm_e3 = np.linalg.norm(m_e3, axis=-1, keepdims=True)
        norm_e4 = np.linalg.norm(m_e4, axis=-1, keepdims=True)

        # Compute cosine terms
        m_cosA1 = np.sum(m_e0 * m_e1, axis=-1, keepdims=True) / \
            (norm_e0 * norm_e1)
        m_cosA2 = np.sum(m_e0 * m_e2, axis=-1, keepdims=True) / \
            (norm_e0 * norm_e2)
        m_cosA3 = -np.sum(m_e0 * m_e3, axis=-1,
                          keepdims=True) / (norm_e0 * norm_e3)
        m_cosA4 = -np.sum(m_e0 * m_e4, axis=-1,
                          keepdims=True) / (norm_e0 * norm_e4)

        # Compute sine terms
        cross_e0_e1 = np.cross(m_e0, m_e1, axis=-1)
        cross_e0_e2 = np.cross(m_e0, m_e2, axis=-1)
        cross_e0_e3 = np.cross(m_e0, m_e3, axis=-1)
        cross_e0_e4 = np.cross(m_e0, m_e4, axis=-1)

        m_sinA1 = np.linalg.norm(cross_e0_e1, axis=-1,
                                 keepdims=True) / (norm_e0 * norm_e1)
        m_sinA2 = np.linalg.norm(cross_e0_e2, axis=-1,
                                 keepdims=True) / (norm_e0 * norm_e2)
        m_sinA3 = -np.linalg.norm(cross_e0_e3, axis=-1,
                                  keepdims=True) / (norm_e0 * norm_e3)
        m_sinA4 = -np.linalg.norm(cross_e0_e4, axis=-1,
                                  keepdims=True) / (norm_e0 * norm_e4)

        # Compute normals
        m_nn1 = np.cross(m_e0, m_e3, axis=-1)
        m_nn1_norm = np.linalg.norm(m_nn1, axis=-1, keepdims=True)
        m_nn1 = m_nn1 / m_nn1_norm

        m_nn2 = -np.cross(m_e0, m_e4, axis=-1)
        m_nn2_norm = np.linalg.norm(m_nn2, axis=-1, keepdims=True)
        m_nn2 = m_nn2 / m_nn2_norm

        # Compute h terms
        m_h1 = norm_e0 * m_sinA1
        m_h2 = norm_e0 * m_sinA2
        m_h3 = -norm_e0 * m_sinA3
        m_h4 = -norm_e0 * m_sinA4
        m_h01 = norm_e1 * m_sinA1
        m_h02 = norm_e2 * m_sinA2

        # Gradient computation
        gradTheta = np.zeros((N, 12))
        gradTheta[:, 0:3] = (m_cosA3 * m_nn1 / m_h3) + (m_cosA4 * m_nn2 / m_h4)
        gradTheta[:, 3:6] = (m_cosA1 * m_nn1 / m_h1) + (m_cosA2 * m_nn2 / m_h2)
        gradTheta[:, 6:9] = -m_nn1 / m_h01
        gradTheta[:, 9:12] = -m_nn2 / m_h02

        # Intermediate vectors for Hessian
        m_m1 = np.cross(m_nn1, m_e1, axis=-1) / norm_e1
        m_m2 = -np.cross(m_nn2, m_e2, axis=-1) / norm_e2
        m_m3 = -np.cross(m_nn1, m_e3, axis=-1) / norm_e3
        m_m4 = np.cross(m_nn2, m_e4, axis=-1) / norm_e4
        m_m01 = -np.cross(m_nn1, m_e0, axis=-1) / norm_e0
        m_m02 = np.cross(m_nn2, m_e0, axis=-1) / norm_e0

        # Helper function for M + M^T
        def MMT(mat):
            return mat + np.swapaxes(mat, -1, -2)

        # Compute Hessian components
        M331 = (m_cosA3 / (m_h3**2))[..., None] * \
            np.einsum('ni,nj->nij', m_m3, m_nn1)
        M311 = (m_cosA3 / (m_h3 * m_h1))[..., None] * \
            np.einsum('ni,nj->nij', m_m1, m_nn1)
        M131 = (m_cosA1 / (m_h1 * m_h3))[..., None] * \
            np.einsum('ni,nj->nij', m_m3, m_nn1)
        M3011 = (m_cosA3 / (m_h3 * m_h01)
                 )[..., None] * np.einsum('ni,nj->nij', m_m01, m_nn1)
        M111 = (m_cosA1 / (m_h1**2))[..., None] * \
            np.einsum('ni,nj->nij', m_m1, m_nn1)
        M1011 = (m_cosA1 / (m_h1 * m_h01)
                 )[..., None] * np.einsum('ni,nj->nij', m_m01, m_nn1)

        M442 = (m_cosA4 / (m_h4**2))[..., None] * \
            np.einsum('ni,nj->nij', m_m4, m_nn2)
        M422 = (m_cosA4 / (m_h4 * m_h2))[..., None] * \
            np.einsum('ni,nj->nij', m_m2, m_nn2)
        M242 = (m_cosA2 / (m_h2 * m_h4))[..., None] * \
            np.einsum('ni,nj->nij', m_m4, m_nn2)
        M4022 = (m_cosA4 / (m_h4 * m_h02)
                 )[..., None] * np.einsum('ni,nj->nij', m_m02, m_nn2)
        M222 = (m_cosA2 / (m_h2**2))[..., None] * \
            np.einsum('ni,nj->nij', m_m2, m_nn2)
        M2022 = (m_cosA2 / (m_h2 * m_h02)
                 )[..., None] * np.einsum('ni,nj->nij', m_m02, m_nn2)

        B1 = (1 / (norm_e0**2))[..., None] * \
            np.einsum('ni,nj->nij', m_nn1, m_m01)
        B2 = (1 / (norm_e0**2))[..., None] * \
            np.einsum('ni,nj->nij', m_nn2, m_m02)

        N13 = (1 / (m_h01 * m_h3))[..., None] * \
            np.einsum('ni,nj->nij', m_nn1, m_m3)
        N24 = (1 / (m_h02 * m_h4))[..., None] * \
            np.einsum('ni,nj->nij', m_nn2, m_m4)
        N11 = (1 / (m_h01 * m_h1))[..., None] * \
            np.einsum('ni,nj->nij', m_nn1, m_m1)
        N22 = (1 / (m_h02 * m_h2))[..., None] * \
            np.einsum('ni,nj->nij', m_nn2, m_m2)
        N101 = (1 / (m_h01**2))[..., None] * \
            np.einsum('ni,nj->nij', m_nn1, m_m01)
        N202 = (1 / (m_h02**2))[..., None] * \
            np.einsum('ni,nj->nij', m_nn2, m_m02)

        # Initialize Hessian
        hessTheta = np.zeros((N, 12, 12))

        # Fill Hessian blocks
        hessTheta[:, 0:3, 0:3] = MMT(M331) - B1 + MMT(M442) - B2
        hessTheta[:, 0:3, 3:6] = M311 + \
            np.swapaxes(M131, -1, -2) + B1 + M422 + \
            np.swapaxes(M242, -1, -2) + B2
        hessTheta[:, 0:3, 6:9] = M3011 - N13
        hessTheta[:, 0:3, 9:12] = M4022 - N24

        hessTheta[:, 3:6, 3:6] = MMT(M111) - B1 + MMT(M222) - B2
        hessTheta[:, 3:6, 6:9] = M1011 - N11
        hessTheta[:, 3:6, 9:12] = M2022 - N22

        hessTheta[:, 6:9, 6:9] = -MMT(N101)
        hessTheta[:, 9:12, 9:12] = -MMT(N202)

        # Fill symmetric parts
        hessTheta[:, 3:6, 0:3] = np.swapaxes(hessTheta[:, 0:3, 3:6], -1, -2)
        hessTheta[:, 6:9, 0:3] = np.swapaxes(hessTheta[:, 0:3, 6:9], -1, -2)
        hessTheta[:, 9:12, 0:3] = np.swapaxes(hessTheta[:, 0:3, 9:12], -1, -2)
        hessTheta[:, 6:9, 3:6] = np.swapaxes(hessTheta[:, 3:6, 6:9], -1, -2)
        hessTheta[:, 9:12, 3:6] = np.swapaxes(hessTheta[:, 3:6, 9:12], -1, -2)

        return gradTheta, hessTheta
