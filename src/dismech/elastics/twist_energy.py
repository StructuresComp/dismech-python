import numpy as np
import typing

from ..springs import BendTwistSpring
from .elastic_energy import ElasticEnergy


class TwistEnergy(ElasticEnergy):
    def __init__(self, springs: typing.List[BendTwistSpring]):
        super().__init__(np.array([s.stiff_GJ / s.voronoi_len for s in springs]),
                         np.array([s.undef_ref_twist for s in springs]),
                         np.array([s.nodes_ind for s in springs]),
                         np.array([s.ind for s in springs]))
        self._sgn = np.array([s.sgn for s in springs])
        self._edges_ind = np.array([s.ind[-2:] for s in springs])

        N = len(springs)

        # Create sign matrices
        self._sign_grad = np.ones((N, 11))
        for dof_idx, signs in [(9, self._sgn[:,0]), (10, self._sgn[:,1])]:
            self._sign_grad[:, dof_idx] = signs
        self._sign_hess = self._sign_grad[:, :, None] * \
            self._sign_grad[:, None, :]

        self.theta_e = np.empty((N))
        self.theta_f = np.empty((N))

    def _set_thetas(self, q: np.ndarray):
        self.theta_e[:] = q[self._edges_ind[:, 0]] * self._sgn[:, 0]
        self.theta_f[:] = q[self._edges_ind[:, 1]] * self._sgn[:, 1]

    def get_strain(self, q: np.ndarray, **kwargs) -> np.ndarray:
        self._set_thetas(q)
        return self.theta_f - self.theta_e + kwargs['ref_twist']

    def grad_hess_strain(self, q: np.ndarray, **kwargs) -> typing.Tuple[np.ndarray, np.ndarray]:
        n0p, n1p, n2p = self._get_node_pos(q)
        N = n0p.shape[0]  # Number of springs in the batch

        # Edge vectors
        ee = n1p - n0p
        ef = n2p - n1p

        # Norms and tangents
        norm_e = np.linalg.norm(ee, axis=1, keepdims=True)
        norm_f = np.linalg.norm(ef, axis=1, keepdims=True)
        te = ee / norm_e
        tf = ef / norm_f

        # Dot product and chi
        dot_te_tf = np.sum(te * tf, axis=1, keepdims=True)
        chi = 1.0 + dot_te_tf

        # Curvature binormal
        kb = 2.0 * np.cross(te, tf, axis=1) / chi

        # tilde_t
        tilde_t = (te + tf) / chi

        # Initialize reduced gradTwist (N, 11)
        grad_twist = np.zeros((N, 11))
        grad_twist[:, 0:3] = (-0.5 / norm_e) * kb
        grad_twist[:, 6:9] = (0.5 / norm_f) * kb
        grad_twist[:, 3:6] = - (grad_twist[:, 0:3] + grad_twist[:, 6:9])
        grad_twist[:, 9] = -1.0
        grad_twist[:, 10] = 1.0

        def cross_mat_batch(v):
            """Batch version of cross product matrix"""
            zeros = np.zeros_like(v[:, 0])
            return np.array([
                [zeros, -v[:, 2], v[:, 1]],
                [v[:, 2], zeros, -v[:, 0]],
                [-v[:, 1], v[:, 0], zeros]
            ]).transpose(2, 0, 1)

        # Cross product matrices
        cross_te = cross_mat_batch(te)
        cross_tf = cross_mat_batch(tf)

        # Compute second derivatives
        norm2_e = norm_e ** 2
        norm2_f = norm_f ** 2
        norm_e_norm_f = norm_e * norm_f

        D2mDe2 = (-0.5 / norm2_e)[:, :, np.newaxis] * (
            np.einsum('ni,nj->nij', kb, te + tilde_t) +
            (2.0 / chi)[:, :, np.newaxis] * cross_tf
        )
        D2mDf2 = (-0.5 / norm2_f)[:, :, np.newaxis] * (
            np.einsum('ni,nj->nij', kb, tf + tilde_t) +
            (2.0 / chi)[:, :, np.newaxis] * cross_te
        )
        D2mDeDf = (0.5 / norm_e_norm_f)[:, :, np.newaxis] * (
            (2.0 / chi)[:, :, np.newaxis] * cross_te -
            np.einsum('ni,nj->nij', kb, tilde_t)
        )
        D2mDfDe = (0.5 / norm_e_norm_f)[:, :, np.newaxis] * (
            (-2.0 / chi)[:, :, np.newaxis] * cross_tf -
            np.einsum('ni,nj->nij', kb, tilde_t)
        )

        # Assemble reduced DDtwist (N, 11, 11)
        hess_twist = np.zeros((N, 11, 11))

        hess_twist[:, 0:3, 0:3] = D2mDe2
        hess_twist[:, 0:3, 3:6] = -D2mDe2 + D2mDeDf
        hess_twist[:, 3:6, 0:3] = -D2mDe2 + D2mDfDe
        hess_twist[:, 3:6, 3:6] = D2mDe2 - (D2mDeDf + D2mDfDe) + D2mDf2
        hess_twist[:, 0:3, 6:9] = -D2mDeDf
        hess_twist[:, 6:9, 0:3] = -D2mDfDe
        hess_twist[:, 6:9, 3:6] = D2mDfDe - D2mDf2
        hess_twist[:, 3:6, 6:9] = D2mDeDf - D2mDf2
        hess_twist[:, 6:9, 6:9] = D2mDf2
        return grad_twist, hess_twist
