import numpy as np
import typing

from ..springs import TwistSprings
from ..state import RobotState
from .elastic_energy import ElasticEnergy


class TwistEnergy(ElasticEnergy):
    def __init__(self, springs: TwistSprings, initial_state: RobotState):
        super().__init__(springs, initial_state)
        self._sgn = np.array([s.sgn for s in springs])
        self._edges_ind = np.array([s.ind[-2:] for s in springs])

        N = len(springs)

        # Create sign matrices
        self._sign_grad = np.ones((N, 11))
        for dof_idx, signs in [(9, self._sgn[:, 0]), (10, self._sgn[:, 1])]:
            self._sign_grad[:, dof_idx] = signs
        self._sign_hess = self._sign_grad[:, :, None] * \
            self._sign_grad[:, None, :]

    @property
    def K(self):
        return self._springs.GJ / self._springs.voronoi_len

    def _get_thetas(self, q: np.ndarray):
        return q[self._edges_ind[:, 0]] * self._sgn[:, 0], q[self._edges_ind[:, 1]] * self._sgn[:, 1]

    def get_strain(self, state: RobotState) -> np.ndarray:
        theta_e, theta_f = self._get_thetas(state.q)
        return theta_f - theta_e + state.ref_twist

    def grad_hess_strain(self, state: RobotState) -> typing.Tuple[np.ndarray, np.ndarray]:
        return twist_grad_hess(*self._get_node_pos(state.q))


def twist_grad_hess(n0p: np.ndarray, n1p: np.ndarray, n2p: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Gradient (N, 11) and Hessian (N, 11, 11) of the DER twist strain of triplets
    (n0, e, n1, f, n2) w.r.t. [x0, x1, x2, theta_e, theta_f] (edge signs not applied)."""

    N = n0p.shape[0]  # Number of springs in the batch]

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

    # Compute second derivatives
    norm2_e = norm_e ** 2
    norm2_f = norm_f ** 2
    norm_e_norm_f = norm_e * norm_f

    # Symmetrized second derivatives of twist (matches the reference
    # DER hessTwist.m; the Hessian must be symmetric)
    D2mDe2 = (-0.25 / norm2_e)[:, :, np.newaxis] * (
        np.einsum('ni,nj->nij', kb, te + tilde_t) +
        np.einsum('ni,nj->nij', te + tilde_t, kb)
    )
    D2mDf2 = (-0.25 / norm2_f)[:, :, np.newaxis] * (
        np.einsum('ni,nj->nij', kb, tf + tilde_t) +
        np.einsum('ni,nj->nij', tf + tilde_t, kb)
    )
    D2mDeDf = (0.5 / norm_e_norm_f)[:, :, np.newaxis] * (
        (2.0 / chi)[:, :, np.newaxis] * cross_te -
        np.einsum('ni,nj->nij', kb, tilde_t)
    )
    D2mDfDe = D2mDeDf.transpose(0, 2, 1)

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
