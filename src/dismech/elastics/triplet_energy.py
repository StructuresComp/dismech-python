"""
Nonlinear elastic energies on rod triplets (n0, e, n1, f, n2).

A TripletEnergy is a function E(s) of a strain vector s gathered from the DER
strains available on a triplet:

    eps_e, eps_f    stretch of edges e and f,  |x| / l_x - 1
    kappa1, kappa2  DER curvatures at the middle node
    twist           DER twist at the middle node (relative to the reference frame)

A subclass selects the components it needs with STRAINS and implements the
constitutive law in strain space through _energy_derivatives. Forces and
Jacobians follow from the chain rule

    dE/dq   = dE/ds . ds/dq
    d2E/dq2 = dE/ds . d2s/dq2 + (ds/dq)^T d2E/ds2 (ds/dq)

Natural strains (nat_strain) default to the strains of the initial state, so
pre-stretched, pre-curved or pre-twisted rest shapes need no extra work.
"""
import abc
import typing

import numpy as np

from .elastic_energy import ElasticEnergy
from .bend_energy import bend_strain, bend_grad_hess
from .twist_energy import twist_grad_hess
from ..springs.triplet_spring import TripletSprings
from ..state import RobotState

STRAIN_COMPONENTS = ('eps_e', 'eps_f', 'kappa1', 'kappa2', 'twist')


def edge_stretch_grad_hess(p0: np.ndarray, p1: np.ndarray, ref_len: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Gradient (N, 6) and Hessian (N, 6, 6) of eps = |p1 - p0| / ref_len - 1
    w.r.t. [p0, p1]."""
    vec = p1 - p0
    length = np.linalg.norm(vec, axis=1)
    t = vec / length[:, None]
    inv_l = 1.0 / ref_len

    grad = np.concatenate([-t * inv_l[:, None], t * inv_l[:, None]], axis=1)
    H = (np.eye(3) - np.einsum('ni,nj->nij', t, t)) * (inv_l / length)[:, None, None]
    hess = np.zeros((vec.shape[0], 6, 6))
    hess[:, :3, :3] = hess[:, 3:, 3:] = H
    hess[:, :3, 3:] = hess[:, 3:, :3] = -H
    return grad, hess


class TripletEnergy(ElasticEnergy):
    """
    Base class for multi-strain nonlinear energies on rod triplets.

    Subclasses define:
        STRAINS:              ordered subset of STRAIN_COMPONENTS
        _energy_derivatives:  E (N,), dE/ds (N, C) and d2E/ds2 (N, C, C)
    Any arrays _energy_derivatives needs must be set before calling
    super().__init__, since the base constructor evaluates K.
    """

    STRAINS: typing.Tuple[str, ...] = ()

    def __init__(self, springs: TripletSprings, initial_state: RobotState):
        unknown = set(self.STRAINS) - set(STRAIN_COMPONENTS)
        if unknown or not self.STRAINS:
            raise ValueError(f"STRAINS must be a non-empty subset of {STRAIN_COMPONENTS}, got {self.STRAINS}")
        self._need_stretch = bool({'eps_e', 'eps_f'} & set(self.STRAINS))
        self._need_bend = bool({'kappa1', 'kappa2'} & set(self.STRAINS))
        self._need_twist = 'twist' in self.STRAINS

        self._sgn = springs.sgn.astype(np.float64)
        self._edges_ind = springs.edges_ind
        self._theta_ind = springs.ind[:, -2:]

        # Edge signs act on the theta DOFs; the same rule holds for every component
        N = springs.N
        self._sign_grad = np.ones((N, 11))
        self._sign_grad[:, 9] = self._sgn[:, 0]
        self._sign_grad[:, 10] = self._sgn[:, 1]
        self._sign_hess = self._sign_grad[:, :, None] * self._sign_grad[:, None, :]

        super().__init__(springs, initial_state)

    @property
    def K(self):
        """Linearized stiffness at the natural state (diagonal of d2E/ds2 at s = 0)."""
        zero = np.zeros((self._springs.N, len(self.STRAINS)))
        _, _, hess = self._energy_derivatives(zero)
        return np.diagonal(hess, axis1=1, axis2=2).copy()

    @abc.abstractmethod
    def _energy_derivatives(self, del_strain: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """E (N,), dE/ds (N, C) and d2E/ds2 (N, C, C) at s = strain - natural strain."""
        pass

    # Strains

    def _get_adjusted_material_directors(self, m1: np.ndarray, m2: np.ndarray):
        m1e = m1[self._edges_ind[:, 0]]
        m2e = m2[self._edges_ind[:, 0]] * self._sgn[:, 0, None]
        m1f = m1[self._edges_ind[:, 1]]
        m2f = m2[self._edges_ind[:, 1]] * self._sgn[:, 1, None]
        return m1e, m2e, m1f, m2f

    def get_strain(self, state: RobotState) -> np.ndarray:
        n0p, n1p, n2p = self._get_node_pos(state.q)
        parts = {}
        if self._need_stretch:
            ref = self._springs.ref_len
            parts['eps_e'] = np.linalg.norm(n1p - n0p, axis=1) / ref[:, 0] - 1.0
            parts['eps_f'] = np.linalg.norm(n2p - n1p, axis=1) / ref[:, 1] - 1.0
        if self._need_bend:
            kappa = bend_strain(n0p, n1p, n2p,
                                *self._get_adjusted_material_directors(state.m1, state.m2))
            parts['kappa1'], parts['kappa2'] = kappa[:, 0], kappa[:, 1]
        if self._need_twist:
            theta_e = state.q[self._theta_ind[:, 0]] * self._sgn[:, 0]
            theta_f = state.q[self._theta_ind[:, 1]] * self._sgn[:, 1]
            parts['twist'] = theta_f - theta_e + state.ref_twist[self._springs.triplet_ind]
        return np.stack([parts[c] for c in self.STRAINS], axis=1)

    def grad_hess_strain(self, state: RobotState) -> typing.Tuple[np.ndarray, np.ndarray]:
        n0p, n1p, n2p = self._get_node_pos(state.q)
        N = n0p.shape[0]
        grads, hesses = {}, {}

        if self._need_stretch:
            ref = self._springs.ref_len
            for c, (p0, p1, sl) in enumerate(((n0p, n1p, slice(0, 6)), (n1p, n2p, slice(3, 9)))):
                name = ('eps_e', 'eps_f')[c]
                g, h = edge_stretch_grad_hess(p0, p1, ref[:, c])
                grads[name] = np.zeros((N, 11))
                hesses[name] = np.zeros((N, 11, 11))
                grads[name][:, sl] = g
                hesses[name][:, sl, sl] = h
        if self._need_bend:
            g, h = bend_grad_hess(n0p, n1p, n2p,
                                  *self._get_adjusted_material_directors(state.m1, state.m2))
            grads['kappa1'], grads['kappa2'] = g[..., 0], g[..., 1]
            hesses['kappa1'], hesses['kappa2'] = h[..., 0], h[..., 1]
        if self._need_twist:
            grads['twist'], hesses['twist'] = twist_grad_hess(n0p, n1p, n2p)

        grad = np.stack([grads[c] for c in self.STRAINS], axis=-1)
        hess = np.stack([hesses[c] for c in self.STRAINS], axis=-1)
        return grad, hess

    # Energy, force and Jacobian through the chain rule

    def get_energy_linear_elastic(self, state: RobotState, output_scalar: bool = True):
        E, _, _ = self._energy_derivatives(self._get_del_strain(state))
        return np.sum(E) if output_scalar else E

    def grad_hess_energy_linear_elastic(self, state: RobotState, sparse: bool = False):
        _, dE, d2E = self._energy_derivatives(self._get_del_strain(state))
        grad_strain, hess_strain = self.grad_hess_strain(state)

        grad_energy = np.einsum('nc,nic->ni', dE, grad_strain)
        hess_energy = np.einsum('nc,nijc->nij', dE, hess_strain) + \
            np.einsum('nic,ncd,njd->nij', grad_strain, d2E, grad_strain)
        return self._assemble(grad_energy, hess_energy, state, sparse)
