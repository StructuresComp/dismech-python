import abc
from typing import List

import numpy as np
import sympy as sp

from .contact_pairs import ContactPair


def compute_energy_grad_hess(Delta, delta, h, k_1):
    E = np.zeros_like(Delta)
    grad_E = np.zeros_like(Delta)
    hess_E = np.zeros_like(Delta)

    mask1 = (Delta > 0) & (Delta <= 2 * h - delta)
    mask2 = (Delta > 2 * h - delta) & (Delta < 2 * h + delta)

    # Quadratic region
    E[mask1] = (2 * h - Delta[mask1]) ** 2
    grad_E[mask1] = -2 * (2 * h - Delta[mask1])
    hess_E[mask1] = 2.0

    # Smooth transition region
    exp_term = np.exp(k_1 * (2 * h - Delta[mask2]))
    log_term = np.log(1 + exp_term)
    denom = 1 + exp_term

    E[mask2] = (1 / k_1 * log_term) ** 2
    grad_E[mask2] = -2 * (1 / k_1 * log_term) * (k_1 * exp_term / denom)
    hess_E[mask2] = (
        2 * exp_term / (k_1 * denom ** 2) *
        (log_term - (k_1 * exp_term / denom))
    )

    return E, grad_E, hess_E

class ContactEnergy(metaclass=abc.ABCMeta):

    def __init__(self, pairs: List[ContactPair], delta: float, h: float, k_1: float, scale: bool = True):
        self.pairs = np.vstack([p.pair_nodes for p in pairs])
        self.ind = np.vstack([p.ind for p in pairs])
        if scale:
            self.scale = 1.0 / h
        else:
            self.scale = 1.0

        Delta = sp.symbols("Delta")
        norm_delta = delta * self.scale
        norm_k_1 = k_1 / self.scale
        norm_h = h * self.scale

        self.norm_delta = delta * self.scale
        self.norm_h = h * self.scale
        self.norm_k_1 = k_1 / self.scale

        # debug:
        print("delta:", self.norm_delta)
        print("h:", self.norm_h)
        print("K1:", self.norm_k_1)
        print("scale: ", self.scale)

        print("upper limit for quadratic:", 2 * self.norm_h - self.norm_delta)
        print("upper limit for smooth:", 2 * self.norm_h + self.norm_delta)
        
        # self.__expr = get_E(Delta, norm_delta, norm_h, norm_k_1)
        # grad_expr = get_grad_E(Delta, norm_delta, norm_h, norm_k_1)
        # hess_expr = get_hess_E(Delta, norm_delta, norm_h, norm_k_1)

        # self.__fn = sp.lambdify(Delta, self.__expr, modules='numpy')
        # self.__grad_fn = sp.lambdify(Delta, grad_expr, modules='numpy')
        # self.__hess_fn = sp.lambdify(Delta, hess_expr, modules='numpy')

        # self.__fn = sp.lambdify(Delta, self.__expr, modules='numpy')
        # self.__grad_fn = sp.lambdify(Delta, sp.diff(
        #     self.__expr, Delta), modules='numpy')
        # self.__hess_fn = sp.lambdify(Delta, sp.diff(
        #     self.__expr, Delta, Delta), modules='numpy')

    def get_energy(self, q, output_scalar: bool = True):
        Delta = self.get_Delta(q)
        E, _, _ = compute_energy_grad_hess(Delta, self.norm_delta, self.norm_h, self.norm_k_1)
        print("contact energy: ", E)
        return np.sum(E) if output_scalar else E

    def grad_hess_energy(self, state, robot, F, first_iter):
        q = state.q
        q = q * self.scale
        Delta = self.get_Delta(q)
        grad_Delta, hess_Delta = self.get_grad_hess_Delta(q)

        E, grad_E_D, hess_E_D = compute_energy_grad_hess(Delta, self.norm_delta, self.norm_h, self.norm_k_1)

        grad_E = grad_Delta * grad_E_D[:, None]
        hess_E = hess_E_D[:, None, None] * np.einsum('ni,nj->nij', grad_Delta, grad_Delta)
        hess_E += grad_E_D[:, None, None] * hess_Delta

        if first_iter:
            self.k_c = self.get_contact_stiffness(robot, F)
            print(self.k_c)

        grad_E *= self.scale * self.k_c
        hess_E *= self.scale ** 2 * self.k_c

        n_dof = q.shape[0]

        Fs = np.zeros(n_dof)
        np.add.at(Fs, self.ind, -grad_E)
        Js = np.zeros((n_dof, n_dof))
        np.add.at(Js, (self.ind[:, :, None], self.ind[:, None, :]), -hess_E)

        return Fs, Js
    
    def get_contact_stiffness(self, robot, F: np.ndarray):
        if np.sum(np.abs(F)) < 1e-9:
            return np.ones(self.pairs.shape[0]) * 100
        valid_dofs = robot.map_node_to_dof(
            self.pairs)
        force_per_stencil = np.max(np.linalg.norm(F[valid_dofs], axis=2))
        return force_per_stencil * 1e5

    @abc.abstractmethod
    def get_Delta(self, q):
        pass

    @abc.abstractmethod
    def get_grad_hess_Delta(self, q):
        pass
