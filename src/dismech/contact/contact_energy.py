import abc
from typing import List

import numpy as np
import sympy as sp

from .contact_pairs import ContactPair


def get_E(Delta, delta, h, k_1):
    return sp.Piecewise(
        ((2 * h - Delta) ** 2, sp.And(Delta > 0, Delta <= 2 * h - delta)),
        ((1 / k_1 * (sp.log(1 + sp.exp(k_1 * (2 * h - Delta))))) ** 2,
         sp.And(Delta > 2 * h - delta, Delta < 2 * h + delta)),
        (0, True),
    )


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
        self.__expr = get_E(Delta, norm_delta, norm_h, norm_k_1)

        self.__fn = sp.lambdify(Delta, self.__expr, modules='numpy')
        self.__grad_fn = sp.lambdify(Delta, sp.diff(
            self.__expr, Delta), modules='numpy')
        self.__hess_fn = sp.lambdify(Delta, sp.diff(
            self.__expr, Delta, Delta), modules='numpy')

    def get_energy(self, q, output_scalar: bool = True):
        Delta = self.get_Delta(q)
        energy = self.__fn(Delta)
        return np.sum(energy) if output_scalar else energy

    def grad_hess_energy(self, q):
        q = q * self.scale
        Delta = self.get_Delta(q)
        grad_Delta, hess_Delta = self.get_grad_hess_Delta(q)

        grad_E_D = self.__grad_fn(Delta)
        hess_E_D = self.__hess_fn(Delta)  # shape (N,)

        grad_E = grad_Delta * grad_E_D[:, None]

        hess_E = hess_E_D[:, None, None] * \
            np.einsum('ni,nj->nij', grad_Delta, grad_Delta)
        hess_E += grad_E_D[:, None, None] * hess_Delta

        # Scale
        grad_E *= self.scale
        hess_E *= self.scale ** 2

        n_dof = q.shape[0]

        Fs = np.zeros(n_dof)
        np.add.at(Fs, self.ind, -grad_E)
        Js = np.zeros((n_dof, n_dof))
        np.add.at(Js, (self.ind[:, :, None],
                       self.ind[:, None, :]), -hess_E)

        return Fs, Js

    @abc.abstractmethod
    def get_Delta(self, q):
        pass

    @abc.abstractmethod
    def get_grad_hess_Delta(self, q):
        pass
