import abc

import numpy as np
import sympy as sp


def get_E(Delta, delta, h, k_1):
    return sp.Piecewise(
        ((2 * h - Delta) ** 2, sp.And(Delta > 0, Delta <= 2 * h - delta)),
        ((1 / k_1 * (sp.log(1 + sp.exp(k_1 * (2 * h - Delta))))) ** 2,
         sp.And(Delta > 2 * h - delta, Delta < 2 * h + delta)),
        (0, True),
    )


class ContactEnergy(metaclass=abc.ABCMeta):

    def __init__(self, delta: float, h: float, k_1: float):
        Delta = sp.symbols("Delta")
        self.__expr = get_E(Delta, delta, h, k_1)
        self.__fn = sp.lambdify(Delta, self.__expr, modules='numpy')
        self.__grad_fn = sp.lambdify(Delta, sp.diff(
            self.__expr, Delta), modules='numpy')
        self.__hess_fn = sp.lambdify(Delta, sp.diff(
            self.__expr, Delta, Delta), modules='numpy')

    def get_energy(self, state, output_scalar: bool = True):
        Delta = self.get_Delta(state)
        energy = self.__fn(Delta)
        return np.sum(energy) if output_scalar else energy

    def grad_hess_energy(self, state):
        Delta = self.get_Delta(state)
        grad_Delta, hess_Delta = self.get_grad_hess_Delta(state)

        # Batching issue
        grad_Delta = grad_Delta
        hess_Delta = hess_Delta

        grad_E_D = self.__grad_fn(Delta)
        hess_E_D = self.__hess_fn(Delta)  # shape (N,)

        grad_E = grad_Delta * grad_E_D[:, None]

        hess_E = hess_E_D[:, None, None] * \
            np.einsum('ni,nj->nij', grad_Delta, grad_Delta)
        hess_E += grad_E_D[:, None, None] * hess_Delta

        return grad_E, hess_E

    @abc.abstractmethod
    def get_Delta(self, state):
        pass

    @abc.abstractmethod
    def get_grad_hess_Delta(self, state):
        pass
