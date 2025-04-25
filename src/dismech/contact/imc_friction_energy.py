import numpy as np
from typing import List

from .contact_pairs import ContactPair
from .imc_energy import IMCEnergy
from .imc_friction_helper import generate_velocity_jacobian_funcs


class IMCFrictionEnergy(IMCEnergy):
    """ Adds frictional gradient and hessian to IMC energy. """

    def __init__(self,
                 pairs: List[ContactPair],
                 delta: float, h: float,
                 dt: float, vel_tol: float,
                 k_1: float = None, scale=True):
        super().__init__(pairs, delta, h, k_1, scale)
        self.dfr_dx_func, self.dfr_df_func, self.dfr_dx_func2, self.dfr_df_func2 = generate_velocity_jacobian_funcs()
        self.dt = np.asarray([dt])
        self.vel_tol = np.asarray([vel_tol])
        self.mu = np.vstack([p.mu for p in pairs])

    def grad_hess_energy(self, state):
        grad_C, hess_C = super().grad_hess_energy(state.q)
        return self.grad_friction(state), self.hess_friction(state, grad_C)

    def grad_friction(self, state):
        pass

    def hess_friction(self, state, contact_force):
        input_matrix = np.concat((state.u[self.ind], contact_force[self.ind], self.mu, self.vel_tol[:,None]), axis=1)
        for i in input_matrix:
            # if contact force not 0
            if np.any(i[12:24] != 0):
                Jn = self.dfr_dx_func(*i)
                print(Jn)