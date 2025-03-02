import numpy as np
import abc
import typing

from ..softrobot import SoftRobot


class ElasticEnergy(metaclass=abc.ABCMeta):
    """
    Abstract elastic energy class. Objects of this class can be used to calculate the energy of a list of springs.
    """

    def __init__(self, K: np.ndarray, nat_strain: np.ndarray, nodes_ind: np.ndarray, ind: np.ndarray):
        self.K = K
        self.nat_strain = nat_strain
        self.ind = ind

        # Get vectorized node indices
        self.node_dof_ind = SoftRobot.map_node_to_dof(nodes_ind.flatten('F'))
        self.n_nodes = nodes_ind.shape[1]

    def _get_node_pos(self, q: np.ndarray):
        """Return a M x N x 3 matrix """
        return q[self.node_dof_ind].reshape(self.n_nodes, -1, 3)

    def get_energy_linear_elastic(self, q: np.ndarray,
                                  m1: np.ndarray | None = None,
                                  m2: np.ndarray | None = None,
                                  ref_twist: np.ndarray | None = None):
        # stiffness (with discrete geometry considerations) : unit Nm (same unit as energy)
        strain = self.get_strain(q, m1, m2, ref_twist)
        del_strain = strain - self.natural_strain

        if isinstance(self.K, np.ndarray):  # rod bending
            del_strain = del_strain.reshape(2, 1)
            Energy = 0.5 * del_strain.T @ self.K @ del_strain
        else:
            Energy = 0.5 * del_strain**2
        return Energy

    def grad_hess_energy_linear_elastic(self, q: np.ndarray,
                                        m1: np.ndarray | None = None,
                                        m2: np.ndarray | None = None,
                                        ref_twist: np.ndarray | None = None) -> typing.Tuple[np.ndarray, np.ndarray]:
        strain = self.get_strain(q, m1, m2, ref_twist)
        grad_strain, hess_strain = self.grad_hess_strain(q, m1, m2, ref_twist)

        del_strain = strain - self.nat_strain
        gradE_strain = self.K * del_strain

        grad_energy = gradE_strain[:, None] * grad_strain
        hess_energy = gradE_strain[:, None, None] * hess_strain + \
            self.K[:, None, None] * \
            np.einsum('...i,...j->...ij', grad_strain, grad_strain)

        n_dof = q.shape[0]
        Fs = np.zeros(n_dof)
        Js = np.zeros((n_dof, n_dof))

        # Vectorized accumulation using numpy's ufunc.at
        np.add.at(Fs, self.ind, -grad_energy)
        np.add.at(Js, (self.ind[:, :, None],
                  self.ind[:, None, :]), -hess_energy)

        return Fs, Js

    @abc.abstractmethod
    def get_strain(self, q: np.ndarray):
        pass

    @abc.abstractmethod
    def grad_hess_strain(self, q: np.ndarray):
        pass
