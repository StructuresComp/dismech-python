import numpy as np
import abc
import typing

from ..soft_robot import SoftRobot


class ElasticEnergy(metaclass=abc.ABCMeta):
    """
    Abstract elastic energy class. Objects of this class can be used to calculate the energy of a list of springs.
    """

    def __init__(self, K: np.ndarray, nat_strain: np.ndarray, nodes_ind: np.ndarray, ind: np.ndarray):
        self._K = K
        self._n_K = 1 if self._K.ndim == 1 else self._K.shape[1]

        self._nat_strain = nat_strain
        self._ind = ind

        # Get vectorized node indices
        self._node_dof_ind = SoftRobot.map_node_to_dof(nodes_ind.flatten('F'))
        self._n_nodes = nodes_ind.shape[1]

    def _get_node_pos(self, q: np.ndarray):
        """Return a M x N x 3 matrix """
        return q[self._node_dof_ind].reshape(self._n_nodes, -1, 3)

    # FIXME: Never called so didn't fix
    def get_energy_linear_elastic(self, q: np.ndarray, **kwargs):
        # stiffness (with discrete geometry considerations) : unit Nm (same unit as energy)
        strain = self.get_strain(q, **kwargs)
        del_strain = strain - self._nat_strain

        if isinstance(self.K, np.ndarray):  # rod bending
            del_strain = del_strain.reshape(2, 1)
            Energy = 0.5 * del_strain.T @ self.K @ del_strain
        else:
            Energy = 0.5 * del_strain**2
        return Energy

    def grad_hess_energy_linear_elastic(self, q: np.ndarray, **kwargs) -> typing.Tuple[np.ndarray, np.ndarray]:
        strain = self.get_strain(q, **kwargs)
        grad_strain, hess_strain = self.grad_hess_strain(q, **kwargs)

        del_strain = strain - self._nat_strain
        gradE_strain = self._K * del_strain

        # Reshape to handle multiple strain components (EI1, EI2)
        gradE_strain = gradE_strain.reshape(gradE_strain.shape[0], self._n_K)
        grad_strain = grad_strain.reshape(
            grad_strain.shape[0], grad_strain.shape[1], self._n_K)
        hess_strain = hess_strain.reshape(
            hess_strain.shape[0], grad_strain.shape[1], grad_strain.shape[1], self._n_K)

        # Gradient
        grad_energy = np.sum(gradE_strain[:, None, :] * grad_strain, axis=-1)

        # Term 1: gradE_strain * hess_strain summed over components
        hess_term1 = np.sum(
            gradE_strain[:, None, None, :] * hess_strain, axis=-1)

        # Term 2: K * (grad_strain ⊗ grad_strain) summed over components
        outer = np.einsum('nmc,nkc->nmkc', grad_strain, grad_strain)
        hess_term2 = np.einsum(
            'nc,nmkc->nmk', self._K.reshape(-1, self._n_K), outer)

        hess_energy = hess_term1 + hess_term2

        # Correct signs
        if (sign_grad := getattr(self, '_sign_grad', None)) is not None:
            grad_energy *= sign_grad

        if (sign_hess := getattr(self, '_sign_hess', None)) is not None:
            hess_energy *= sign_hess

        n_dof = q.shape[0]
        Fs = np.zeros(n_dof)
        Js = np.zeros((n_dof, n_dof))

        # Accumulate gradients and Hessians
        np.add.at(Fs, self._ind, -grad_energy)
        np.add.at(Js, (self._ind[:, :, None],
                  self._ind[:, None, :]), -hess_energy)

        return Fs, Js

    @abc.abstractmethod
    def get_strain(self, q: np.ndarray, **kwargs) -> np.ndarray:
        pass

    @abc.abstractmethod
    def grad_hess_strain(self, q: np.ndarray, **kwargs) -> typing.Tuple[np.ndarray, np.ndarray]:
        pass
