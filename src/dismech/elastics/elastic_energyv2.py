import numpy as np
import scipy.sparse as sp
import abc
import typing

from ..soft_robot import SoftRobot
from ..springs import Springs
from ..state import RobotState


class PostInitABCMeta(abc.ABCMeta):
    """ Simple metaclass to call post_init after subclass init"""
    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        obj.__post_init__()
        return obj


class ElasticEnergy2(metaclass=PostInitABCMeta):
    """
    Abstract elastic energy class.
    """

    def __init__(self, springs: Springs, initial_state: RobotState):
        """
        Args:
            springs (Springs): Springs object containing fixed indices and mutable nat_strain and inc_strain.
            initial_state (RobotState): State associated with nat_strain.
        """
        self._springs = springs
        self._n_K = 1 if self.K.ndim == 1 else self.K.shape[1]

        # Get vectorized node indices
        self._node_dof_ind = SoftRobot.map_node_to_dof(
            springs.nodes_ind.flatten('F'))
        self._n_nodes = springs.nodes_ind.shape[1]

        # for __post_init__
        self._initial_state = initial_state

        # sparse index creation
        stencil_n_dof = self._springs.ind.shape[1]
        self._rows = np.repeat(
            self._springs.ind, stencil_n_dof, axis=1).ravel()
        self._cols = np.tile(self._springs.ind, (1, stencil_n_dof)).ravel()

    def __post_init__(self):
        # If nat_strain was not set by the user, set it to initial state strain
        nat_strain = self.get_strain(self._initial_state)
        self._springs.nat_strain = np.where(
            np.isnan(self._springs.nat_strain),
            nat_strain,
            self._springs.nat_strain
        )

    @property
    @abc.abstractmethod
    def K(self) -> np.ndarray:
        pass

    def _get_node_pos(self, q: np.ndarray):
        """Return a M x N x 3 matrix """
        return q[self._node_dof_ind].reshape(self._n_nodes, -1, 3)

    def _get_del_strain(self, state: RobotState) -> np.ndarray:
        base = self._springs.nat_strain + self._springs.inc_strain
        return (self.get_strain(state) - base).reshape(-1, self._n_K)

    def get_energy_linear_elastic(self, state: RobotState, output_scalar: bool = True):
        del_strain = self._get_del_strain(state)
        energy = 0.5 * self.K.reshape(-1, self._n_K) * del_strain**2
        return np.sum(energy) if output_scalar else energy

    def _compute_grad_hess_energy_terms(self, del_strain: np.ndarray, grad_strain: np.ndarray, hess_strain: np.ndarray):
        """ Calculate gradient and hessian of energy with chain rule """
        K = self.K.reshape(-1, self._n_K)
        gradE_strain = K * del_strain

        # Ensure correct shape
        gradE_strain = gradE_strain.reshape(-1, self._n_K)
        grad_strain = grad_strain.reshape(-1, grad_strain.shape[1], self._n_K)
        hess_strain = hess_strain.reshape(
            -1, hess_strain.shape[1], hess_strain.shape[2], self._n_K)

        grad_energy = np.sum(gradE_strain[:, None, :] * grad_strain, axis=-1)
        hess_term1 = np.sum(
            gradE_strain[:, None, None, :] * hess_strain, axis=-1)
        hess_term2 = np.einsum(
            'nc,nmkc->nmk', K, np.einsum('nmc,nkc->nmkc', grad_strain, grad_strain))

        hess_energy = hess_term1 + hess_term2

        # Sign correction
        if hasattr(self, '_sign_grad'):
            grad_energy *= self._sign_grad
        if hasattr(self, '_sign_hess'):
            hess_energy *= self._sign_hess

        return grad_energy, hess_energy

    def _assemble_force_hessian(self, grad_energy: np.ndarray, hess_energy: np.ndarray, n_dof: int, sparse: bool):
        """ Accumulate gradient and hessian within global DOF matrices """
        Fs = np.zeros(n_dof)
        np.add.at(Fs, self._springs.ind, -grad_energy)

        if sparse:
            Js = sp.coo_matrix((-hess_energy.ravel(),
                                (self._rows, self._cols)),
                               shape=(n_dof, n_dof)).tocsr()
        else:
            Js = np.zeros((n_dof, n_dof))
            np.add.at(Js, (self._springs.ind[:, :, None],
                           self._springs.ind[:, None, :]), -hess_energy)
        return Fs, Js

    def grad_hess_energy_linear_elastic(self, state: RobotState, sparse: bool = False) -> typing.Tuple[np.ndarray, np.ndarray] | typing.Tuple[np.ndarray, sp.csr_array]:
        del_strain = self._get_del_strain(state)
        grad_strain, hess_strain = self.grad_hess_strain(state)

        grad_energy, hess_energy = self._compute_grad_hess_energy_terms(
            del_strain, grad_strain, hess_strain)

        return self._assemble_force_hessian(grad_energy, hess_energy, state.q.shape[0], sparse)

    @abc.abstractmethod
    def get_strain(self, state: RobotState) -> np.ndarray:
        pass

    @abc.abstractmethod
    def grad_hess_strain(self, state: RobotState) -> typing.Tuple[np.ndarray, np.ndarray]:
        pass
