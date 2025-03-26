import numpy as np
import scipy.sparse as sp
import abc
import typing

from ..soft_robot import SoftRobot
from ..state import RobotState


class PostInitABCMeta(abc.ABCMeta):
    """ Simple metaclass to call post_init after subclass init"""
    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        obj.__post_init__()
        return obj


class ElasticEnergy(metaclass=PostInitABCMeta):
    """
    Abstract elastic energy class. Objects of this class can be used to calculate the energy of a list of springs.
    """

    def __init__(self, K: np.ndarray,
                 nodes_ind: np.ndarray,
                 ind: np.ndarray,
                 initial_state: RobotState):
        self._K = K
        self._n_K = 1 if self._K.ndim == 1 else self._K.shape[1]

        # Get vectorized node indices
        self._node_dof_ind = SoftRobot.map_node_to_dof(nodes_ind.flatten('F'))
        self._n_nodes = nodes_ind.shape[1]

        self._initial_state = initial_state
        self._ind = ind

        # sparse index creation
        stencil_n_dof = self._ind.shape[1]
        self._rows = np.repeat(self._ind, stencil_n_dof, axis=1).ravel()
        self._cols = np.tile(self._ind, (1, stencil_n_dof)).ravel()

    def __post_init__(self):
        self._nat_strain = self.get_strain(self._initial_state).copy()

    def _get_node_pos(self, q: np.ndarray):
        """Return a M x N x 3 matrix """
        return q[self._node_dof_ind].reshape(self._n_nodes, -1, 3)

    def get_energy_linear_elastic(self, state: RobotState, output_scalar: bool = True):
        strain = self.get_strain(state)
        del_strain = (strain - self._nat_strain).reshape(-1, self._n_K)
        if output_scalar:
            return 0.5 * np.sum(self._K.reshape(-1, self._n_K) * del_strain**2)
        return 0.5 * self._K.reshape(-1, self._n_K) * del_strain**2

    def grad_hess_energy_linear_elastic(self, state: RobotState, sparse: bool = False) -> typing.Tuple[np.ndarray, np.ndarray] | typing.Tuple[np.ndarray, sp.csr_array]:
        strain = self.get_strain(state)
        grad_strain, hess_strain = self.grad_hess_strain(state)

        del_strain = (strain - self._nat_strain).reshape(-1, self._n_K)
        gradE_strain = self._K.reshape(-1, self._n_K) * del_strain

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

        n_dof = state.q.shape[0]

        # Force always dense
        Fs = np.zeros(n_dof)
        np.add.at(Fs, self._ind, -grad_energy)

        if sparse:
            Js = sp.coo_matrix((-hess_energy.ravel(),
                                (self._rows, self._cols)),
                               shape=(n_dof, n_dof)).tocsr()
        else:
            Js = np.zeros((n_dof, n_dof))
            np.add.at(Js, (self._ind[:, :, None],
                           self._ind[:, None, :]), -hess_energy)

        return Fs, Js

    @abc.abstractmethod
    def get_strain(self, state: RobotState) -> np.ndarray:
        pass

    @abc.abstractmethod
    def grad_hess_strain(self, state: RobotState) -> typing.Tuple[np.ndarray, np.ndarray]:
        pass
