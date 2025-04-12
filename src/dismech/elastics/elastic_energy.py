import numpy as np
import scipy.sparse as sp
import abc
import typing
import dataclasses
import matplotlib.pyplot as plt

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
                 initial_state: RobotState,
                 get_strain = None):
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

        if get_strain is None:
            self._nat_strain = None
        else:
            self._nat_strain = get_strain(self._get_node_pos(self._initial_state.q)).copy()
            

    def __post_init__(self):
        if self._nat_strain is None:
            self._nat_strain = self.get_strain(self._initial_state).copy()
        else:
            self._nat_strain = np.where(np.isnan(self._nat_strain), self.get_strain(self._initial_state), self._nat_strain)

    def _get_node_pos(self, q: np.ndarray):
        """Return a M x N x 3 matrix """
        return q[self._node_dof_ind].reshape(self._n_nodes, -1, 3)
    
    def set_nat_strain(self, strain: np.ndarray):
        if strain.shape == self._nat_strain.shape:
            self._nat_strain = strain

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

    def fdm_check_grad_hess_strain(self, state: RobotState) -> typing.Tuple[np.ndarray, np.ndarray, bool, bool]:
        strain = self.get_strain(state)
        grad_strain, hess_strain = self.grad_hess_strain(state)

        change = 1e-8
        grad_tol = np.mean(np.abs(grad_strain))*1e-3
        hess_tol = np.mean(np.abs(hess_strain))*1e-3

        grad_FDM = np.zeros_like(grad_strain)
        hess_FDM = np.zeros_like(hess_strain)

        # perturb each DOF by small value "change" (one at a time)
        # compute perturbed_strain and perturbed_grad_strain
        # grad_FDM [i] = (perturbed_strain - strain)/change
        # hess_FDM [:,i] = (perturbed_grad_strain - grad_strain)/ change

        for i in range(grad_strain.shape[1]):
            # Perturb state
            q_perturbed = state.q.copy()
            q_perturbed[self._ind[:,i]] += change
            state_perturbed = dataclasses.replace(state, q=q_perturbed)  # Create a new instance

            # Compute perturbed strain and gradient of strain
            perturbed_strain = self.get_strain(state_perturbed)
            perturbed_grad_strain, _ = self.grad_hess_strain(state_perturbed)

            # Compute finite difference approximations
            grad_FDM[:,i] = (perturbed_strain - strain) / change
            hess_FDM[:,:,i] = (perturbed_grad_strain - grad_strain) / change

        # for j in range(grad_strain.shape[0]):
        #     for i in range(grad_strain.shape[1]):
        #         # Perturb state
        #         q_perturbed = state.q.copy()
        #         q_perturbed[self._ind[j,i]] += change
        #         state_perturbed = dataclasses.replace(state, q=q_perturbed)  # Create a new instance

        #         # Compute perturbed strain and gradient of strain
        #         perturbed_strain = self.get_strain(state_perturbed)
        #         perturbed_grad_strain, _ = self.grad_hess_strain(state_perturbed)

        #         # Compute finite difference approximations
        #         grad_FDM[j,i] = (perturbed_strain[i] - strain[i]) / change
        #         hess_FDM[j,:,i] = (perturbed_grad_strain[i,:] - grad_strain[i,:]) / change
        
        # Compute boolean matches
        grad_FDM_match = np.all(np.abs(grad_FDM - grad_strain) < grad_tol)
        hess_FDM_match = np.all(np.abs(hess_FDM - hess_strain) < hess_tol)

        # print("grad of strain using FDM:", grad_FDM)
        # print("grad of strain:", grad_strain)

        # # Plot results
        # plt.figure(1)
        # plt.clf()
        # plt.scatter(np.arange(len(grad_FDM.flatten())), grad_FDM.flatten(), label='grad_FDM', marker='o', color='blue')
        # plt.scatter(np.arange(len(grad_strain.flatten())), grad_strain.flatten(), label='grad_strain', marker='x', color='red')
        # plt.legend()
        # plt.title("Gradient FDM vs Analytical")
        # plt.xlabel("Index")
        # plt.ylabel("Value")
        # plt.grid()
        # plt.pause(0.1)
        
        # plt.figure(2)
        # plt.clf()
        # plt.scatter(np.arange(len(hess_FDM.flatten())), hess_FDM.flatten(), label='hess_FDM', marker='o', color='blue')
        # plt.scatter(np.arange(len(hess_strain.flatten())), hess_strain.flatten(), label='hess_strain', marker='x', color='red')
        # plt.legend()
        # plt.title("Hessian FDM vs Analytical")
        # plt.xlabel("Index")
        # plt.ylabel("Value")
        # plt.grid()
        # plt.pause(0.1)
        
        return grad_FDM, hess_FDM, grad_FDM_match, hess_FDM_match

    @abc.abstractmethod
    def get_strain(self, state: RobotState) -> np.ndarray:
        pass

    @abc.abstractmethod
    def grad_hess_strain(self, state: RobotState) -> typing.Tuple[np.ndarray, np.ndarray]:
        pass
