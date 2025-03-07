import numpy as np
import typing

from ..springs import TriangleSpring
from ..state import RobotState
from .elastic_energy import ElasticEnergy
from ..frame_util import compute_tfc_midedge


class TriangleEnergy(ElasticEnergy):
    def __init__(self, springs: typing.List[TriangleSpring], initial_state: RobotState):
        super().__init__(np.array([s.kb for s in springs]),
                         np.array([s.nodes_ind for s in springs]),
                         np.array([s.ind for s in springs]),
                         initial_state)
        self._edges_ind = np.array([s.ind[-3:] for s in springs])
        self._face_edges = np.array([s.face_edges for s in springs])

        self._kb = np.array([s.kb for s in springs])
        self._nu = np.array([s.nu for s in springs])
        self._A = np.array([s.A for s in springs])
        self._init_ts = np.array([s.init_ts for s in springs])
        self._init_fs = np.array([s.init_fs for s in springs])
        self._init_cs = np.array([s.init_cs for s in springs])
        self._init_xis = np.array([s.init_xis for s in springs])
        self._ls = np.array([s.ref_len for s in springs])
        self._s_s = np.array([s.sgn for s in springs])

    def _get_xi_is(self, q: np.ndarray) -> np.ndarray:
        return q[self._edges_ind]

    def _get_tau(self, tau: np.ndarray) -> np.ndarray:
        return (tau[:, self._face_edges] * self._s_s[None, ...]).transpose(1, 0, 2)

    # FIXME: Override main function as strain is difficult to isolate right now

    def get_energy_linear_elastic(self, state: RobotState) -> np.ndarray:
        return np.empty(0)

    def grad_hess_energy_linear_elastic(self, state: RobotState) -> typing.Tuple[np.ndarray, np.ndarray]:
        n0p, n1p, n2p = self._get_node_pos(state.q)
        xis = self._get_xi_is(state.q)
        tau = self._get_tau(state.tau)
        N = n0p.shape[0]

        # Calculate t, f, c
        vi = n2p - n1p
        vj = n0p - n2p
        vk = n1p - n0p

        norm = np.cross(vk, vi)
        unit_norm = norm / (np.linalg.norm(norm, axis=1)[:, None])

        t = np.stack(
            [
                np.cross(vi, unit_norm),
                np.cross(vj, unit_norm),
                np.cross(vk, unit_norm)
            ], axis=2
        )
        f = np.sum(unit_norm[:, :, None] * tau, axis=1)
        c = 1 / (self._A[:, None] * self._ls * np.sum((t /
                 np.linalg.norm(t, axis=1)[:, None]) * tau, axis=1))
        tau_test = state.tau[:, self._face_edges].transpose(1, 2, 0)
        t_test, f_test, c_test = compute_tfc_midedge(
            self._get_node_pos(state.q).transpose(1, 0, 2), tau_test, self._s_s)
        gradE = np.zeros((N, 12))

        n_dof = state.q.shape[0]
        Fs = np.zeros(n_dof)
        Js = np.zeros((n_dof, n_dof))

        # Accumulate gradients and Hessians
        np.add.at(Fs, self._ind, -gradE_with_stiff)
        # np.add.at(Js, (self._ind[:, :, None],
        #         self._ind[:, None, :]), -hessE_with_stiff)

        return Fs, Js

    # Placeholders

    def get_strain(self, state: RobotState) -> np.ndarray:
        return np.empty(0)

    def grad_hess_strain(self, state: RobotState) -> typing.Tuple[np.ndarray, np.ndarray]:
        return np.empty(0), np.empty(0)
