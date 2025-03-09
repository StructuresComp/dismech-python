import typing
import numpy as np

from ..springs import TriangleSpring
from ..state import RobotState
from .elastic_energy import ElasticEnergy


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

    def _get_t_f_c(self, q: np.ndarray, tau: np.ndarray) -> typing.Tuple[np.ndarray, ...]:
        n0p, n1p, n2p = self._get_node_pos(q)

        vi = n2p - n1p
        vj = n0p - n2p
        vk = n1p - n0p

        norm = np.cross(vk, vi)
        unit_norm = norm / np.linalg.norm(norm, axis=1, keepdims=True)

        t = np.stack([
            np.cross(vi, unit_norm),
            np.cross(vj, unit_norm),
            np.cross(vk, unit_norm)
        ], axis=2)

        f = np.sum(unit_norm[:, :, None] * tau, axis=1)
        c = 1 / (self._A[:, None] * self._ls * np.sum(t /
                 np.linalg.norm(t, axis=1, keepdims=True) * tau, axis=1))

        return t, f, c, unit_norm
    
    def _delfi_by_delpk(self, t, tau, unit_norm):
        """
        Returns a N x 3 x 3 x 3 matrix where [N, i, j] corresponds to delfi vector for t_i for tau(:,j)
        """
        factor = np.einsum('njk,njl->nlk', tau, t) / (2 * self._A)[:, None, None]
        return factor[..., None] * unit_norm[:, None, None, :]

    # FIXME: Override main function as strain is difficult to isolate right now

    def get_energy_linear_elastic(self, state: RobotState) -> np.ndarray:
        return np.empty(0)

    # Placeholders

    def get_strain(self, state: RobotState) -> np.ndarray:
        return np.empty(0)

    def grad_hess_strain(self, state: RobotState) -> typing.Tuple[np.ndarray, np.ndarray]:
        return np.empty(0), np.empty(0)

    def grad_hess_energy_linear_elastic(self, state: RobotState) -> typing.Tuple[np.ndarray, np.ndarray]:
        tau = self._get_tau(state.tau)
        xis = self._get_xi_is(state.q)
        t, f, c, unit_norm = self._get_t_f_c(state.q, tau)

        N = t.shape[0]
        # Precompute common terms
        s_xis = self._s_s * xis - f
        # FIXME: matlab is broken???
        s_initxis = self._s_s * self._init_xis - self._init_cs

        ci_cj = np.einsum('ni, nj->nij', c, c)
        # FIXME: matlab is broken???
        ci_init_cj = np.einsum('ni, nj->nij', c, self._init_fs)

        # for E1
        t_dot_t = np.einsum('nij,nik->njk', t, t)
        t_dot_init_t = np.einsum('nij,nik->njk', t, self._init_ts)
        t_dot_t_sq = t_dot_t ** 2
        t_dot_init_t_sq = t_dot_init_t ** 2

        # for E2
        t_norm_sq = np.diagonal(t_dot_t, axis1=1, axis2=2)
        t_norms_sq_sq = np.einsum('ni,nj->nij', t_norm_sq, t_norm_sq)
        t_norms_sq_ls_sq = np.einsum('ni,nj->nij', t_norm_sq, self._ls**2)

        delfi = self._delfi_by_delpk(t, tau, unit_norm)

        # useful broadcasts
        s_xis_i = s_xis[:, None, :]
        s_xis_j = s_xis[:, :, None]
        s_initxis_j = s_initxis[:, :, None]

        coeff_1_pre = -ci_cj * s_xis_i
        coeff_2_pre = -ci_cj * s_xis_j
        coeff_3_pre = 2 * ci_init_cj * s_initxis_j

        # Precompute coeffs (sum over i = 1, j = 2)
        e_11_coeff = np.sum(coeff_1_pre * t_dot_t_sq, axis=2)[:, :, None]
        e_12_coeff = np.sum(coeff_2_pre * t_dot_t_sq, axis=1)[:, None, :]
        e_13_coeff = np.sum(coeff_3_pre * t_dot_init_t_sq, axis=2)[:, :, None]

        e_21_coeff = np.sum(coeff_1_pre * t_norms_sq_sq, axis=2)[:, :, None]
        e_22_coeff = np.sum(coeff_2_pre * t_norms_sq_sq, axis=1)[:, None, :]
        e_23_coeff = np.sum(coeff_3_pre * t_norms_sq_ls_sq, axis=2)[:, :, None]

        def compute_G_block(delfi_i):
            """Compute E1 and E2 node gradients"""
            delfi_j = delfi_i.transpose(0, 2, 1)
            M11 = np.sum(delfi_i * e_11_coeff, axis=1)
            M12 = np.sum(delfi_j * e_12_coeff, axis=2)
            M13 = np.sum(delfi_i * e_13_coeff, axis=1)

            M21 = np.sum(delfi_i * e_21_coeff, axis=1)
            M22 = np.sum(delfi_j * e_22_coeff, axis=2)
            M23 = np.sum(delfi_i * e_23_coeff, axis=1)

            return M11 + M12 + M13, M21 + M22 + M23

        G11, G21 = compute_G_block(delfi[:,0])
        G12, G22 = compute_G_block(delfi[:,1])
        G13, G23 = compute_G_block(delfi[:,2])

        ci_sj = 2 * np.einsum('ni,ni->ni', c, self._s_s)[:, None, :]

        coeff_1_pre = (ci_sj * c[:, :, None] * s_xis_j)
        # FIXME: Broken fs = cs
        coeff_2_pre = (-ci_sj * self._init_fs[:, :, None] * s_initxis_j)

        T11 = np.sum(coeff_1_pre * t_dot_t_sq, axis=1)
        T12 = np.sum(coeff_2_pre * t_dot_init_t_sq, axis=1)

        T21 = np.sum(coeff_1_pre * t_norms_sq_sq, axis=1)
        T22 = np.sum(coeff_2_pre * t_norms_sq_ls_sq, axis=1)

        X1 = T11 + T12
        X2 = T21 + T22

        grad_e1 = np.hstack([G11, G12, G13, X1])
        grad_e2 = np.hstack([G21, G22, G23, X2])

        gradE_with_stiff = self._kb[..., None] * \
            ((1-self._nu)[..., None] * grad_e1 +
             self._nu[..., None]*grad_e2) * self._A[..., None]

        # Accumulate into global array
        n_dof = state.q.shape[0]
        Fs = np.zeros(n_dof)
        Js = np.zeros((n_dof, n_dof))
        np.add.at(Fs, self._ind, -gradE_with_stiff)
        # Hessian accumulation would follow similar logic
        Js = np.zeros((n_dof, n_dof))

        return Fs, Js
