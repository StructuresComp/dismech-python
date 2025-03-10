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
    
    def _ddelfi_by_del_p_k1_p_k2(self, t, tau, unit_norm):
        """
        Returns a N x 3 x 3 x 3 x 3 x 3 tensor where:
        - [N, i, j, k] corresponds to the 3x3 Hessian matrix for parameters (k1=i, k2=j)
        and tau column k.
        """
        t_dot_tau = np.einsum('nij,nik->njk', t, tau)
        scale = 1 / (4 * self._A ** 2)

        # TODO: Find a more efficient way to do this :(
        outer_1 = np.einsum('ni,nj->nij', unit_norm, t[..., 0])
        outer_2 = np.einsum('ni,nj->nij', unit_norm, t[..., 1])
        outer_3 = np.einsum('ni,nj->nij', unit_norm, t[..., 2])
        outer = np.stack([outer_1, outer_2, outer_3], axis=1)
        outer_sum = outer + outer.transpose(0, 1, 3, 2)

        return np.einsum('nij,nklm,n->nikjlm', t_dot_tau, outer_sum, scale)

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
        # FIXME: SUPER WRONG!!! I AM STUPID
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

        G11, G21 = compute_G_block(delfi[:, 0])
        G12, G22 = compute_G_block(delfi[:, 1])
        G13, G23 = compute_G_block(delfi[:, 2])

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

        # Hessian - nightmare :(

        
        # print(ddelfi[0, 0, 0, 0])

        # 1: d/dx^2 super easy (N x 3 x 3)
        si_sj = np.einsum('ni, nj->nij', self._s_s, self._s_s)

        tri_inds = np.triu_indices(3)
        common = 2 * ci_cj[:, tri_inds] * si_sj[:, tri_inds]

        def compute_dxdx(factor):
            """ Compute upper triangle (66% computation) """
            dxdx_tri = common * factor[:, tri_inds]

            dxdx = np.zeros((N, 3, 3))
            dxdx[:, tri_inds] = dxdx_tri
            dxdx[:, tri_inds] = dxdx_tri
            return dxdx

        dxdx_1 = compute_dxdx(t_dot_t_sq)
        dxdx_2 = compute_dxdx(t_norms_sq_sq)

        # 2: d/dp^2 HELLL (N x 9 x 9)

        # [N, i, j, k, a, b] corresponds to the a x b Hessian matrix for parameters (k1=i, k2=j) and tau column k.
        ddelfi = self._ddelfi_by_del_p_k1_p_k2(t, tau, unit_norm)

        # delfi_sq is a N x 3 x 3 x 3 x 3 x 3 x 3 where t_i, tau_j, t_k, tau_l returns delfi(t_i, tau_j).T * delfi(t_k, tau_l)
        delfi_sq = np.einsum('nabc,ndef->nabdecf', delfi, delfi)

        # Not using above because the broadcast was wrong

        common_1 = -ci_cj * s_xis[:, :, None]
        common_3 = -ci_cj * s_xis[:, None, :]
        common_5 = 2 * ci_init_cj * s_initxis[:, None, :]

        def compute_dpdp(factor1, factor2):
            # -ci_cj * s_xis_j * factor1 * ddelfi[k2, k1] summed over i,j
            M1 = np.einsum('nij,nij,ntkjab->ntakb', common_1,factor1, ddelfi)
            # -ci_cj * factor1 * (delfi_j_k1.T @ delfi_i_k2)
            M2 = np.einsum('nij,nij,nkjtiab->ntakb', ci_cj, factor1, delfi_sq)
            # -ci_cj * s_xis_i * factor1 * ddelfi[k2, k1] summed over i,j
            M3 = np.einsum('nij,nij,ntkiab->ntakb', common_3, factor1, ddelfi)
            # -ci_cj * factor1 * (delfi_i_k1.T @ delfi_j_k2) summed over i,j
            M4 = np.einsum('nij,nij,nkitjab->ntakb', ci_cj, factor1, delfi_sq)
            # Term M5: 2 * ci_init_cj * factor2 * s_initxis_j * ddelfi[k2, k1] summed over i,j
            M5 = np.einsum('nij,nij,ntkiab->ntakb', common_5, factor2, ddelfi)

            return (M1 + M2 + M3 + M4 + M5).reshape(N, 9, 9)
        
        dpdp_1 = compute_dpdp(t_dot_t_sq, t_dot_init_t_sq)
        dpdp_2 = compute_dpdp(t_norms_sq_sq, t_norms_sq_ls_sq)
        
        

        # 3: d/dxdp meh (N x 9 x 3)

        common = -2 * c[:, :, None] * self._s_s[:, :, None] * c[:, None, :]

        def compute_dxdp(factor):
            """Compute a N x 3 x 9 matrix (for upper triangle)"""
            return np.einsum('nij,nljc->nlci', common * factor, delfi).reshape(-1,9,3)

        dxdp_1 = compute_dxdp(t_dot_t_sq)
        dxdp_2 = compute_dxdp(t_norms_sq_sq)

        # 4: put together the hessian

        def get_hess_blocks(TL, TR, BR):
            return np.concatenate([np.concatenate([TL, TR], axis=2), np.concatenate([np.transpose(TR, (0, 2, 1)), BR], axis=2)], axis=1)

        hess_e1 = get_hess_blocks(dpdp_1, dxdp_1, dxdx_1)
        hess_e2 = get_hess_blocks(dpdp_2, dxdp_2, dxdx_2)

        hessE_with_stiff = self._kb[..., None, None] * \
            ((1-self._nu)[..., None, None] * hess_e1 +
             self._nu[..., None, None]*hess_e2) * self._A[..., None, None]

        # Accumulate into global array
        n_dof = state.q.shape[0]
        Fs = np.zeros(n_dof)
        Js = np.zeros((n_dof, n_dof))

        np.add.at(Fs, self._ind, -gradE_with_stiff)
        np.add.at(Js, (self._ind[:, :, None],
                  self._ind[:, None, :]), -hessE_with_stiff)

        return Fs, Js
