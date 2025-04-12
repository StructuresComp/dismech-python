import typing
import scipy.sparse as sp
import numpy as np

from ..springs import TriangleSpring
from ..state import RobotState
from .elastic_energy import ElasticEnergy

from .triangle_helper import compute_dp_jit, compute_delfi_sq_jit, compute_dpdp_jit


class TriangleEnergy(ElasticEnergy):
    def __init__(self, springs: typing.List[TriangleSpring], initial_state: RobotState, get_strain = None):
        super().__init__(np.array([s.kb for s in springs]),
                         np.array([s.nodes_ind for s in springs]),
                         np.array([s.ind for s in springs]),
                         initial_state,
                         get_strain)
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

        # sparse index creation
        stencil_n_dof = self._ind.shape[1]
        self._rows = np.repeat(self._ind, stencil_n_dof, axis=1).ravel()
        self._cols = np.tile(self._ind, (1, stencil_n_dof)).ravel()

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

        return t, f, c, unit_norm, np.linalg.norm(norm, axis=1) / 2

    def _delfi_by_delpk(self, t, tau, unit_norm, A):
        """
        Returns a N x 3 x 3 x 3 tensor where [N, i, j] corresponds to delfi vector for t_i and tau(:,j).
        """
        factor = np.einsum('njk,njl->nlk', tau, t) / \
            (2 * A)[:, None, None]
        return factor[..., None] * unit_norm[:, None, None, :]

    def _ddelfi_by_del_p_k1_p_k2(self, t, tau, unit_norm):
        """
        Returns a N x 3 x 3 x 3 x 3 x 3 tensor where [N, i, j, k] corresponds to the 3x3 Hessian matrix for t_i, t_j, and tau(:,k).
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

    # Placeholders

    def get_strain(self, state: RobotState) -> np.ndarray:
        return np.empty(0)

    def grad_hess_strain(self, state: RobotState) -> typing.Tuple[np.ndarray, np.ndarray]:
        return np.empty(0), np.empty(0)
    
    # Override

    def get_energy_linear_elastic(self, state: RobotState) -> np.ndarray:
        tau = self._get_tau(state.tau)
        xis = self._get_xi_is(state.q)
        t, f, c, _, _ = self._get_t_f_c(state.q, tau)

        # s_s terms
        s_xis = self._s_s * xis - f
        s_initxis = self._s_s * self._init_xis - self._init_fs

        # c terms
        ci_cj = np.einsum('ni, nj->nij', c, c)
        ci_init_cj = np.einsum('ni, nj->nij', c, self._init_cs)
        init_ci_init_cj = np.einsum(
            'ni, nj->nij', self._init_cs, self._init_cs)

        # for E1
        t_dot_t = np.einsum('nij,nik->njk', t, t)
        t_dot_init_t = np.einsum('nij,nik->njk', t, self._init_ts)
        init_t_dot_init_t = np.einsum(
            'nij,nik->njk', self._init_ts, self._init_ts)
        t_dot_t_sq = t_dot_t ** 2
        t_dot_init_t_sq = t_dot_init_t ** 2
        init_t_dot_init_t_sq = init_t_dot_init_t ** 2

        # for E2
        t_norm_sq = np.diagonal(t_dot_t, axis1=1, axis2=2)
        t_norms_sq_sq = np.einsum('ni,nj->nij', t_norm_sq, t_norm_sq)
        t_norms_sq_ls_sq = np.einsum('ni,nj->nij', t_norm_sq, self._ls**2)
        ls_sq_ls_sq = np.einsum('ni,nj->nij', self._ls**2, self._ls**2)

        # Extend to N x 3 x 3
        s_xis_i = s_xis[:, :, None]
        s_xis_j = s_xis[:, None, :]
        s_initxis_i = s_initxis[:, :, None]
        s_initxis_j = s_initxis[:, None, :]

        e_coeff1 = ci_cj * s_xis_i * s_xis_j
        e_coeff2 = init_ci_init_cj * s_initxis_i * s_initxis_j
        e_coeff3 = 2 * ci_init_cj * s_xis_i * s_initxis_j

        def compute_e(factor1, factor2, factor3):
            return np.einsum('nij->n', e_coeff1 * factor1 + e_coeff2 * factor2 + e_coeff3 * factor3)

        e1 = compute_e(t_dot_t_sq, init_t_dot_init_t_sq, t_dot_init_t_sq)
        e2 = compute_e(t_norms_sq_sq, ls_sq_ls_sq, t_norms_sq_ls_sq)

        return self._kb * ((1-self._nu) * e1 + self._nu*e2) * self._A

    def grad_hess_energy_linear_elastic(self, state: RobotState, sparse=False) -> typing.Tuple[np.ndarray, np.ndarray]:
        tau = self._get_tau(state.tau)
        xis = self._get_xi_is(state.q)
        t, f, c, unit_norm, A = self._get_t_f_c(state.q, tau)
        N = t.shape[0]

        # s_s terms
        s_xis = self._s_s * xis - f
        s_initxis = self._s_s * self._init_xis - self._init_fs

        si_sj = np.einsum('ni, nj->nij', self._s_s, self._s_s)

        # c terms
        ci_cj = np.einsum('ni, nj->nij', c, c)
        ci_init_cj = np.einsum('ni, nj->nij', c, self._init_cs)

        # hybrid
        ci_sj = 2 * np.einsum('ni,ni->ni', c, self._s_s)[:, None, :]

        # for E1
        t_dot_t = np.einsum('nij,nik->njk', t, t)
        t_dot_init_t = np.einsum('nij,nik->njk', t, self._init_ts)
        t_dot_t_sq = t_dot_t ** 2
        t_dot_init_t_sq = t_dot_init_t ** 2

        # for E2
        t_norm_sq = np.diagonal(t_dot_t, axis1=1, axis2=2)
        t_norms_sq_sq = np.einsum('ni,nj->nij', t_norm_sq, t_norm_sq)
        t_norms_sq_ls_sq = np.einsum('ni,nj->nij', t_norm_sq, self._ls**2)

        # delfi
        delfi = self._delfi_by_delpk(t, tau, unit_norm, A)
        ddelfi = self._ddelfi_by_del_p_k1_p_k2(t, tau, unit_norm)
        delfi_sq = compute_delfi_sq_jit(delfi)

        # Extend to N x 3 x 3
        s_xis_i = s_xis[:, :, None]
        s_xis_j = s_xis[:, None, :]
        s_initxis_i = s_initxis[:, :, None]
        s_initxis_j = s_initxis[:, None, :]

        # 1: dp (N x 9)
        dp_coeff_1 = -ci_cj * s_xis_i
        dp_coeff_2 = -ci_cj * s_xis_j
        dp_coeff_3 = 2 * ci_init_cj * s_initxis_j

        dp_1 = compute_dp_jit(dp_coeff_1, dp_coeff_2,
                              dp_coeff_3, t_dot_t_sq, t_dot_init_t_sq, delfi)
        dp_2 = compute_dp_jit(dp_coeff_1, dp_coeff_2,
                              dp_coeff_3, t_norms_sq_sq, t_norms_sq_ls_sq, delfi)

        # 2: dx (N x 3)
        dx_coeff_1 = ci_sj * c[:, :, None] * s_xis_i
        dx_coeff_2 = -ci_sj * self._init_cs[:, :, None] * s_initxis_i

        def compute_dx(factor1, factor2):
            return np.sum(dx_coeff_1 * factor1 + dx_coeff_2 * factor2, axis=1)
        dx_1 = compute_dx(t_dot_t_sq, t_dot_init_t_sq)
        dx_2 = compute_dx(t_norms_sq_sq, t_norms_sq_ls_sq)

        grad_e1 = np.hstack([dp_1, dx_1])
        grad_e2 = np.hstack([dp_2, dx_2])

        gradE_with_stiff = self._kb[..., None] * \
            ((1-self._nu)[..., None] * grad_e1 +
             self._nu[..., None]*grad_e2) * self._A[..., None]

        # Hessian - nightmare :(

        # 1: d/dx^2 (N x 3 x 3)
        tri_inds = np.triu_indices(3)
        ddx_coeff = 2 * ci_cj[:, tri_inds] * si_sj[:, tri_inds]

        def compute_dxdx(factor):
            """ Compute upper triangle (66% computation) """
            dxdx_tri = ddx_coeff * factor[:, tri_inds]

            dxdx = np.zeros((N, 3, 3))
            dxdx[:, tri_inds] = dxdx_tri
            dxdx[:, tri_inds] = dxdx_tri
            return dxdx

        dxdx_1 = compute_dxdx(t_dot_t_sq)
        dxdx_2 = compute_dxdx(t_norms_sq_sq)

        # 2: d/dp^2 (N x 9 x 9)
        dpdp_1 = compute_dpdp_jit(dp_coeff_1, dp_coeff_2, dp_coeff_3,
                                  t_dot_t_sq, t_dot_init_t_sq, ddelfi, delfi_sq, ci_cj)
        dpdp_2 = compute_dpdp_jit(dp_coeff_1, dp_coeff_2, dp_coeff_3,
                                  t_norms_sq_sq, t_norms_sq_ls_sq, ddelfi, delfi_sq, ci_cj)

        # 3: d/dxdp (N x 9 x 3)
        dxdp_coeff_1 = -2 * c[:, :, None] * \
            self._s_s[:, :, None] * c[:, None, :]

        def compute_dxdp(factor):
            """Compute a N x 3 x 9 matrix (for upper triangle)"""
            return np.einsum('nij,nljc->nlci', dxdp_coeff_1 * factor, delfi).reshape(-1, 9, 3)

        dxdp_1 = compute_dxdp(t_dot_t_sq)
        dxdp_2 = compute_dxdp(t_norms_sq_sq)

        # 4: Arrange Hessian
        def get_hess_blocks(TL, TR, BR):
            return np.concatenate([np.concatenate([TL, TR], axis=2),
                                   np.concatenate([np.transpose(TR, (0, 2, 1)), BR], axis=2)], axis=1)

        hess_e1 = get_hess_blocks(dpdp_1, dxdp_1, dxdx_1)
        hess_e2 = get_hess_blocks(dpdp_2, dxdp_2, dxdx_2)

        hessE_with_stiff = self._kb[..., None, None] * \
            ((1-self._nu)[..., None, None] * hess_e1 +
             self._nu[..., None, None]*hess_e2) * self._A[..., None, None]

        # Accumulate into n_dof matrix
        n_dof = state.q.shape[0]
        Fs = np.zeros(n_dof)
        np.add.at(Fs, self._ind, -gradE_with_stiff)

        if sparse:
            Js = sp.coo_matrix((-hessE_with_stiff.ravel(),
                                (self._rows, self._cols)),
                               shape=(n_dof, n_dof)).tocsr()
        else:
            Js = np.zeros((n_dof, n_dof))
            np.add.at(Js, (self._ind[:, :, None],
                           self._ind[:, None, :]), -hessE_with_stiff)

        return Fs, Js
