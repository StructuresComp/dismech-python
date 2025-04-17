import numpy as np
import typing

from ..springs import BendTwistSpring
from ..state import RobotState
from .elastic_energy import ElasticEnergy


class BendEnergy(ElasticEnergy):
    def __init__(self, springs: typing.List[BendTwistSpring], initial_state: RobotState):
        super().__init__(
            np.array([s.stiff_EI / s.voronoi_len for s in springs]),
            np.array([s.nodes_ind for s in springs]),
            np.array([s.ind for s in springs]),
            initial_state,
        )
        self._sgn = np.array([s.sgn for s in springs])
        self._edges_ind = np.array([s.edges_ind for s in springs])

        N = len(springs)

        # Create sign matrices
        self._sign_grad = np.ones((N, 11))
        for dof_idx, signs in [(9, self._sgn[:, 0]), (10, self._sgn[:, 1])]:
            self._sign_grad[:, dof_idx] = signs
        self._sign_hess = self._sign_grad[:, :, None] * \
            self._sign_grad[:, None, :]

    def _get_adjusted_material_directors(self, m1: np.ndarray, m2: np.ndarray):
        m1e = m1[self._edges_ind[:, 0]]
        m2e = m2[self._edges_ind[:, 0]] * self._sgn[:, 0, None]
        m1f = m1[self._edges_ind[:, 1]]
        m2f = m2[self._edges_ind[:, 1]] * self._sgn[:, 1, None]
        return m1e, m2e, m1f, m2f

    def get_strain(self, state: RobotState) -> np.ndarray:
        n0p, n1p, n2p = self._get_node_pos(state.q)
        m1e, m2e, m1f, m2f = self._get_adjusted_material_directors(state.m1, state.m2)

        # Precompute common terms
        ee = n1p - n0p
        ef = n2p - n1p
        norm_e = np.linalg.norm(ee, axis=1)
        norm_f = np.linalg.norm(ef, axis=1)
        te = ee / norm_e[:, None]
        tf = ef / norm_f[:, None]

        chi = 1.0 + np.sum(te * tf, axis=1)
        chi_inv = 1.0 / chi
        kb = 2.0 * np.cross(te, tf) * chi_inv[:, None]

        kappa1 = 0.5 * np.sum(kb * (m2e + m2f), axis=1)
        kappa2 = -0.5 * np.sum(kb * (m1e + m1f), axis=1)

        return np.stack([kappa1, kappa2], axis=-1)

    def grad_hess_strain(self, state: RobotState) -> typing.Tuple[np.ndarray, np.ndarray]:
        n0p, n1p, n2p = self._get_node_pos(state.q)
        m1e, m2e, m1f, m2f = self._get_adjusted_material_directors(state.m1, state.m2)
        n_springs = n0p.shape[0]
        
        Id3 = np.eye(3)[None, :, :]  # For broadcasting

        # Precompute common terms
        ee = n1p - n0p
        ef = n2p - n1p
        norm_e = np.linalg.norm(ee, axis=1)
        norm_f = np.linalg.norm(ef, axis=1)
        te = ee / norm_e[:, None]
        tf = ef / norm_f[:, None]

        chi = 1.0 + np.sum(te * tf, axis=1)
        chi_inv = 1.0 / chi
        kb = 2.0 * np.cross(te, tf) * chi_inv[:, None]

        tilde_t = (te + tf) * chi_inv[:, None]
        tilde_d1 = (m1e + m1f) * chi_inv[:, None]
        tilde_d2 = (m2e + m2f) * chi_inv[:, None]

        # Curvatures
        kappa1 = 0.5 * np.sum(kb * (m2e + m2f), axis=1)
        kappa2 = -0.5 * np.sum(kb * (m1e + m1f), axis=1)

        # First derivatives
        Dkappa1De = (1.0 / norm_e[:, None]) * \
            (-kappa1[:, None] * tilde_t + np.cross(tf, tilde_d2))
        Dkappa1Df = (1.0 / norm_f[:, None]) * \
            (-kappa1[:, None] * tilde_t - np.cross(te, tilde_d2))
        Dkappa2De = (1.0 / norm_e[:, None]) * \
            (-kappa2[:, None] * tilde_t - np.cross(tf, tilde_d1))
        Dkappa2Df = (1.0 / norm_f[:, None]) * \
            (-kappa2[:, None] * tilde_t + np.cross(te, tilde_d1))

        # Gradient assembly
        gradKappa = np.zeros((n_springs, 11, 2))
        gradKappa[:, 0:3, 0] = -Dkappa1De
        gradKappa[:, 3:6, 0] = Dkappa1De - Dkappa1Df
        gradKappa[:, 6:9, 0] = Dkappa1Df
        gradKappa[:, 0:3, 1] = -Dkappa2De
        gradKappa[:, 3:6, 1] = Dkappa2De - Dkappa2Df
        gradKappa[:, 6:9, 1] = Dkappa2Df

        # Twist terms
        gradKappa[:, 9, 0] = -0.5 * np.sum(kb * m1e, axis=1)
        gradKappa[:, 10, 0] = -0.5 * np.sum(kb * m1f, axis=1)
        gradKappa[:, 9, 1] = -0.5 * np.sum(kb * m2e, axis=1)
        gradKappa[:, 10, 1] = -0.5 * np.sum(kb * m2f, axis=1)

        # Second derivatives
        norm2_e = norm_e**2
        norm2_f = norm_f**2

        # Helper functions for batch outer products
        def batch_outer(a, b):
            return np.einsum('...i,...j->...ij', a, b)

        def cross_mat_batch(v):
            """Batch version of cross product matrix"""
            zeros = np.zeros_like(v[:, 0])
            return np.array([
                [zeros, -v[:, 2], v[:, 1]],
                [v[:, 2], zeros, -v[:, 0]],
                [-v[:, 1], v[:, 0], zeros]
            ]).transpose(2, 0, 1)

        # Kappa1 second derivatives
        tt_o_tt = batch_outer(tilde_t, tilde_t)
        tf_c_d2t = np.cross(tf, tilde_d2)
        tf_c_d2t_o_tt = batch_outer(tf_c_d2t, tilde_t)
        tt_o_tf_c_d2t = batch_outer(tilde_t, tf_c_d2t)
        kb_o_d2e = batch_outer(kb, m2e)

        D2kappa1De2 = (1/norm2_e[:, None, None])*(2*kappa1[:, None, None]*tt_o_tt - tf_c_d2t_o_tt - tt_o_tf_c_d2t) \
            - (kappa1[:, None, None]/(chi[:, None, None]*norm2_e[:, None, None]))*(Id3 - batch_outer(te, te)) \
            + (1/(2*norm2_e[:, None, None]))*kb_o_d2e

        te_c_d2t = np.cross(te, tilde_d2)
        te_c_d2t_o_tt = batch_outer(te_c_d2t, tilde_t)
        tt_o_te_c_d2t = batch_outer(tilde_t, te_c_d2t)
        kb_o_d2f = batch_outer(kb, m2f)

        D2kappa1Df2 = (1/norm2_f[:, None, None])*(2*kappa1[:, None, None]*tt_o_tt + te_c_d2t_o_tt + tt_o_te_c_d2t) \
            - (kappa1[:, None, None]/(chi[:, None, None]*norm2_f[:, None, None]))*(Id3 - batch_outer(tf, tf)) \
            + (1/(2*norm2_f[:, None, None]))*kb_o_d2f

        te_o_tf = batch_outer(te, tf)
        D2kappa1DeDf = (-kappa1[:, None, None]/(chi[:, None, None]*norm_e[:, None, None]*norm_f[:, None, None]))*(Id3 + te_o_tf) \
            + (1/(norm_e[:, None, None]*norm_f[:, None, None]))*(2*kappa1[:, None, None]*tt_o_tt
                                                                 - tf_c_d2t_o_tt + tt_o_te_c_d2t - cross_mat_batch(tilde_d2))

        # Kappa2 second derivatives
        tf_c_d1t = np.cross(tf, tilde_d1)
        tf_c_d1t_o_tt = batch_outer(tf_c_d1t, tilde_t)
        tt_o_tf_c_d1t = batch_outer(tilde_t, tf_c_d1t)
        kb_o_d1e = batch_outer(kb, m1e)

        D2kappa2De2 = (1/norm2_e[:, None, None])*(2*kappa2[:, None, None]*tt_o_tt + tf_c_d1t_o_tt + tt_o_tf_c_d1t) \
            - (kappa2[:, None, None]/(chi[:, None, None]*norm2_e[:, None, None]))*(Id3 - batch_outer(te, te)) \
            - (1/(2*norm2_e[:, None, None]))*kb_o_d1e

        te_c_d1t = np.cross(te, tilde_d1)
        te_c_d1t_o_tt = batch_outer(te_c_d1t, tilde_t)
        tt_o_te_c_d1t = batch_outer(tilde_t, te_c_d1t)
        kb_o_d1f = batch_outer(kb, m1f)

        D2kappa2Df2 = (1/norm2_f[:, None, None])*(2*kappa2[:, None, None]*tt_o_tt - te_c_d1t_o_tt - tt_o_te_c_d1t) \
            - (kappa2[:, None, None]/(chi[:, None, None]*norm2_f[:, None, None]))*(Id3 - batch_outer(tf, tf)) \
            - (1/(2*norm2_f[:, None, None]))*kb_o_d1f

        D2kappa2DeDf = (-kappa2[:, None, None]/(chi[:, None, None]*norm_e[:, None, None]*norm_f[:, None, None]))*(Id3 + te_o_tf) \
            + (1/(norm_e[:, None, None]*norm_f[:, None, None]))*(2*kappa2[:, None, None]*tt_o_tt
                                                                 + tf_c_d1t_o_tt - tt_o_te_c_d1t + cross_mat_batch(tilde_d1))

        # Twist terms
        D2kappa1Dthetae2 = -0.5 * np.sum(kb * m2e, axis=1)
        D2kappa1Dthetaf2 = -0.5 * np.sum(kb * m2f, axis=1)
        D2kappa2Dthetae2 = 0.5 * np.sum(kb * m1e, axis=1)
        D2kappa2Dthetaf2 = 0.5 * np.sum(kb * m1f, axis=1)

        # Coupled terms
        D2kappa1DeDthetae = (1/norm_e[:, None, None]) * (
            0.5 * np.einsum('sij,sjk->sik', batch_outer(kb, m1e),
                            tilde_t[:, :, None])
            - (1/chi[:, None, None]) * np.cross(tf, m1e)[:, :, None]
        )

        D2kappa1DeDthetaf = (1/norm_e[:, None, None]) * (
            0.5 * np.einsum('sij,sjk->sik', batch_outer(kb, m1f),
                            tilde_t[:, :, None])
            - (1/chi[:, None, None]) * np.cross(tf, m1f)[:, :, None]
        )

        D2kappa1DfDthetae = (1/norm_f[:, None, None]) * (
            0.5 * np.einsum('sij,sjk->sik', batch_outer(kb, m1e),
                            tilde_t[:, :, None])
            + (1/chi[:, None, None]) * np.cross(te, m1e)[:, :, None]
        )

        D2kappa1DfDthetaf = (1/norm_f[:, None, None]) * (
            0.5 * np.einsum('sij,sjk->sik', batch_outer(kb, m1f),
                            tilde_t[:, :, None])
            + (1/chi[:, None, None]) * np.cross(te, m1f)[:, :, None]
        )

        D2kappa2DeDthetae = (1/norm_e[:, None, None]) * (
            0.5 * np.einsum('sij,sjk->sik', batch_outer(kb, m2e),
                            tilde_t[:, :, None])
            - (1/chi[:, None, None]) * np.cross(tf, m2e)[:, :, None]
        )

        D2kappa2DeDthetaf = (1/norm_e[:, None, None]) * (
            0.5 * np.einsum('sij,sjk->sik', batch_outer(kb, m2f),
                            tilde_t[:, :, None])
            - (1/chi[:, None, None]) * np.cross(tf, m2f)[:, :, None]
        )

        D2kappa2DfDthetae = (1/norm_f[:, None, None]) * (
            0.5 * np.einsum('sij,sjk->sik', batch_outer(kb, m2e),
                            tilde_t[:, :, None])
            + (1/chi[:, None, None]) * np.cross(te, m2e)[:, :, None]
        )

        D2kappa2DfDthetaf = (1/norm_f[:, None, None]) * (
            0.5 * np.einsum('sij,sjk->sik', batch_outer(kb, m2f),
                            tilde_t[:, :, None])
            + (1/chi[:, None, None]) * np.cross(te, m2f)[:, :, None]
        )

        def batch_assign_blocks(DDkappa, D2De2, D2DeDf, D2DfDe, D2Df2, D2t1, D2t2, D2ct):
            # Position blocks
            DDkappa[:, :3, :3] = D2De2
            DDkappa[:, :3, 3:6] = -D2De2 + D2DeDf
            DDkappa[:, :3, 6:9] = -D2DeDf

            DDkappa[:, 3:6, :3] = -D2De2 + D2DfDe
            DDkappa[:, 3:6, 3:6] = D2De2 - D2DeDf - D2DfDe + D2Df2
            DDkappa[:, 3:6, 6:9] = D2DeDf - D2Df2

            DDkappa[:, 6:9, :3] = -D2DfDe
            DDkappa[:, 6:9, 3:6] = D2DfDe - D2Df2
            DDkappa[:, 6:9, 6:9] = D2Df2

            # Twist terms
            DDkappa[:, 9, 9] = D2t1
            DDkappa[:, 10, 10] = D2t2
            DDkappa[:, :3, 9:10] = -D2ct[0][0]
            DDkappa[:, 3:6, 9:10] = D2ct[0][0] - D2ct[0][1]
            DDkappa[:, 6:9, 9:10] = D2ct[0][1]

            # Row entries
            DDkappa[:, 9:10, :3] = - \
                D2ct[0][0].transpose(0, 2, 1)
            DDkappa[:, 9:10, 3:6] = (
                D2ct[0][0] - D2ct[0][1]).transpose(0, 2, 1)
            DDkappa[:, 9:10, 6:9] = D2ct[0][1].transpose(0, 2, 1)
            DDkappa[:, :3, 10:11] = -D2ct[1][0]
            DDkappa[:, 3:6, 10:11] = D2ct[1][0] - D2ct[1][1]
            DDkappa[:, 6:9, 10:11] = D2ct[1][1]

            DDkappa[:, 10:11, :3] = -D2ct[1][0].transpose(0, 2, 1)
            DDkappa[:, 10:11, 3:6] = (
                D2ct[1][0] - D2ct[1][1]).transpose(0, 2, 1)
            DDkappa[:, 10:11, 6:9] = D2ct[1][1].transpose(0, 2, 1)

        # Initialize Hessians
        DDkappa1 = np.zeros((n_springs, 11, 11))
        DDkappa2 = np.zeros((n_springs, 11, 11))

        batch_assign_blocks(DDkappa1,
                            D2kappa1De2,
                            D2kappa1DeDf,
                            D2kappa1DeDf.transpose(0, 2, 1),
                            D2kappa1Df2,
                            D2kappa1Dthetae2,
                            D2kappa1Dthetaf2,
                            [(D2kappa1DeDthetae, D2kappa1DfDthetae),
                             (D2kappa1DeDthetaf, D2kappa1DfDthetaf)])

        batch_assign_blocks(DDkappa2,
                            D2kappa2De2,
                            D2kappa2DeDf,
                            D2kappa2DeDf.transpose(0, 2, 1),
                            D2kappa2Df2,
                            D2kappa2Dthetae2,
                            D2kappa2Dthetaf2,
                            [(D2kappa2DeDthetae, D2kappa2DfDthetae),
                             (D2kappa2DeDthetaf, D2kappa2DfDthetaf)])

        return gradKappa, np.stack([DDkappa1, DDkappa2], axis=-1)
