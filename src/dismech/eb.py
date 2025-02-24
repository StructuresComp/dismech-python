import numpy as np


def cross_mat(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def gradEs_hessEs_struct(n_dof, ind, node0p, node1p, stretch_spring):
    l_k = stretch_spring.ref_len
    EA = stretch_spring.EA

    # Convert nodes to numpy arrays
    node0p = np.array(node0p).flatten()
    node1p = np.array(node1p).flatten()

    # Initialize gradient and Hessian
    dF = np.zeros(n_dof)
    dJ = np.zeros((n_dof, n_dof))

    # Edge vector and properties
    edge = node1p - node0p  # 3x1 vector
    edge_len = np.linalg.norm(edge)
    tangent = edge / edge_len
    epsX = edge_len / l_k - 1.0

    # Gradient computation
    dF_unit = EA * tangent * epsX
    dF[ind[0:3]] = -dF_unit
    dF[ind[3:6]] = dF_unit

    # Hessian computation
    Id3 = np.eye(3)
    term1 = (1.0 / l_k - 1.0 / edge_len) * Id3
    term2 = (1.0 / edge_len) * np.outer(edge, edge) / (edge_len ** 2)
    M = EA * (term1 + term2)

    # Assign blocks to Hessian
    rows_0 = ind[0:3]
    rows_1 = ind[3:6]
    dJ[np.ix_(rows_0, rows_0)] = M
    dJ[np.ix_(rows_1, rows_1)] = M
    dJ[np.ix_(rows_0, rows_1)] = -M
    dJ[np.ix_(rows_1, rows_0)] = -M

    return dF, dJ


def gradEs_hessEs_struct_vectorized(node0p, node1p, l_k, EA):
    n_springs = node0p.shape[0]

    edge = node1p - node0p
    edge_len = np.linalg.norm(edge, axis=1)
    tangent = edge / edge_len[:, None]
    epsX = edge_len / l_k - 1.0

    # Gradient computation
    dF_unit = EA[:, None] * tangent * epsX[:, None]
    dF_springs = np.concatenate((-dF_unit, dF_unit), axis=1)

    # Hessian computation
    Id3 = np.eye(3)
    edge_outer = np.einsum('...i,...j->...ij', edge, edge)
    edge_len_cubed = edge_len ** 3

    term1 = (1.0 / l_k - 1.0 / edge_len)[:, None, None] * Id3[None, :, :]
    term2 = edge_outer / edge_len_cubed[:, None, None]
    M = EA[:, None, None] * (term1 + term2)

    H_blocks = np.zeros((n_springs, 6, 6))
    H_blocks[:, :3, :3] = M
    H_blocks[:, 3:, 3:] = M
    H_blocks[:, :3, 3:] = -M
    H_blocks[:, 3:, :3] = -M

    return dF_springs, H_blocks


def gradEb_hessEb_panetta(n_dof, ind, node0, node1, node2, m1e, m2e, m1f, m2f, bend_twist_spring):
    kappa_bar = bend_twist_spring.kappa_bar
    l_k = bend_twist_spring.voronoi_len
    EI1 = bend_twist_spring.stiff_EI[0]
    EI2 = bend_twist_spring.stiff_EI[1]

    # Convert inputs to numpy arrays
    node0 = np.array(node0).flatten()
    node1 = np.array(node1).flatten()
    node2 = np.array(node2).flatten()
    m1e = np.array(m1e).flatten()
    m2e = np.array(m2e).flatten()
    m1f = np.array(m1f).flatten()
    m2f = np.array(m2f).flatten()

    # Edge vectors and norms
    ee = node1 - node0
    ef = node2 - node1
    norm_e = np.linalg.norm(ee)
    norm_f = np.linalg.norm(ef)
    te = ee / norm_e
    tf = ef / norm_f

    # Curvature binormal and related quantities
    chi = 1.0 + np.dot(te, tf)
    kb = 2.0 * np.cross(te, tf) / chi
    tilde_t = (te + tf) / chi
    tilde_d1 = (m1e + m1f) / chi
    tilde_d2 = (m2e + m2f) / chi

    # Curvatures
    kappa1 = 0.5 * np.dot(kb, m2e + m2f)
    kappa2 = -0.5 * np.dot(kb, m1e + m1f)

    # First derivatives
    Dkappa1De = (1.0/norm_e) * (-kappa1*tilde_t + np.cross(tf, tilde_d2))
    Dkappa1Df = (1.0/norm_f) * (-kappa1*tilde_t - np.cross(te, tilde_d2))
    Dkappa2De = (1.0/norm_e) * (-kappa2*tilde_t - np.cross(tf, tilde_d1))
    Dkappa2Df = (1.0/norm_f) * (-kappa2*tilde_t + np.cross(te, tilde_d1))

    # Gradient assembly
    gradKappa = np.zeros((n_dof, 2))
    gradKappa[ind[0:3], 0] = -Dkappa1De
    gradKappa[ind[3:6], 0] = Dkappa1De - Dkappa1Df
    gradKappa[ind[6:9], 0] = Dkappa1Df
    gradKappa[ind[0:3], 1] = -Dkappa2De
    gradKappa[ind[3:6], 1] = Dkappa2De - Dkappa2Df
    gradKappa[ind[6:9], 1] = Dkappa2Df
    gradKappa[ind[9], 0] = -0.5 * np.dot(kb, m1e)
    gradKappa[ind[10], 0] = -0.5 * np.dot(kb, m1f)
    gradKappa[ind[9], 1] = -0.5 * np.dot(kb, m2e)
    gradKappa[ind[10], 1] = -0.5 * np.dot(kb, m2f)

    # Second derivatives
    DDkappa1 = np.zeros((n_dof, n_dof))
    DDkappa2 = np.zeros((n_dof, n_dof))
    norm2_e, norm2_f = norm_e**2, norm_f**2
    Id3 = np.eye(3)

    # Kappa1 second derivatives
    tt_o_tt = np.outer(tilde_t, tilde_t)
    tf_c_d2t = np.cross(tf, tilde_d2)
    tf_c_d2t_o_tt = np.outer(tf_c_d2t, tilde_t)
    tt_o_tf_c_d2t = np.outer(tilde_t, tf_c_d2t)
    kb_o_d2e = np.outer(kb, m2e)
    D2kappa1De2 = (1/norm2_e)*(2*kappa1*tt_o_tt - tf_c_d2t_o_tt - tt_o_tf_c_d2t) \
        - (kappa1/(chi*norm2_e))*(Id3 - np.outer(te, te)) \
        + (1/(2*norm2_e))*kb_o_d2e

    te_c_d2t = np.cross(te, tilde_d2)
    te_c_d2t_o_tt = np.outer(te_c_d2t, tilde_t)
    tt_o_te_c_d2t = np.outer(tilde_t, te_c_d2t)
    kb_o_d2f = np.outer(kb, m2f)
    D2kappa1Df2 = (1/norm2_f)*(2*kappa1*tt_o_tt + te_c_d2t_o_tt + tt_o_te_c_d2t) \
        - (kappa1/(chi*norm2_f))*(Id3 - np.outer(tf, tf)) \
        + (1/(2*norm2_f))*kb_o_d2f

    te_o_tf = np.outer(te, tf)
    D2kappa1DeDf = (-kappa1/(chi*norm_e*norm_f))*(Id3 + te_o_tf) \
        + (1/(norm_e*norm_f))*(2*kappa1*tt_o_tt -
                               tf_c_d2t_o_tt + tt_o_te_c_d2t - cross_mat(tilde_d2))
    D2kappa1DfDe = D2kappa1DeDf.T

    # Kappa2 second derivatives
    tf_c_d1t = np.cross(tf, tilde_d1)
    tf_c_d1t_o_tt = np.outer(tf_c_d1t, tilde_t)
    tt_o_tf_c_d1t = np.outer(tilde_t, tf_c_d1t)
    kb_o_d1e = np.outer(kb, m1e)
    D2kappa2De2 = (1/norm2_e)*(2*kappa2*tt_o_tt + tf_c_d1t_o_tt + tt_o_tf_c_d1t) \
        - (kappa2/(chi*norm2_e))*(Id3 - np.outer(te, te)) \
        - (1/(2*norm2_e))*kb_o_d1e

    te_c_d1t = np.cross(te, tilde_d1)
    te_c_d1t_o_tt = np.outer(te_c_d1t, tilde_t)
    tt_o_te_c_d1t = np.outer(tilde_t, te_c_d1t)
    kb_o_d1f = np.outer(kb, m1f)
    D2kappa2Df2 = (1/norm2_f)*(2*kappa2*tt_o_tt - te_c_d1t_o_tt - tt_o_te_c_d1t) \
        - (kappa2/(chi*norm2_f))*(Id3 - np.outer(tf, tf)) \
        - (1/(2*norm2_f))*kb_o_d1f

    D2kappa2DeDf = (-kappa2/(chi*norm_e*norm_f))*(Id3 + np.outer(te, tf)) \
        + (1/(norm_e*norm_f))*(2*kappa2*tt_o_tt +
                               tf_c_d1t_o_tt - tt_o_te_c_d1t + cross_mat(tilde_d1))
    D2kappa2DfDe = D2kappa2DeDf.T

    # Twist terms
    D2kappa1Dthetae2 = -0.5 * np.dot(kb, m2e)
    D2kappa1Dthetaf2 = -0.5 * np.dot(kb, m2f)
    D2kappa2Dthetae2 = 0.5 * np.dot(kb, m1e)
    D2kappa2Dthetaf2 = 0.5 * np.dot(kb, m1f)

    # Coupled terms
    D2kappa1DeDthetae = (1/norm_e)*(0.5*np.dot(kb, m1e) *
                                    tilde_t - (1/chi)*np.cross(tf, m1e))
    D2kappa1DeDthetaf = (1/norm_e)*(0.5*np.dot(kb, m1f) *
                                    tilde_t - (1/chi)*np.cross(tf, m1f))
    D2kappa1DfDthetae = (1/norm_f)*(0.5*np.dot(kb, m1e) *
                                    tilde_t + (1/chi)*np.cross(te, m1e))
    D2kappa1DfDthetaf = (1/norm_f)*(0.5*np.dot(kb, m1f) *
                                    tilde_t + (1/chi)*np.cross(te, m1f))

    D2kappa2DeDthetae = (1/norm_e)*(0.5*np.dot(kb, m2e) *
                                    tilde_t - (1/chi)*np.cross(tf, m2e))
    D2kappa2DeDthetaf = (1/norm_e)*(0.5*np.dot(kb, m2f) *
                                    tilde_t - (1/chi)*np.cross(tf, m2f))
    D2kappa2DfDthetae = (1/norm_f)*(0.5*np.dot(kb, m2e) *
                                    tilde_t + (1/chi)*np.cross(te, m2e))
    D2kappa2DfDthetaf = (1/norm_f)*(0.5*np.dot(kb, m2f) *
                                    tilde_t + (1/chi)*np.cross(te, m2f))

    # Hessian assembly
    def assign_blocks(DDkappa, D2De2, D2DeDf, D2DfDe, D2Df2, D2t1, D2t2, D2ct):
        indices = [(0, 3), (3, 6), (6, 9), [9], [10]]

        # Curvature terms
        DDkappa[np.ix_(ind[0:3], ind[0:3])] = D2De2
        DDkappa[np.ix_(ind[0:3], ind[3:6])] = -D2De2 + D2DeDf
        DDkappa[np.ix_(ind[0:3], ind[6:9])] = -D2DeDf
        DDkappa[np.ix_(ind[3:6], ind[0:3])] = -D2De2 + D2DfDe
        DDkappa[np.ix_(ind[3:6], ind[3:6])] = D2De2 - D2DeDf - D2DfDe + D2Df2
        DDkappa[np.ix_(ind[3:6], ind[6:9])] = D2DeDf - D2Df2
        DDkappa[np.ix_(ind[6:9], ind[0:3])] = -D2DfDe
        DDkappa[np.ix_(ind[6:9], ind[3:6])] = D2DfDe - D2Df2
        DDkappa[np.ix_(ind[6:9], ind[6:9])] = D2Df2

        # Twist terms
        DDkappa[ind[9], ind[9]] = D2t1
        DDkappa[ind[10], ind[10]] = D2t2

        # Coupled terms
        for col_idx in [9, 10]:
            if col_idx == 9:
                ct_terms = D2ct[0]
            else:
                ct_terms = D2ct[1]

            DDkappa[ind[0:3], col_idx] = -ct_terms[0]
            DDkappa[ind[3:6], col_idx] = ct_terms[0] - ct_terms[1]
            DDkappa[ind[6:9], col_idx] = ct_terms[1]
            DDkappa[col_idx, ind[0:3]] = DDkappa[ind[0:3], col_idx].T
            DDkappa[col_idx, ind[3:6]] = DDkappa[ind[3:6], col_idx].T
            DDkappa[col_idx, ind[6:9]] = DDkappa[ind[6:9], col_idx].T

    # Assign blocks for DDkappa1
    assign_blocks(DDkappa1,
                  D2kappa1De2, D2kappa1DeDf, D2kappa1DfDe, D2kappa1Df2,
                  D2kappa1Dthetae2, D2kappa1Dthetaf2,
                  [(D2kappa1DeDthetae, D2kappa1DfDthetae), (D2kappa1DeDthetaf, D2kappa1DfDthetaf)])

    # Assign blocks for DDkappa2
    assign_blocks(DDkappa2,
                  D2kappa2De2, D2kappa2DeDf, D2kappa2DfDe, D2kappa2Df2,
                  D2kappa2Dthetae2, D2kappa2Dthetaf2,
                  [(D2kappa2DeDthetae, D2kappa2DfDthetae), (D2kappa2DeDthetaf, D2kappa2DfDthetaf)])

    # Final energy calculations
    EIMat = np.array([[EI1, 0], [0, EI2]])
    kappa_vector = np.array([kappa1, kappa2])
    dkappa_vector = kappa_vector - kappa_bar
    dF = (gradKappa @ EIMat @ dkappa_vector) / l_k

    dJ = (gradKappa @ EIMat @ gradKappa.T) / l_k
    temp = (dkappa_vector.T @ EIMat) / l_k
    dJ += temp[0]*DDkappa1 + temp[1]*DDkappa2
    dJ = dJ.T

    return dF, dJ


def cross_mat_batch(v):
    """Batch version of cross product matrix"""
    zeros = np.zeros_like(v[:, 0])
    return np.array([
        [zeros, -v[:, 2], v[:, 1]],
        [v[:, 2], zeros, -v[:, 0]],
        [-v[:, 1], v[:, 0], zeros]
    ]).transpose(2, 0, 1)


def gradEb_hessEb_panetta_vectorized(node0, node1, node2,
                                     m1e, m2e, m1f, m2f,
                                     kappa_bar, l_k, EI1, EI2):
    # Input reshaping (batch_size, 3)
    node0 = np.asarray(node0).reshape(-1, 3)
    node1 = np.asarray(node1).reshape(-1, 3)
    node2 = np.asarray(node2).reshape(-1, 3)
    m1e = np.asarray(m1e).reshape(-1, 3)
    m2e = np.asarray(m2e).reshape(-1, 3)
    m1f = np.asarray(m1f).reshape(-1, 3)
    m2f = np.asarray(m2f).reshape(-1, 3)

    n_springs = node0.shape[0]
    Id3 = np.eye(3)[None, :, :]  # For broadcasting

    # Precompute common terms
    ee = node1 - node0
    ef = node2 - node1
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

    # Coupled terms (corrected)
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

    # Similar corrections for D2kappa2 terms
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
        # Position blocks (unchanged)
        DDkappa[:, :3, :3] = D2De2
        DDkappa[:, :3, 3:6] = -D2De2 + D2DeDf
        DDkappa[:, :3, 6:9] = -D2DeDf

        DDkappa[:, 3:6, :3] = -D2De2 + D2DfDe
        DDkappa[:, 3:6, 3:6] = D2De2 - D2DeDf - D2DfDe + D2Df2
        DDkappa[:, 3:6, 6:9] = D2DeDf - D2Df2

        DDkappa[:, 6:9, :3] = -D2DfDe
        DDkappa[:, 6:9, 3:6] = D2DfDe - D2Df2
        DDkappa[:, 6:9, 6:9] = D2Df2

        # Twist terms (unchanged)
        DDkappa[:, 9, 9] = D2t1
        DDkappa[:, 10, 10] = D2t2

        # Corrected coupled terms handling
        # For column 9 (theta_e)
        # Column entries (keep as 3D arrays)
        DDkappa[:, :3, 9:10] = -D2ct[0][0]  # shape (n, 3, 1)
        DDkappa[:, 3:6, 9:10] = D2ct[0][0] - D2ct[0][1]
        DDkappa[:, 6:9, 9:10] = D2ct[0][1]

        # Row entries (transpose and maintain 3D structure)
        DDkappa[:, 9:10, :3] = - \
            D2ct[0][0].transpose(0, 2, 1)  # shape (n, 1, 3)
        DDkappa[:, 9:10, 3:6] = (D2ct[0][0] - D2ct[0][1]).transpose(0, 2, 1)
        DDkappa[:, 9:10, 6:9] = D2ct[0][1].transpose(0, 2, 1)

        # For column 10 (theta_f)
        # Column entries
        DDkappa[:, :3, 10:11] = -D2ct[1][0]
        DDkappa[:, 3:6, 10:11] = D2ct[1][0] - D2ct[1][1]
        DDkappa[:, 6:9, 10:11] = D2ct[1][1]

        # Row entries
        DDkappa[:, 10:11, :3] = -D2ct[1][0].transpose(0, 2, 1)
        DDkappa[:, 10:11, 3:6] = (D2ct[1][0] - D2ct[1][1]).transpose(0, 2, 1)
        DDkappa[:, 10:11, 6:9] = D2ct[1][1].transpose(0, 2, 1)

    # Initialize Hessians
    DDkappa1 = np.zeros((n_springs, 11, 11))
    DDkappa2 = np.zeros((n_springs, 11, 11))

    # Assign blocks for DDkappa1
    batch_assign_blocks(DDkappa1,
                        D2kappa1De2,
                        D2kappa1DeDf,
                        # D2DfDe is transpose of DeDf
                        D2kappa1DeDf.transpose(0, 2, 1),
                        D2kappa1Df2,
                        D2kappa1Dthetae2,
                        D2kappa1Dthetaf2,
                        [(D2kappa1DeDthetae, D2kappa1DfDthetae),
                         (D2kappa1DeDthetaf, D2kappa1DfDthetaf)])

    # Assign blocks for DDkappa2
    batch_assign_blocks(DDkappa2,
                        D2kappa2De2,
                        D2kappa2DeDf,
                        D2kappa2DeDf.transpose(0, 2, 1),
                        D2kappa2Df2,
                        D2kappa2Dthetae2,
                        D2kappa2Dthetaf2,
                        [(D2kappa2DeDthetae, D2kappa2DfDthetae),
                         (D2kappa2DeDthetaf, D2kappa2DfDthetaf)])

    # Final energy computations
    EIMat = np.zeros((n_springs, 2, 2))
    EIMat[:, 0, 0] = EI1  # Shape (n_springs, 2, 2)
    EIMat[:, 1, 1] = EI2
    dkappa = np.stack([kappa1, kappa2], axis=1) - \
        kappa_bar  # Shape (n_springs, 2)

    # Compute forces
    dF_springs = np.einsum('sij,sjk,sk->si', gradKappa,
                           EIMat, dkappa) / l_k[:, None]

    # First term: gradKappa @ EIMat @ gradKappa^T / l_k
    term1 = np.einsum('sij,sjk,slk->sil', gradKappa, EIMat,
                      gradKappa) / l_k[:, None, None]

    # Second term: (dkappa^T EIMat / l_k) * (DDkappa1 + DDkappa2)
    # Compute coefficient matrix (n_springs, 2)
    temp = np.einsum('si,sji->sj', dkappa, EIMat) / l_k[:, None]

    # Stack Hessians (n_springs, 2, 11, 11)
    stacked_hessians = np.stack([DDkappa1, DDkappa2], axis=1)

    # Contract coefficients with Hessians
    term2 = np.einsum('sj,sjkl->skl', temp, stacked_hessians)

    # Combine terms
    dJ_springs = term1 + term2  # Shape (n_springs, 11, 11)

    return dF_springs, dJ_springs

def gradEt_hessEt_panetta_vectorized(n_dof, ind, node0, node1, node2, theta_e, theta_f, refTwist, l_k, GJ, undef_refTwist):
    N = node0.shape[0]  # Number of springs in the batch
    
    # Edge vectors
    ee = node1 - node0
    ef = node2 - node1
    
    # Norms and tangents
    norm_e = np.linalg.norm(ee, axis=1, keepdims=True)
    norm_f = np.linalg.norm(ef, axis=1, keepdims=True)
    te = ee / norm_e
    tf = ef / norm_f
    
    # Dot product and chi
    dot_te_tf = np.sum(te * tf, axis=1, keepdims=True)
    chi = 1.0 + dot_te_tf
    
    # Curvature binormal
    kb = 2.0 * np.cross(te, tf, axis=1) / chi
    
    # tilde_t
    tilde_t = (te + tf) / chi
    
    # Reduce indexing to only the 11 relevant DOFs
    spring_dofs = ind[:, :11]
    
    # Initialize reduced gradTwist (N, 11)
    gradTwist = np.zeros((N, 11))
    gradTwist[:, 0:3] = (-0.5 / norm_e) * kb
    gradTwist[:, 6:9] = (0.5 / norm_f) * kb
    gradTwist[:, 3:6] = - (gradTwist[:, 0:3] + gradTwist[:, 6:9])
    gradTwist[:, 9] = -1.0
    gradTwist[:, 10] = 1.0
    
    # Cross product matrices
    cross_te = cross_mat_batch(te)
    cross_tf = cross_mat_batch(tf)
    
    # Compute second derivatives
    norm2_e = norm_e ** 2
    norm2_f = norm_f ** 2
    norm_e_norm_f = norm_e * norm_f

    D2mDe2 = (-0.5 / norm2_e)[:, :, np.newaxis] * (
        np.einsum('ni,nj->nij', kb, te + tilde_t) + (2.0 / chi)[:, :, np.newaxis] * cross_tf
    )
    D2mDf2 = (-0.5 / norm2_f)[:, :, np.newaxis] * (
        np.einsum('ni,nj->nij', kb, tf + tilde_t) + (2.0 / chi)[:, :, np.newaxis] * cross_te
    )
    D2mDeDf = (0.5 / norm_e_norm_f)[:, :, np.newaxis] * (
        (2.0 / chi)[:, :, np.newaxis] * cross_te - np.einsum('ni,nj->nij', kb, tilde_t)
    )
    D2mDfDe = (0.5 / norm_e_norm_f)[:, :, np.newaxis] * (
        (-2.0 / chi)[:, :, np.newaxis] * cross_tf - np.einsum('ni,nj->nij', kb, tilde_t)
    )
    
    # Assemble reduced DDtwist (N, 11, 11)
    DDtwist = np.zeros((N, 11, 11))
    
    DDtwist[:, 0:3, 0:3] = D2mDe2
    DDtwist[:, 0:3, 3:6] = -D2mDe2 + D2mDeDf
    DDtwist[:, 3:6, 0:3] = -D2mDe2 + D2mDfDe
    DDtwist[:, 3:6, 3:6] = D2mDe2 - (D2mDeDf + D2mDfDe) + D2mDf2
    DDtwist[:, 0:3, 6:9] = -D2mDeDf
    DDtwist[:, 6:9, 0:3] = -D2mDfDe
    DDtwist[:, 6:9, 3:6] = D2mDfDe - D2mDf2
    DDtwist[:, 3:6, 6:9] = D2mDeDf - D2mDf2
    DDtwist[:, 6:9, 6:9] = D2mDf2

    # Integrated twist
    integratedTwist = theta_f - theta_e + refTwist - undef_refTwist
    
    # Compute dF (N, 11)
    scaling_factor = (GJ / l_k) * integratedTwist
    dF = scaling_factor[:, np.newaxis] * gradTwist
    
    # Compute dJ (N, 11, 11)
    term1 = integratedTwist[:, np.newaxis, np.newaxis] * DDtwist
    term2 = np.einsum('ni,nj->nij', gradTwist, gradTwist)
    dJ = (GJ / l_k)[:, np.newaxis, np.newaxis] * (term1 + term2)
    
    return dF, dJ
