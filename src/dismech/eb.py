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
