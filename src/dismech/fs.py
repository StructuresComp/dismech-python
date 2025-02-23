import numpy as np

from . import eb


def get_fs_js(robot, q):
    n_dof = robot.n_dof

    Fs = np.zeros(n_dof)
    Js = np.zeros((n_dof, n_dof))

    for spring in robot.stretch_springs:
        n0, n1 = spring.node_indices
        n0p = q[robot.map_node_to_dof(n0)]
        n1p = q[robot.map_node_to_dof(n1)]
        ind = spring.indices

        dF, dJ = eb.gradEs_hessEs_struct(
            n_dof, ind, n0p, n1p, spring)

        Fs[ind] -= dF[ind]
        Js[np.ix_(ind, ind)] -= dJ[np.ix_(ind, ind)]

    return Fs, Js


def get_fb_jb(robot, q, m1, m2):
    n_dof = robot.n_dof

    Fb = np.zeros(n_dof)
    Jb = np.zeros((n_dof, n_dof))

    for spring in robot.bend_twist_springs:
        n0, n1, n2 = spring.nodes_ind
        e0, e1 = spring.edges_ind

        n0p = q[robot.map_node_to_dof(n0)]
        n1p = q[robot.map_node_to_dof(n1)]
        n2p = q[robot.map_node_to_dof(n2)]

        m1e = m1[e0]
        m2e = spring.sgn[0] * m2[e0]
        m1f = m1[e1]
        m2f = spring.sgn[1] * m2[e1]

        ind = spring.ind

        dF, dJ = eb.gradEb_hessEb_panetta(
            n_dof, ind, n0p, n1p, n2p, m1e, m2e, m1f, m2f, spring)

        if spring.sgn[0] != 1:
            e0_dof = robot.map_edge_to_dof(e0)
            dF[e0_dof] *= -1
            dJ[e0_dof, :] *= -1
            dJ[:, e0_dof] *= -1

        if spring.sgn[1] != 1:
            e1_dof = robot.map_edge_to_dof(e1)
            dF[e1_dof] *= -1
            dJ[e1_dof] *= -1
            dJ[:, e1_dof] *= -1

        Fb[ind] -= dF[ind]
        Jb[np.ix_(ind, ind)] -= dJ[np.ix_(ind, ind)]

    return Fb, Jb


def get_fs_js_vectorized(robot, q):
    """Vectorized version of stretch spring force/Jacobian calculation"""
    springs = robot.stretch_springs

    # Batch collect all spring data
    n0_indices = np.array([s.node_indices[0] for s in springs])
    n1_indices = np.array([s.node_indices[1] for s in springs])
    all_indices = np.array([s.indices for s in springs])

    # Vectorized position retrieval
    n0_pos = q[robot.map_node_to_dof(n0_indices)]  # shape (n_springs, 3)
    n1_pos = q[robot.map_node_to_dof(n1_indices)]  # shape (n_springs, 3)

    # Vectorized l_k and EA
    l_k = np.array([s.ref_len for s in springs])
    EA = np.array([s.EA for s in springs])

    # Vectorized gradient/hessian calculation
    dF_all, dJ_all = eb.gradEs_hessEs_struct_vectorized(
        n0_pos, n1_pos, l_k, EA)

    # Batch accumulate forces and Jacobians
    Fs = np.zeros(robot.n_dof)
    Js = np.zeros((robot.n_dof, robot.n_dof))

    # Vectorized accumulation using numpy's ufunc.at
    np.add.at(Fs, all_indices, -dF_all)
    for idx, j in zip(all_indices, dJ_all):
        Js[np.ix_(idx, idx)] -= j

    return Fs, Js


def get_fb_jb_vectorized(robot, q, m1, m2):
    """Vectorized version of bend-twist spring force/Jacobian calculation"""
    springs = robot.bend_twist_springs

    # Batch collect all spring data
    node_indices = np.array([(s.nodes_ind[0], s.nodes_ind[1], s.nodes_ind[2])
                             for s in springs])
    edge_indices = np.array(
        [(s.edges_ind[0], s.edges_ind[1]) for s in springs])
    signs = np.array([s.sgn for s in springs])
    all_indices = np.array([s.ind for s in springs])

    # Vectorized position and material director retrieval
    n0_pos = q[robot.map_node_to_dof(node_indices[:, 0])]
    n1_pos = q[robot.map_node_to_dof(node_indices[:, 1])]
    n2_pos = q[robot.map_node_to_dof(node_indices[:, 2])]

    # Vectorized material directors with sign adjustments
    e0_mask = edge_indices[:, 0]
    e1_mask = edge_indices[:, 1]
    m1e = m1[e0_mask] * signs[:, 0, None]
    m2e = m2[e0_mask] * signs[:, 0, None]
    m1f = m1[e1_mask] * signs[:, 1, None]
    m2f = m2[e1_mask] * signs[:, 1, None]

    # Vectorized l_k and EA
    l_k = np.array([s.voronoi_len for s in springs])
    kappa_bar = np.stack([s.kappa_bar for s in springs])
    EI1 = np.array([s.stiff_EI[0] for s in springs])
    EI2 = np.array([s.stiff_EI[1] for s in springs])

    # Batch gradient/hessian calculation
    dF_all, dJ_all = eb.gradEb_hessEb_panetta_vectorized(n0_pos, n1_pos, n2_pos, m1e,
                                                         m2e, m1f, m2f, kappa_bar, l_k, EI1, EI2)

    # Apply sign corrections using vector operations
    sign_mask = signs != 1
    edge_dofs = robot.map_edge_to_dof(edge_indices[sign_mask.any(axis=1)])
    dF_all[edge_dofs] *= -1
    dJ_all[edge_dofs] *= -1
    dJ_all[:, edge_dofs] *= -1

    # Batch accumulate results
    Fb = np.zeros(robot.n_dof)
    Jb = np.zeros((robot.n_dof, robot.n_dof))

    np.add.at(Fb, all_indices, -dF_all)
    for idx, j in zip(all_indices, dJ_all):
        Jb[np.ix_(idx, idx)] -= j

    return Fb, Jb
