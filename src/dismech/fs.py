import numpy as np

from .eb import gradEs_hessEs_struct, gradEb_hessEb_panetta


def get_fs_js(robot, q):
    n_dof = robot.n_dof

    Fs = np.zeros(n_dof)
    Js = np.zeros((n_dof, n_dof))

    for spring in robot.stretch_springs:
        n0, n1 = spring.node_indices
        n0p = q[robot.map_node_to_dof(n0)]
        n1p = q[robot.map_node_to_dof(n1)]
        ind = spring.indices

        dF, dJ = gradEs_hessEs_struct(
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

        dF, dJ = gradEb_hessEb_panetta(
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
