import numpy as np
import pytest

import dismech
from dismech.state import RobotState
from dismech.elastics import LoveEnergy, love_constitutive


R_COIL, PITCH, B, C = 2e-3, np.deg2rad(15), 1e-6, 0.8e-6


def _sim():
    return dismech.SimParams(static_sim=True, two_d_sim=False, use_mid_edge=False,
                             use_line_search=False, log_data=False, log_step=1,
                             show_floor=False, dt=1.0, max_iter=50, total_time=1.0,
                             plot_step=1, tol=1e-9, ftol=1e-12, dtol=1e-13)


def _robot(nodes, edges):
    geo = dismech.Geometry(np.asarray(nodes, float), np.asarray(edges), np.empty(0))
    geom = dismech.GeomParams(rod_r0=1e-3, shell_h=0)
    mat = dismech.Material(density=1000, youngs_rod=1e7, youngs_shell=0,
                           poisson_rod=0.3, poisson_shell=0)
    return dismech.SoftRobot(geom, mat, geo, _sim(), dismech.Environment())


def _y_network(n=4, L=0.02, seed=0):
    """Three branches meeting at node 0 (a junction with three edges)."""
    rng = np.random.default_rng(seed)
    dirs = np.array([[1, 0, 0.2], [-0.5, 0.8, -0.1], [-0.4, -0.9, 0.3]])
    nodes, edges = [np.zeros(3)], []
    for d in dirs:
        prev = 0
        for k in range(1, n + 1):
            nodes.append(k * L / n * d / np.linalg.norm(d) + 1e-3 * rng.standard_normal(3))
            edges.append([prev, len(nodes) - 1])
            prev = len(nodes) - 1
    return _robot(nodes, edges)


def _state_at(base, q):
    """State at q with reference frames transported from base (as in a Newton step)."""
    a1, a2 = base.compute_time_parallel(base.state.a1, base.state.q, q)
    m1, m2 = base.compute_material_directors(q, a1, a2)
    rt = base.compute_reference_twist(base.twist_springs, q, a1, base.state.ref_twist)
    return RobotState.init(q, a1, a2, m1, m2, rt, np.empty(0))


@pytest.fixture
def deformed_y_network():
    robot = _y_network().add_love_springs(coil_r=R_COIL, pitch=PITCH, EI=B, GJ=C, chirality=-1)
    stepper = dismech.ImplicitEulerTimeStepper(robot)
    rng = np.random.default_rng(3)
    nn = 3 * robot.n_nodes
    q0 = robot.q0.copy()
    q0[:nn] += 1e-3 * rng.standard_normal(nn)
    q0[nn:] += 0.3 * rng.standard_normal(q0.size - nn)
    base = stepper._finalize_update(robot, q0)
    return robot, stepper.elastic_energies[dismech.LOVE], base


def test_constitutive_derivatives():
    rng = np.random.default_rng(0)
    eps, vth = rng.uniform(-0.3, 1.5, 20), rng.uniform(-200, 200, 20)
    w, R, K, Tee, Tev, Tvv = love_constitutive(eps, vth, R_COIL, PITCH, B, C)
    h = 1e-7
    f = lambda e, v: np.stack(love_constitutive(e, v, R_COIL, PITCH, B, C)[:3])
    de = (f(eps + h, vth) - f(eps - h, vth)) / (2 * h)
    dv = (f(eps, vth + h) - f(eps, vth - h)) / (2 * h)
    assert np.allclose(de[0], R, rtol=1e-6)
    assert np.allclose(dv[0], K, rtol=1e-6)
    assert np.allclose(de[1], Tee, rtol=1e-5)
    assert np.allclose(dv[1], Tev, rtol=1e-5)
    assert np.allclose(de[2], Tev, rtol=1e-5)
    assert np.allclose(dv[2], Tvv, rtol=1e-5)


def test_love_eq40():
    """Stress reproduces Love's eq. (40) for the deformed helix (r1, alpha1)."""
    eps, vth = 0.7, 35.0
    s1 = (1 + eps) * np.sin(PITCH)
    a1 = np.arcsin(s1)
    k = np.cos(PITCH) / R_COIL + vth * np.sin(PITCH)
    r1 = np.cos(a1) / k
    dk = np.cos(a1) ** 2 / r1 - np.cos(PITCH) ** 2 / R_COIL
    dt = s1 * np.cos(a1) / r1 - np.sin(PITCH) * np.cos(PITCH) / R_COIL
    R40 = (C * np.cos(a1) * dt - B * s1 * dk) / r1
    K40 = C * s1 * dt + B * np.cos(a1) * dk
    _, R, K, *_ = love_constitutive(eps, vth, R_COIL, PITCH, B, C)
    assert np.isclose(R, R40, rtol=1e-12)
    assert np.isclose(K, K40, rtol=1e-12)


def test_add_love_springs_replaces_stretch_and_twist():
    robot = _y_network()
    love = robot.add_love_springs(coil_r=R_COIL, pitch=PITCH, EI=B, GJ=C)
    assert love.love_springs.N == robot.bend_springs.N
    assert np.all(love.stretch_springs.EA == 0)
    assert np.all(love.twist_springs.GJ == 0)
    # original robot untouched
    assert np.all(robot.stretch_springs.EA > 0) and np.all(robot.twist_springs.GJ > 0)
    # every edge's pure stretch is carried exactly once
    ls = love.love_springs
    carried = np.zeros(robot.edges.shape[0])
    np.add.at(carried, ls.edges_ind.ravel(), ls.stretch_w.ravel())
    assert np.allclose(carried, robot.ref_len)


def test_strain_space_derivatives(deformed_y_network):
    _, energy, base = deformed_y_network
    s0 = energy._get_del_strain(base.state)
    E, g, H = energy._energy_derivatives(s0)
    h = 1e-7
    for c in range(3):
        ds = np.zeros_like(s0)
        ds[:, c] = h
        Ep, gp, _ = energy._energy_derivatives(s0 + ds)
        Em, gm, _ = energy._energy_derivatives(s0 - ds)
        assert np.allclose((Ep - Em) / (2 * h), g[:, c], rtol=1e-5, atol=1e-12)
        assert np.allclose((gp - gm) / (2 * h), H[:, :, c], rtol=1e-5, atol=1e-10)


def test_gradient_fd(deformed_y_network):
    _, energy, base = deformed_y_network
    q0 = base.state.q
    F, _ = energy.grad_hess_energy_linear_elastic(_state_at(base, q0))
    nn = 3 * base.n_nodes
    g_fd = np.zeros_like(q0)
    for i in range(q0.size):
        h = 1e-10 if i < nn else 1e-7
        dq = np.zeros_like(q0)
        dq[i] = h
        g_fd[i] = (energy.get_energy_linear_elastic(_state_at(base, q0 + dq)) -
                   energy.get_energy_linear_elastic(_state_at(base, q0 - dq))) / (2 * h)
    assert np.abs(-F - g_fd).max() < 1e-6 * np.abs(g_fd).max()


def test_hessian_fd_theta_blocks(deformed_y_network):
    """Columns for twist DOFs are exact (the node-node block inherits DER's
    approximate reference-twist Hessian)."""
    _, energy, base = deformed_y_network
    q0 = base.state.q
    _, J = energy.grad_hess_energy_linear_elastic(_state_at(base, q0))
    assert np.abs(J - J.T).max() < 1e-12 * np.abs(J).max()
    nn = 3 * base.n_nodes
    for i in range(nn, q0.size):
        h = 1e-7
        dq = np.zeros_like(q0)
        dq[i] = h
        Fp, _ = energy.grad_hess_energy_linear_elastic(_state_at(base, q0 + dq))
        Fm, _ = energy.grad_hess_energy_linear_elastic(_state_at(base, q0 - dq))
        col = (Fp - Fm) / (2 * h)
        assert np.abs(col - J[:, i]).max() < 1e-6 * np.abs(col).max()


def test_uniform_straight_rod_energy():
    """A straight rod uniformly stretched and twisted stores exactly
    L w(eps, 0) + l_triplets [w(eps, vtheta) - w(eps, 0)]."""
    n, ell = 11, 1e-3
    nodes = np.column_stack([np.zeros(n), np.zeros(n), ell * np.arange(n)])
    edges = np.column_stack([np.arange(n - 1), np.arange(1, n)])
    robot = _robot(nodes, edges).add_love_springs(coil_r=R_COIL, pitch=PITCH, EI=B, GJ=C)
    stepper = dismech.ImplicitEulerTimeStepper(robot)
    energy = stepper.elastic_energies[dismech.LOVE]

    eps, vth = 0.4, 50.0
    q = robot.q0.copy()
    q[2:3 * n:3] *= 1 + eps
    q[3 * n:] = vth * ell * np.arange(n - 1)      # twist per rest length vth
    state = stepper._finalize_update(robot, q).state

    w = lambda e, v: love_constitutive(e, v, R_COIL, PITCH, B, C)[0]
    expected = (n - 1) * ell * w(eps, 0.0) + (n - 2) * ell * (w(eps, vth) - w(eps, 0.0))
    assert np.isclose(energy.get_energy_linear_elastic(state), expected, rtol=1e-10)
