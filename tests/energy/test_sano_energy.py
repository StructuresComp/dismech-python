import numpy as np
import pytest

import dismech
from dismech.state import RobotState
from dismech.elastics import sano_coupling


ZETA = 1e-2


def _sim():
    return dismech.SimParams(static_sim=True, two_d_sim=False, use_mid_edge=False,
                             use_line_search=False, log_data=False, log_step=1,
                             show_floor=False, dt=1.0, max_iter=50, total_time=1.0,
                             plot_step=1, tol=1e-9, ftol=1e-12, dtol=1e-13)


def _robot(nodes, ixs=(4e-12, 1e-12)):
    """Rod through nodes with a ribbon-like cross section (EI1 > EI2)."""
    nodes = np.asarray(nodes, float)
    edges = np.column_stack([np.arange(len(nodes) - 1), np.arange(1, len(nodes))])
    geo = dismech.Geometry(nodes, edges, np.empty(0))
    geom = dismech.GeomParams(rod_r0=1e-3, shell_h=0, ixs1=ixs[0], ixs2=ixs[1], jxs=2e-12)
    mat = dismech.Material(density=1000, youngs_rod=1e7, youngs_shell=0,
                           poisson_rod=0.3, poisson_shell=0)
    return dismech.SoftRobot(geom, mat, geo, _sim(), dismech.Environment())


def _curved_rod(n=8, L=0.04, seed=0):
    rng = np.random.default_rng(seed)
    s = np.linspace(0, 1, n)
    nodes = np.column_stack([L * s, 0.2 * L * np.sin(3 * s), 0.1 * L * s ** 2])
    return _robot(nodes + 1e-4 * rng.standard_normal(nodes.shape))


def _state_at(base, q):
    """State at q with reference frames transported from base (as in a Newton step)."""
    a1, a2 = base.compute_time_parallel(base.state.a1, base.state.q, q)
    m1, m2 = base.compute_material_directors(q, a1, a2)
    rt = base.compute_reference_twist(base.twist_springs, q, a1, base.state.ref_twist)
    return RobotState.init(q, a1, a2, m1, m2, rt, np.empty(0))


def _deform(robot, stepper, seed=3):
    rng = np.random.default_rng(seed)
    nn = 3 * robot.n_nodes
    q0 = robot.q0.copy()
    q0[:nn] += 1e-3 * rng.standard_normal(nn)
    q0[nn:] += 0.3 * rng.standard_normal(q0.size - nn)
    return stepper._finalize_update(robot, q0)


@pytest.fixture
def deformed_rod():
    robot = _curved_rod().add_sano_springs(zeta=ZETA)
    stepper = dismech.ImplicitEulerTimeStepper(robot)
    base = _deform(robot, stepper)
    energy = stepper.elastic_energies[dismech.SANO]
    # the coupling must be active for the checks to mean anything
    s = energy._get_del_strain(base.state)
    assert np.abs(s[:, 2]).max() > 0.1
    return robot, energy, base


def test_coupling_derivatives():
    rng = np.random.default_rng(0)
    t, k, c = rng.uniform(-1, 1, 30), rng.uniform(-1, 1, 30), rng.uniform(0.01, 1, 30)
    f, f_t, f_k, f_tt, f_tk, f_kk = sano_coupling(t, k, c)
    h = 1e-6
    ft = lambda d: np.stack(sano_coupling(t + d, k, c)[:3])
    fk = lambda d: np.stack(sano_coupling(t, k + d, c)[:3])
    dt, dk = (ft(h) - ft(-h)) / (2 * h), (fk(h) - fk(-h)) / (2 * h)
    assert np.allclose(dt[0], f_t, rtol=1e-6)
    assert np.allclose(dk[0], f_k, rtol=1e-6, atol=1e-10)
    assert np.allclose(dt[1], f_tt, rtol=1e-5)
    assert np.allclose(dk[1], f_tk, rtol=1e-5, atol=1e-8)
    assert np.allclose(dt[2], f_tk, rtol=1e-5, atol=1e-8)
    assert np.allclose(dk[2], f_kk, rtol=1e-5, atol=1e-8)


def test_coupling_kirchhoff_limit_is_finite():
    out = sano_coupling(np.array([0.0, 0.5]), np.array([0.0, 0.2]), np.inf)
    assert all(np.all(v == 0) for v in out)


def test_add_sano_springs_replaces_bend_and_twist():
    robot = _curved_rod()
    sano = robot.add_sano_springs(zeta=ZETA)
    assert sano.sano_springs.N == robot.bend_springs.N
    assert np.all(sano.bend_springs.EI == 0)
    assert np.all(sano.twist_springs.GJ == 0)
    assert np.allclose(sano.sano_springs.EI, robot.bend_springs.EI)
    assert np.allclose(sano.sano_springs.GJ, robot.twist_springs.GJ)
    assert np.all(sano.stretch_springs.EA == robot.stretch_springs.EA)
    # original robot untouched
    assert np.all(robot.bend_springs.EI > 0) and np.all(robot.twist_springs.GJ > 0)


def test_add_sano_springs_warns_on_hard_easy_axis():
    with pytest.warns(UserWarning, match="easy axis"):
        _curved_rod().add_sano_springs(zeta=ZETA, EI=np.array([1e-5, 4e-5]))


def test_strain_space_derivatives(deformed_rod):
    _, energy, base = deformed_rod
    s0 = energy._get_del_strain(base.state)
    E, g, H = energy._energy_derivatives(s0)
    h = 1e-7
    for c in range(3):
        ds = np.zeros_like(s0)
        ds[:, c] = h
        Ep, gp, _ = energy._energy_derivatives(s0 + ds)
        Em, gm, _ = energy._energy_derivatives(s0 - ds)
        assert np.allclose((Ep - Em) / (2 * h), g[:, c], rtol=1e-5, atol=1e-14)
        assert np.allclose((gp - gm) / (2 * h), H[:, :, c], rtol=1e-5, atol=1e-12)


def test_gradient_fd(deformed_rod):
    _, energy, base = deformed_rod
    q0 = base.state.q
    F, _ = energy.grad_hess_energy_linear_elastic(_state_at(base, q0))
    nn = 3 * base.n_nodes
    g_fd = np.zeros_like(q0)
    for i in range(q0.size):
        h = 1e-9 if i < nn else 1e-7
        dq = np.zeros_like(q0)
        dq[i] = h
        g_fd[i] = (energy.get_energy_linear_elastic(_state_at(base, q0 + dq)) -
                   energy.get_energy_linear_elastic(_state_at(base, q0 - dq))) / (2 * h)
    assert np.abs(-F - g_fd).max() < 1e-6 * np.abs(g_fd).max()


def test_hessian_fd_theta_blocks(deformed_rod):
    """Columns for twist DOFs are exact (the node-node block inherits DER's
    approximate, and for bending unsymmetric, reference-frame Hessian)."""
    _, energy, base = deformed_rod
    q0 = base.state.q
    _, J = energy.grad_hess_energy_linear_elastic(_state_at(base, q0))
    nn = 3 * base.n_nodes
    assert np.abs(J - J.T)[nn:].max() < 1e-12 * np.abs(J).max()
    for i in range(nn, q0.size):
        h = 1e-7
        dq = np.zeros_like(q0)
        dq[i] = h
        Fp, _ = energy.grad_hess_energy_linear_elastic(_state_at(base, q0 + dq))
        Fm, _ = energy.grad_hess_energy_linear_elastic(_state_at(base, q0 - dq))
        col = (Fp - Fm) / (2 * h)
        assert np.abs(col - J[:, i]).max() < 1e-6 * np.abs(col).max()


def test_kirchhoff_limit_matches_bend_and_twist():
    """zeta = 0 removes the coupling: Sano springs reproduce DER bend + twist."""
    robot = _curved_rod()
    sano_robot = robot.add_sano_springs(zeta=0.0)
    der = dismech.ImplicitEulerTimeStepper(robot)
    sano = dismech.ImplicitEulerTimeStepper(sano_robot)
    state = _deform(robot, der).state

    E_der = [der.elastic_energies[k] for k in (dismech.BEND, dismech.TWIST)]
    E_sano = sano.elastic_energies[dismech.SANO]
    F_ref = sum(e.grad_hess_energy_linear_elastic(state)[0] for e in E_der)
    J_ref = sum(e.grad_hess_energy_linear_elastic(state)[1] for e in E_der)
    F, J = E_sano.grad_hess_energy_linear_elastic(state)
    assert np.isclose(E_sano.get_energy_linear_elastic(state),
                      sum(e.get_energy_linear_elastic(state) for e in E_der), rtol=1e-12)
    assert np.abs(F - F_ref).max() < 1e-12 * np.abs(F_ref).max()
    assert np.abs(J - J_ref).max() < 1e-12 * np.abs(J_ref).max()


def test_uniform_twist_straight_ribbon():
    """A straight ribbon (kappa = 0) uniformly twisted stores per triplet
    GJ/2l t^2 + EI2/2l t^4 (zeta/l)^2 -- bounded where Sadowsky diverges."""
    n, ell, rate = 11, 1e-3, 50.0
    nodes = np.column_stack([np.zeros(n), np.zeros(n), ell * np.arange(n)])
    robot = _robot(nodes).add_sano_springs(zeta=ZETA)
    stepper = dismech.ImplicitEulerTimeStepper(robot)
    energy = stepper.elastic_energies[dismech.SANO]

    q = robot.q0.copy()
    q[3 * n:] = rate * ell * np.arange(n - 1)
    state = stepper._finalize_update(robot, q).state

    t = rate * ell
    EI2, GJ = robot.sano_springs.EI[0, 1], robot.sano_springs.GJ[0]
    expected = (n - 2) * (0.5 * GJ / ell * t ** 2 + 0.5 * EI2 / ell * t ** 4 * (ZETA / ell) ** 2)
    assert np.isclose(energy.get_energy_linear_elastic(state), expected, rtol=1e-10)


def test_rest_state_is_force_free():
    """Natural strains default to the (curved) initial state."""
    robot = _curved_rod().add_sano_springs(zeta=ZETA)
    stepper = dismech.ImplicitEulerTimeStepper(robot)
    F, _ = stepper.elastic_energies[dismech.SANO].grad_hess_energy_linear_elastic(robot.state)
    assert np.abs(F).max() < 1e-14
