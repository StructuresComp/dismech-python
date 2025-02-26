import scipy
import numpy as np
import pytest
import pathlib


def rel_path(fname: str) -> pathlib.Path:
    """
    Localizes path to module path
    """
    return pathlib.Path(__file__).parent / fname


def test_static_sim_shell_cantilever_n51(time_stepper_rod_cantilever_n51):
    stepper = time_stepper_rod_cantilever_n51
    robot = stepper.robot
    robot.env.set_static()
    valid_data = scipy.io.loadmat(
        rel_path('resources/rod_cantilever/rod_cantilever_n51_static_sim.mat'))

    qs = []
    steps = int(robot.sim_params.total_time / robot.sim_params.dt) + 1
    for i in range(1, steps):
        robot.env.g = robot.env.static_g * (i) / steps
        new_robot = stepper.step()
        qs.append(new_robot.q)
    qs = np.stack(qs)

    # Numerical stability
    assert (np.allclose(qs, valid_data['qs'], rtol=1e-1))


def test_dynamic_sim_shell_cantilever_n40(time_stepper_shell_cantilever_n40):
    stepper = time_stepper_shell_cantilever_n40
    robot = stepper.robot
    valid_data = scipy.io.loadmat(
        rel_path('resources/shell_cantilever/shell_cantilever_n40_dynamic_sim.mat'))

    robot.sim_params.total_time = 0.4   # way too long otherwise

    qs = []
    steps = int(robot.sim_params.total_time / robot.sim_params.dt) + 1
    for i in range(1, steps):
        new_robot = stepper.step()
        qs.append(new_robot.q)
    qs = np.stack(qs)

    np.allclose(qs, valid_data['qs'][:40])

def test_dynamic_sim_contortion_n21(time_stepper_contortion_n21):
    u0 = 0.1
    w0 = 2

    stepper = time_stepper_contortion_n21
    robot = stepper.robot   # update robot
    new_robot = robot
    valid_data = scipy.io.loadmat(
        rel_path('resources/contortion/contortion_n21_dynamic_sim.mat'))
    robot.sim_params.total_time = 1.0
    
    qs = []
    steps = int(robot.sim_params.total_time / robot.sim_params.dt) + 1
    for i in range(1, steps):
        new_robot = stepper.step(new_robot)
        qs.append(new_robot.q)

        # Move fixed node/edges
        if i * robot.sim_params.dt < 0.15:
            q = new_robot.q
            q[robot.map_node_to_dof(stepper.fixed_nodes[:3])[:,0]] += u0 * robot.sim_params.dt
            new_robot = new_robot.update(q)
        else:
            q = new_robot.q
            q[63:65] += w0 * robot.sim_params.dt
            new_robot = new_robot.update(q)
    qs = np.stack(qs)

    np.allclose(qs, valid_data['qs'])


def test_aerodynamic_hexparachute_n6(time_stepper_hexparachute_n6):
    stepper = time_stepper_hexparachute_n6
    robot = stepper.robot
    valid_data = scipy.io.loadmat(
        rel_path('resources/parachute/hexparachute_n6_get_aerodynamic.mat'))
    Fd, Jd = stepper._compute_aerodynamic_forces(
        robot, valid_data['q'].flatten(), valid_data['q0'].flatten())
    assert (np.allclose(Fd, valid_data['Fd'].flatten()))
    assert (np.allclose(Jd, valid_data['Jd']))

def test_aerodynamic_vectorized_hexparachute_n6(time_stepper_hexparachute_n6):
    stepper = time_stepper_hexparachute_n6
    robot = stepper.robot
    valid_data = scipy.io.loadmat(
        rel_path('resources/parachute/hexparachute_n6_get_aerodynamic.mat'))
    Fd, Jd = stepper._compute_aerodynamic_forces_vectorized(
        robot, valid_data['q'].flatten(), valid_data['q0'].flatten())
    assert (np.allclose(Fd, valid_data['Fd'].flatten()))
    assert (np.allclose(Jd, valid_data['Jd']))
