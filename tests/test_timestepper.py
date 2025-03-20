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
    stepper.robot.env.set_static()
    robots = stepper.simulate()
    qs = np.stack([robot.state.q for robot in robots])

    valid_data = scipy.io.loadmat(
        rel_path('resources/rod_cantilever/rod_cantilever_n51_static_sim.mat'))
    assert (np.allclose(qs[:, stepper.robot.end_node_dof_index],
            valid_data['qs'][:, stepper.robot.end_node_dof_index]))


def test_dynamic_sim_shell_cantilever_n40(time_stepper_shell_cantilever_n40):
    stepper = time_stepper_shell_cantilever_n40
    stepper.robot.sim_params.total_time = 0.4
    robots = stepper.simulate()
    qs = np.stack([robot.state.q for robot in robots])

    valid_data = scipy.io.loadmat(
        rel_path('resources/shell_cantilever/shell_cantilever_n40_dynamic_sim.mat'))
    np.allclose(qs, valid_data['qs'][:40])


def test_dynamic_sim_contortion_n21(time_stepper_contortion_n21):
    stepper = time_stepper_contortion_n21
    stepper.robot.sim_params.total_time = 1.0
    robots = stepper.simulate()
    qs = np.stack([robot.state.q for robot in robots])

    valid_data = scipy.io.loadmat(
        rel_path('resources/contortion/contortion_n21_dynamic_sim.mat'))
    np.allclose(qs, valid_data['qs'])
