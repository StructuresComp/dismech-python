import scipy
import numpy as np
import pytest
import pathlib


def rel_path(fname: str) -> pathlib.Path:
    """
    Localizes path to module path
    """
    return pathlib.Path(__file__).parent / fname


def test_static_sim_cantilever_n51(time_stepper_cantilever_n51):
    stepper = time_stepper_cantilever_n51
    robot = stepper.robot
    robot.env.set_static()

    qs = []
    steps = int(robot.sim_params.total_time / robot.sim_params.dt) + 1
    for i in range(1, steps):
        robot.env.g = robot.env.static_g * (i) / steps
        new_robot = stepper.step()
        qs.append(new_robot.q)
    qs = np.stack(qs)

    valid_data = scipy.io.loadmat(
        rel_path('resources/rod_cantilever/rod_cantilever_n51_static_sim.mat'))
    # Numerical stability
    assert (np.allclose(qs, valid_data['qs'], rtol=1e-1))
