import pytest
import copy
import numpy as np
import scipy

import pathlib

from dismech.state import RobotState
from dismech.elastics import BendEnergy


def rel_path(fname: str) -> pathlib.Path:
    """
    Localizes path to module path
    """
    return pathlib.Path(__file__).parent / fname


def bend_energy_helper(robot, truth):
    energy = BendEnergy(robot.bend_twist_springs, robot.state)
    new_state = RobotState.init(truth['q'].flatten(), np.ndarray(
        []), np.ndarray([]), truth['m1'], truth['m2'], np.ndarray([]), np.ndarray([]))
    Fb, Jb = energy.grad_hess_energy_linear_elastic(new_state)

    assert (np.allclose(Fb, truth['Fb'].flatten()))
    assert (np.allclose(Jb, truth['Jb'], rtol=1e-2))


def test_bend_energy_cantilever_n51(softrobot_rod_cantilever_n51):
    robot = softrobot_rod_cantilever_n51

    valid_data = scipy.io.loadmat(
        rel_path('../resources/rod_cantilever/rod_cantilever_n51_get_fb_jb_q.mat'))
    bend_energy_helper(robot, valid_data)

    valid_data = scipy.io.loadmat(
        rel_path('../resources/rod_cantilever/rod_cantilever_n51_get_fb_jb_q0.mat'))
    bend_energy_helper(robot, valid_data)
