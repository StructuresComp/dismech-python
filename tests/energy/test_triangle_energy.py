import pytest
import copy
import numpy as np
import scipy

import pathlib

from dismech.state import RobotState
from dismech.elastics import TriangleEnergy


def rel_path(fname: str) -> pathlib.Path:
    """
    Localizes path to module path
    """
    return pathlib.Path(__file__).parent / fname

def triangle_energy_helper(robot, truth):
    energy = TriangleEnergy(robot.triangle_springs, robot.state)
    new_state = RobotState.init(truth['q'].flatten(),
                                np.ndarray([]),
                                np.ndarray([]),
                                np.ndarray([]),
                                np.ndarray([]),
                                np.ndarray([]),
                                truth['tau_0'])
    Fb, Jb = energy.grad_hess_energy_linear_elastic(new_state)
    e = energy.get_energy_linear_elastic(new_state)
    assert (np.allclose(Fb, truth['Fb_shell'].flatten()))
    assert (np.allclose(Jb, truth['Jb_shell']))


def test_triangle_energy_hexparachute_n6(softrobot_hexparachute_n6_mid_edge):
    robot = softrobot_hexparachute_n6_mid_edge
    valid_data = scipy.io.loadmat(
        rel_path('../resources/parachute/hexparachute_n6_get_fb_jb_midedge_shell.mat'))
    triangle_energy_helper(robot, valid_data)


def test_triangle_energy_shell_cantilever_n40(softrobot_shell_cantilever_n40_mid_edge):
    robot = softrobot_shell_cantilever_n40_mid_edge
    valid_data = scipy.io.loadmat(
        rel_path('../resources/shell_cantilever/shell_cantilever_n40_get_fb_jb_midedge_shell.mat'))
    triangle_energy_helper(robot, valid_data)
