import pytest
import copy
import numpy as np
import scipy

import pathlib

from dismech.state import RobotState
from dismech.elastics import StretchEnergy


def rel_path(fname: str) -> pathlib.Path:
    """
    Localizes path to module path
    """
    return pathlib.Path(__file__).parent / fname

def stretch_energy_helper(robot, truth):
    energy = StretchEnergy(robot.stretch_springs, robot.state)
    new_state = RobotState.init(truth['q'].flatten(), np.ndarray(
        []), np.ndarray([]), np.ndarray([]), np.ndarray([]), np.ndarray([]), np.ndarray([]))
    Fs, Js = energy.grad_hess_energy_linear_elastic(new_state)
    assert (np.allclose(Fs, truth['Fs'].flatten()))
    assert (np.allclose(Js, truth['Js']))

def test_stretch_energy_cantilever_n51(softrobot_rod_cantilever_n51):
    robot = softrobot_rod_cantilever_n51
    valid_data = scipy.io.loadmat(
        rel_path('../resources/rod_cantilever/rod_cantilever_n51_get_fs_js.mat'))
    stretch_energy_helper(robot, valid_data)

def test_stretch_energy_hexparachute_n6(softrobot_hexparachute_n6):
    robot = softrobot_hexparachute_n6
    valid_data = scipy.io.loadmat(
        rel_path('../resources/parachute/hexparachute_n6_get_fs_js.mat'))
    stretch_energy_helper(robot, valid_data)

def test_stretch_energy_shell_cantilever_n40(softrobot_shell_cantilever_n40):
    robot = softrobot_shell_cantilever_n40
    valid_data = scipy.io.loadmat(
        rel_path('../resources/shell_cantilever/shell_cantilever_n40_fs_js_ft_jt_fb_shell_jb_shell.mat'))
    stretch_energy_helper(robot, valid_data)