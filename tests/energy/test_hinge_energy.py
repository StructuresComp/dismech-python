import pytest
import copy
import numpy as np
import scipy

import pathlib

from dismech.elastics import HingeEnergy


def rel_path(fname: str) -> pathlib.Path:
    """
    Localizes path to module path
    """
    return pathlib.Path(__file__).parent / fname

def hinge_energy_helper(robot, truth):
    energy = HingeEnergy(robot.hinge_springs)
    Fb, Jb = energy.grad_hess_energy_linear_elastic(truth['q'].flatten())
    assert (np.allclose(Fb, truth['Fb_shell'].flatten()))
    assert (np.allclose(Jb, truth['Jb_shell']))

def test_hinge_energy_hexparachute_n6(softrobot_hexparachute_n6):
    robot = softrobot_hexparachute_n6
    valid_data = scipy.io.loadmat(
        rel_path('../resources/parachute/hexparachute_n6_get_fb_jb_shell.mat'))
    hinge_energy_helper(robot, valid_data)

def test_hinge_energy_shell_cantilever(softrobot_shell_cantilever_n40):
    robot = softrobot_shell_cantilever_n40
    valid_data = scipy.io.loadmat(
        rel_path('../resources/shell_cantilever/shell_cantilever_n40_get_fb_jb_shell.mat'))
    hinge_energy_helper(robot, valid_data)
