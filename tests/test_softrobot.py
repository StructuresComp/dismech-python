import pytest
import numpy as np
import scipy

import pathlib


def rel_path(fname: str) -> pathlib.Path:
    """
    Localizes path to module path
    """
    return pathlib.Path(__file__).parent / fname


def tangent_helper(softrobot, q, tangent_truth):
    tangent = softrobot.compute_tangent(q)
    assert (np.allclose(tangent, tangent_truth))


def test_softrobot_cantilever_n51(softrobot_cantilever_n51):
    robot = softrobot_cantilever_n51
    valid_data = scipy.io.loadmat(
        rel_path('resources/rod_cantilever/rod_cantilever_n51_multirod.mat'))

    tangent_helper(robot,
                   robot.q, valid_data['q0_tangent'])
    tangent_helper(robot,
                   valid_data['test_q'], valid_data['test_tangent'])

def test_softrobot_hexparachute_n6(softrobot_hexparachute_n6):
    robot = softrobot_hexparachute_n6
    valid_data = scipy.io.loadmat(
        rel_path('resources/parachute/hexparachute_multirod.mat'))
    tangent_helper(robot, robot.q, valid_data['q0_tangent'])