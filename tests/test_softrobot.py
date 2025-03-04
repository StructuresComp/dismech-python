import pytest
import copy
import numpy as np
import scipy

import pathlib


def rel_path(fname: str) -> pathlib.Path:
    """
    Localizes path to module path
    """
    return pathlib.Path(__file__).parent / fname


def tangent_helper(robot, q, tangent_truth):
    tangent = robot._compute_tangent(q)
    assert (np.allclose(tangent, tangent_truth))


def robot_helper(robot, truth):
    assert (np.allclose(robot.q0, truth['q0'][0][0].reshape(-1)))
    assert (np.allclose(robot.state.q, truth['q'][0][0].reshape(-1)))
    assert (np.allclose(robot.state.a1, truth['a1'][0][0]))
    assert (np.allclose(robot.state.a2, truth['a2'][0][0]))


def time_parallel_helper(robot, truth):
    a1, a2 = robot.compute_time_parallel(truth['a1'], truth['q0'], truth['q'])
    assert (np.allclose(a1, truth['a1_iter']))
    assert (np.allclose(a2, truth['a2_iter']))


def test_tangent_cantilever_n51(softrobot_rod_cantilever_n51):
    robot = softrobot_rod_cantilever_n51
    valid_data = scipy.io.loadmat(
        rel_path('resources/rod_cantilever/rod_cantilever_n51_multirod.mat'))

    tangent_helper(robot,
                   robot.state.q, valid_data['q0_tangent'])
    tangent_helper(robot,
                   valid_data['test_q'], valid_data['test_tangent'])


def test_tangent_hexparachute_n6(softrobot_hexparachute_n6):
    robot = softrobot_hexparachute_n6
    valid_data = scipy.io.loadmat(
        rel_path('resources/parachute/hexparachute_multirod.mat'))
    tangent_helper(robot, robot.q0, valid_data['q0_tangent'])


def test_compute_space_parallel_cantilever_n51(softrobot_rod_cantilever_n51):
    robot = softrobot_rod_cantilever_n51
    robot.compute_space_parallel()
    valid_data = scipy.io.loadmat(rel_path(
        'resources/rod_cantilever/rod_cantilever_n51_compute_space_parallel.mat'))
    robot_helper(robot, valid_data['robot'])


def test_compute_time_parallel_cantilever_n51(softrobot_rod_cantilever_n51):
    robot = softrobot_rod_cantilever_n51
    valid_data = scipy.io.loadmat(rel_path(
        'resources/rod_cantilever/rod_cantilever_n51_compute_time_parallel.mat'))
    time_parallel_helper(robot, valid_data)
