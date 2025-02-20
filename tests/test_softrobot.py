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
    tangent = robot.compute_tangent(q)
    assert (np.allclose(tangent, tangent_truth))


def robot_helper(robot, truth):
    assert (np.allclose(robot.q, truth['q'][0][0].reshape(-1)))
    assert (np.allclose(robot.q0, truth['q0'][0][0].reshape(-1)))
    assert (np.allclose(robot.a1, truth['a1'][0][0]))
    assert (np.allclose(robot.a2, truth['a2'][0][0]))


def time_parallel_helper(robot, truth):
    a1, a2 = robot.compute_time_parallel(truth['a1'], truth['q0'], truth['q'])
    assert (np.allclose(a1, truth['a1_iter']))
    assert (np.allclose(a2, truth['a2_iter']))


def reference_twist_helper(robot, truth):
    new_twist = robot.compute_reference_twist(robot.bend_twist_spring, truth['a1'], truth['tangent'], truth['orgTwist'])
    assert(np.allclose(new_twist, truth['refTwist']))

def material_director_helper(robot, truth): 
    m1, m2 = robot.compute_material_directors(truth['a1'], truth['a2'], truth['theta'])
    assert(np.allclose(m1, truth['m1']))
    assert(np.allclose(m2, truth['m2']))

def test_tangent_cantilever_n51(softrobot_cantilever_n51):
    robot = softrobot_cantilever_n51
    valid_data = scipy.io.loadmat(
        rel_path('resources/rod_cantilever/rod_cantilever_n51_multirod.mat'))

    tangent_helper(robot,
                   robot.q, valid_data['q0_tangent'])
    tangent_helper(robot,
                   valid_data['test_q'], valid_data['test_tangent'])


def test_tangent_hexparachute_n6(softrobot_hexparachute_n6):
    robot = softrobot_hexparachute_n6
    valid_data = scipy.io.loadmat(
        rel_path('resources/parachute/hexparachute_multirod.mat'))
    tangent_helper(robot, robot.q, valid_data['q0_tangent'])


def test_compute_space_parallel_cantilever_n51(softrobot_cantilever_n51):
    robot = softrobot_cantilever_n51
    new_robot = robot.compute_space_parallel()
    valid_data = scipy.io.loadmat(rel_path(
        'resources/rod_cantilever/rod_cantilever_n51_compute_space_parallel.mat'))
    robot_helper(new_robot, valid_data['robot'])


def test_compute_time_parallel_cantilever_n51(softrobot_cantilever_n51):
    robot = softrobot_cantilever_n51
    valid_data = scipy.io.loadmat(rel_path(
        'resources/rod_cantilever/rod_cantilever_n51_compute_time_parallel.mat'))
    time_parallel_helper(robot, valid_data)


def test_compute_reference_twist_cantilever_n51(softrobot_cantilever_n51):
    robot = softrobot_cantilever_n51
    valid_data = scipy.io.loadmat(rel_path(
        'resources/rod_cantilever/rod_cantilever_n51_compute_reference_twist.mat'))
    reference_twist_helper(robot, valid_data)
