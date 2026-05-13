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


def test_triangle_energy_default_preserves_legacy_force_and_jacobian(
    softrobot_hexparachute_n6_mid_edge,
):
    """zero_reference=False (default) must match the un-calibrated formulation."""
    robot = softrobot_hexparachute_n6_mid_edge
    valid_data = scipy.io.loadmat(
        rel_path('../resources/parachute/hexparachute_n6_get_fb_jb_midedge_shell.mat'))

    energy_default = TriangleEnergy(robot.triangle_springs, robot.state)
    energy_explicit = TriangleEnergy(
        robot.triangle_springs, robot.state, zero_reference=False
    )

    new_state = RobotState.init(
        valid_data['q'].flatten(),
        np.ndarray([]), np.ndarray([]), np.ndarray([]), np.ndarray([]),
        np.ndarray([]), valid_data['tau_0'],
    )

    Fb_d, Jb_d = energy_default.grad_hess_energy_linear_elastic(new_state)
    Fb_e, Jb_e = energy_explicit.grad_hess_energy_linear_elastic(new_state)

    assert np.allclose(Fb_d, Fb_e)
    assert np.allclose(Jb_d, Jb_e)
    assert np.allclose(Fb_d, valid_data['Fb_shell'].flatten())
    # zero_reference=False stores scalar 0.0 sentinels
    assert energy_default.reference_force == 0.0
    assert energy_default.reference_energy == 0.0


def test_triangle_energy_zero_reference_vanishes_at_initial_state(
    softrobot_hexparachute_n6_mid_edge,
):
    """With zero_reference=True the bending energy and force at the initial state are zero."""
    robot = softrobot_hexparachute_n6_mid_edge
    energy = TriangleEnergy(
        robot.triangle_springs, robot.state, zero_reference=True
    )

    Fb, _ = energy.grad_hess_energy_linear_elastic(robot.state)
    e_per_spring = energy.get_energy_linear_elastic(robot.state)

    assert np.allclose(Fb, 0.0, atol=1e-10)
    assert np.allclose(e_per_spring, 0.0, atol=1e-10)


def test_triangle_energy_zero_reference_shifts_force_by_constant(
    softrobot_hexparachute_n6_mid_edge,
):
    """At an arbitrary state, zero-referenced output differs from the raw output by exactly
    the reference force / per-spring reference energy. The Hessian is identical because
    the subtracted term is a constant."""
    robot = softrobot_hexparachute_n6_mid_edge
    valid_data = scipy.io.loadmat(
        rel_path('../resources/parachute/hexparachute_n6_get_fb_jb_midedge_shell.mat'))

    raw = TriangleEnergy(robot.triangle_springs, robot.state)
    zeroed = TriangleEnergy(
        robot.triangle_springs, robot.state, zero_reference=True
    )

    new_state = RobotState.init(
        valid_data['q'].flatten(),
        np.ndarray([]), np.ndarray([]), np.ndarray([]), np.ndarray([]),
        np.ndarray([]), valid_data['tau_0'],
    )

    Fb_raw, Jb_raw = raw.grad_hess_energy_linear_elastic(new_state)
    Fb_zero, Jb_zero = zeroed.grad_hess_energy_linear_elastic(new_state)
    e_raw = raw.get_energy_linear_elastic(new_state)
    e_zero = zeroed.get_energy_linear_elastic(new_state)

    assert np.allclose(Fb_zero, Fb_raw - zeroed.reference_force)
    assert np.allclose(Jb_zero, Jb_raw)
    assert np.allclose(e_zero, e_raw - zeroed.reference_energy)
