import pathlib
import numpy as np
import scipy

from dismech.external_forces import compute_aerodynamic_forces, compute_aerodynamic_forces_vectorized


def rel_path(fname: str) -> pathlib.Path:
    """
    Localizes path to module path
    """
    return pathlib.Path(__file__).parent / fname


def test_aerodynamic_hexparachute_n6(time_stepper_hexparachute_n6):
    stepper = time_stepper_hexparachute_n6
    robot = stepper.robot
    valid_data = scipy.io.loadmat(
        rel_path('resources/parachute/hexparachute_n6_get_aerodynamic.mat'))
    Fd, Jd = compute_aerodynamic_forces(
        robot, valid_data['q'].flatten(), valid_data['q0'].flatten())
    assert (np.allclose(Fd, valid_data['Fd'].flatten()))
    assert (np.allclose(Jd, valid_data['Jd']))


def test_aerodynamic_vectorized_hexparachute_n6(time_stepper_hexparachute_n6):
    stepper = time_stepper_hexparachute_n6
    robot = stepper.robot
    valid_data = scipy.io.loadmat(
        rel_path('resources/parachute/hexparachute_n6_get_aerodynamic.mat'))
    Fd, Jd = compute_aerodynamic_forces_vectorized(
        robot, valid_data['q'].flatten(), valid_data['q0'].flatten())
    assert (np.allclose(Fd, valid_data['Fd'].flatten()))
    assert (np.allclose(Jd, valid_data['Jd']))
