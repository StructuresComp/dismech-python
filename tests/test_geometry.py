import typing
import pathlib

import pytest
import scipy.io
import numpy as np

import dismech


def rel_path(fname: str) -> pathlib.Path:
    """
    Localizes path to module path
    """
    return pathlib.Path(__file__).parent / fname


def custom_array_equal(arr1: np.ndarray, arr2: np.ndarray):
    """
    When comparing empty arrays, the dtype and "shape" matter and
    scipy.io.loadmat loads empty arrays in its own matter. So, we
    only need to ensure that both are empty.
    """
    if arr1.size:
        return np.array_equal(arr1, arr2)
    return arr2.size == 0


def validate_create_geometry(geo: dismech.Geometry, valid_data: typing.Dict[str, np.ndarray]):
    """
    Checks if all public properties are equivalent
    """
    assert (custom_array_equal(valid_data['nodes'], geo.nodes))
    assert (custom_array_equal(valid_data['edges'] - 1, geo.edges))
    assert (custom_array_equal(valid_data['rod_edges'] - 1, geo.rod_edges))
    assert (custom_array_equal(valid_data['shell_edges'] - 1, geo.shell_edges))
    assert (custom_array_equal(
        valid_data['rod_shell_joint_edges'] - 1, geo.rod_shell_joint_edges))
    assert (custom_array_equal(
        valid_data['rod_shell_joint_total_edges'] - 1, geo.rod_shell_joint_edges_total))
    assert (custom_array_equal(valid_data['face_nodes'] - 1, geo.face_nodes))
    assert (custom_array_equal(valid_data['face_edges'] - 1, geo.face_edges))
    assert (custom_array_equal(
        valid_data['elStretchRod'] - 1, geo.rod_stretch_springs))
    assert (custom_array_equal(
        valid_data['elStretchShell'] - 1, geo.shell_stretch_springs))
    assert (custom_array_equal(
        valid_data['elBendRod'] - 1, geo.bend_twist_springs))
    assert (custom_array_equal(valid_data['elBendSign'], geo.bend_twist_signs))
    assert (custom_array_equal(valid_data['sign_faces'], geo.sign_faces))
    assert (custom_array_equal(
        valid_data['face_unit_norms'], geo.face_unit_norms))

# rod cantilever


def test_rod_cantilever_n26_from_txt(rod_cantilever_n26):
    valid_data = scipy.io.loadmat(
        rel_path('resources/rod_cantilever/rod_cantilever_n26_create_geometry.mat'))
    validate_create_geometry(rod_cantilever_n26, valid_data)


def test_rod_cantilever_n51_from_txt(rod_cantilever_n51):
    valid_data = scipy.io.loadmat(
        rel_path('resources/rod_cantilever/rod_cantilever_n51_create_geometry.mat'))
    validate_create_geometry(rod_cantilever_n51, valid_data)


def test_rod_cantilever_n101_from_txt(rod_cantilever_n101):
    valid_data = scipy.io.loadmat(
        rel_path('resources/rod_cantilever/rod_cantilever_n101_create_geometry.mat'))
    validate_create_geometry(rod_cantilever_n101, valid_data)


# parachute

def test_hexparachute_n6(hexparachute_n6):
    valid_data = scipy.io.loadmat(
        rel_path('resources/parachute/hexparachute_n6_create_geometry.mat'))
    validate_create_geometry(hexparachute_n6, valid_data)

# pneunet


def test_pneunet_shorter(pneunet_shorter):
    valid_data = scipy.io.loadmat(
        rel_path('resources/pneunet/input_straight_horizontal_shorter_create_geometry.mat'))
    validate_create_geometry(pneunet_shorter, valid_data)

# parachute


def test_square_plate_30(square_plate_30):
    valid_data = scipy.io.loadmat(
        rel_path('resources/square_plate/random_mesh_30_create_geometry.mat'))
    validate_create_geometry(square_plate_30, valid_data)
