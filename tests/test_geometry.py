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
    assert (custom_array_equal(valid_data['rod_nodes'], geo.rod_nodes))
    assert (custom_array_equal(valid_data['shell_nodes'], geo.shell_nodes))
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


@pytest.fixture
def rod_cantilever_geom():
    b = 0.02
    h = 0.001

    return dismech.GeomParams(rod_r0=0.001,
                              shell_h=0,
                              axs=b*h,
                              ixs1=b*h**3/12,
                              ixs2=h*b**3/12,
                              jxs=b*h**3/6)


@pytest.fixture
def rod_cantilever_n26():
    return dismech.Geometry.from_txt(rel_path('resources/rod_cantilever/horizontal_rod_n26.txt'))


def test_rod_cantilever_n26_from_txt(rod_cantilever_n26):
    valid_data = scipy.io.loadmat(
        rel_path('resources/rod_cantilever/rod_cantilever_n26_create_geometry.mat'))
    validate_create_geometry(rod_cantilever_n26, valid_data)


@pytest.fixture
def rod_cantilever_n51():
    return dismech.Geometry.from_txt(rel_path('resources/rod_cantilever/horizontal_rod_n51.txt'))


def test_rod_cantilever_n51_from_txt(rod_cantilever_n51):
    valid_data = scipy.io.loadmat(
        rel_path('resources/rod_cantilever/rod_cantilever_n51_create_geometry.mat'))
    validate_create_geometry(rod_cantilever_n51, valid_data)


@pytest.fixture
def rod_cantilever_n101():
    return dismech.Geometry.from_txt(rel_path('resources/rod_cantilever/horizontal_rod_n101.txt'))


def test_rod_cantilever_n101_from_txt(rod_cantilever_n101):
    valid_data = scipy.io.loadmat(
        rel_path('resources/rod_cantilever/rod_cantilever_n101_create_geometry.mat'))
    validate_create_geometry(rod_cantilever_n101, valid_data)


# parachute

@pytest.fixture
def hexparachute_n6():
    return dismech.Geometry.from_txt(rel_path('resources/parachute/hexparachute_n6_python.txt'))


def test_hexparachute_n6(hexparachute_n6):
    valid_data = scipy.io.loadmat(
        rel_path('resources/parachute/hexparachute_create_geometry.mat'))
    validate_create_geometry(hexparachute_n6, valid_data)

# pneunet


@pytest.fixture
def pneunet_shorter():
    return dismech.Geometry.from_txt(rel_path('resources/pneunet/input_straight_horizontal_shorter.txt'))


def test_pneunet_shorter(pneunet_shorter):
    valid_data = scipy.io.loadmat(
        rel_path('resources/pneunet/input_straight_horizontal_shorter_create_geometry.mat'))
    validate_create_geometry(pneunet_shorter, valid_data)
