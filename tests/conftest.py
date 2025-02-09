import pathlib
import pytest
import numpy as np

import sys
import os

# Add the 'src' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import dismech

def rel_path(fname: str) -> pathlib.Path:
    """
    Localizes path to module path
    """
    return pathlib.Path(__file__).parent / fname

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
def rod_cantilever_material():
    return dismech.Material(density=1200,
                            youngs_rod=2e6,
                            youngs_shell=0,
                            poisson_rod=0.5,
                            poisson_shell=0)


@pytest.fixture
def rod_cantilever_n26():
    return dismech.Geometry.from_txt(rel_path('resources/rod_cantilever/horizontal_rod_n26.txt'))


@pytest.fixture
def softrobot_cantilever_n26(rod_cantilever_geom, rod_cantilever_material, rod_cantilever_n26, static_2d_sim, free_fall_env):
    return dismech.SoftRobot(rod_cantilever_geom, rod_cantilever_material, rod_cantilever_n26, static_2d_sim, free_fall_env)


@pytest.fixture
def rod_cantilever_n51():
    return dismech.Geometry.from_txt(rel_path('resources/rod_cantilever/horizontal_rod_n51.txt'))


@pytest.fixture
def softrobot_cantilever_n51(rod_cantilever_geom, rod_cantilever_material, rod_cantilever_n51, static_2d_sim, free_fall_env):
    return dismech.SoftRobot(rod_cantilever_geom, rod_cantilever_material, rod_cantilever_n51, static_2d_sim, free_fall_env)


@pytest.fixture
def rod_cantilever_n101():
    return dismech.Geometry.from_txt(rel_path('resources/rod_cantilever/horizontal_rod_n101.txt'))

# parachute


@pytest.fixture
def hexparachute_n6_geom():
    return dismech.GeomParams(rod_r0=1e-3,
                              shell_h=1e-3)


@pytest.fixture
def hexparachute_n6_material():
    return dismech.Material(density=1500,
                            youngs_rod=10e6,
                            youngs_shell=10e8,
                            poisson_rod=0.5,
                            poisson_shell=0.3)


@pytest.fixture
def hexparachute_n6():
    return dismech.Geometry.from_txt(rel_path('resources/parachute/hexparachute_n6_python.txt'))


@pytest.fixture
def softrobot_hexparachute_n6(hexparachute_n6_geom, hexparachute_n6_material, hexparachute_n6, dynamic_3d_sim, drag_fall_env):
    return dismech.SoftRobot(hexparachute_n6_geom, hexparachute_n6_material, hexparachute_n6, dynamic_3d_sim, drag_fall_env)

# pneunet


@pytest.fixture
def pneunet_shorter():
    return dismech.Geometry.from_txt(rel_path('resources/pneunet/input_straight_horizontal_shorter.txt'))


# square plate
@pytest.fixture
def square_plate_30():
    return dismech.Geometry.from_txt(rel_path('resources/square_plate/random_mesh_30.txt'))


# sim params


@pytest.fixture
def static_2d_sim():
    return dismech.SimParams(static_sim=True,
                             two_d_sim=True,
                             use_mid_edge=False,
                             use_line_search=False,
                             show_floor=False,
                             log_data=True,
                             log_step=1,
                             dt=1e-2,
                             max_iter=25,
                             total_time=1.0,
                             plot_step=1,
                             tol=1e-4,
                             ftol=1e-4,
                             dtol=1e-2)


@pytest.fixture
def dynamic_3d_sim():
    return dismech.SimParams(static_sim=False,
                             two_d_sim=False,
                             use_mid_edge=False,
                             use_line_search=False,
                             show_floor=False,
                             log_data=True,
                             log_step=1,
                             dt=1e-2,
                             max_iter=25,
                             total_time=3,
                             plot_step=10,
                             tol=1e-4,
                             ftol=1e-4,
                             dtol=1e-2)

# envs


@pytest.fixture
def free_fall_env():
    env = dismech.Environment()
    env.add_force('gravity', g=np.array([0.0, 0.0, -9.81]))
    return env


@pytest.fixture
def drag_fall_env():
    env = dismech.Environment()
    env.add_force('gravity', g=np.array([0.0, 0.0, -9.81]))
    env.add_force('aerodynamics', rho=1, cd=10)
    return env
