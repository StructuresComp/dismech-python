import pytest
import numpy as np

import dismech


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
def free_fall_env():
    env = dismech.Environment()
    env.add_force('gravity', g=np.array([0.0, 0.0, -9.81]))


def test_rod_cantilever(static_2d_sim, rod_cantilever_geom, rod_cantilever_material, free_fall_env):
    pass
