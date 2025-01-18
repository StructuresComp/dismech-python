import pytest

import geometry

@pytest.fixture
def rod_cantilever_geom():
    b = 0.02
    h = 0.001

    return geometry.GeomParams(rod_r0=0.001,
                               shell_h=0,
                               axs=b*h,
                               ixs1=b*h**3/12,
                               ixs2=h*b**3/12,
                               jxs=b*h**3/6)

# rod cantilever variations

@pytest.fixture
def rod_cantilever_n21():
    return geometry.Geometry.from_txt('resources/rod_cantilever/horizontal_rod_n21.txt')

def test_rod_cantilever_n21_from_txt(rod_cantilever_n21):
    _ = rod_cantilever_n21

@pytest.fixture
def rod_cantilever_n26():
    return geometry.Geometry.from_txt('resources/rod_cantilever/horizontal_rod_n26.txt')

def test_rod_cantilever_n26_from_txt(rod_cantilever_n26):
    _ = rod_cantilever_n26

@pytest.fixture
def rod_cantilever_n51():
    return geometry.Geometry.from_txt('resources/rod_cantilever/horizontal_rod_n51.txt')

def test_rod_cantilever_n51_from_txt(rod_cantilever_n51):
    _ = rod_cantilever_n51

@pytest.fixture
def rod_cantilever_n101():
    return geometry.Geometry.from_txt('resources/rod_cantilever/horizontal_rod_n101.txt')

def test_rod_cantilever_n101_from_txt(rod_cantilever_n101):
    _ = rod_cantilever_n101