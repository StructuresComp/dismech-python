import pytest
import copy
import numpy as np
import scipy

import pathlib

from dismech import get_fs_js, get_fb_jb


def rel_path(fname: str) -> pathlib.Path:
    """
    Localizes path to module path
    """
    return pathlib.Path(__file__).parent / fname

def fs_js_helper(robot, truth):
    Fs, Js = get_fs_js(robot, truth['q'])
    assert(np.allclose(Fs, truth['Fs'].flatten()))
    assert(np.allclose(Js, truth['Js']))

def fb_jb_helper(robot, truth):
    Fb, Jb = get_fb_jb(robot, truth['q'], truth['m1'], truth['m2'])
    assert(np.allclose(Fb, truth['Fb'].flatten()))
    assert(np.allclose(Jb, truth['Jb'], rtol=1e-2)) # Numerical stability issues

def test_get_fs_js_cantilever_n51(softrobot_cantilever_n51):
    robot = softrobot_cantilever_n51
    valid_data = scipy.io.loadmat(
        rel_path('resources/rod_cantilever/rod_cantilever_n51_get_fs_js.mat'))
    fs_js_helper(robot, valid_data)

def test_get_fb_jb_cantilever_n51(softrobot_cantilever_n51):
    robot = softrobot_cantilever_n51
    valid_data = scipy.io.loadmat(
        rel_path('resources/rod_cantilever/rod_cantilever_n51_get_fb_jb_q.mat'))
    fb_jb_helper(robot, valid_data)
    valid_data = scipy.io.loadmat(
        rel_path('resources/rod_cantilever/rod_cantilever_n51_get_fb_jb_q0.mat'))
    fb_jb_helper(robot, valid_data)