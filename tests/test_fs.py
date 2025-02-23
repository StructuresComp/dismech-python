import pytest
import copy
import numpy as np
import scipy

import pathlib

from dismech import fs


def rel_path(fname: str) -> pathlib.Path:
    """
    Localizes path to module path
    """
    return pathlib.Path(__file__).parent / fname


def fs_js_helper(robot, truth):
    Fs, Js = fs.get_fs_js(robot, truth['q'])
    assert (np.allclose(Fs, truth['Fs'].flatten()))
    assert (np.allclose(Js, truth['Js']))


def fb_jb_helper(robot, truth):
    Fb, Jb = fs.get_fb_jb(robot, truth['q'], truth['m1'], truth['m2'])
    assert (np.allclose(Fb, truth['Fb'].flatten()))
    # Numerical stability issues
    assert (np.allclose(Jb, truth['Jb'], rtol=1e-2))


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


def test_get_fs_js_vectorized_cantilever_n51(softrobot_cantilever_n51):
    robot = softrobot_cantilever_n51
    valid_data = scipy.io.loadmat(
        rel_path('resources/rod_cantilever/rod_cantilever_n51_get_fs_js.mat'))
    Fs, Js = fs.get_fs_js(robot, valid_data['q'].flatten())
    Fs_vec, Js_vec = fs.get_fs_js_vectorized(robot, valid_data['q'].flatten())

    assert (np.allclose(Fs, Fs_vec))
    assert (np.allclose(Js, Js_vec))

def fb_jb_vectorized_helper(robot, truth):
    Fb, Jb = fs.get_fb_jb(
        robot, truth['q'].flatten(), truth['m1'], truth['m2'])
    Fb_vec, Jb_vec = fs.get_fb_jb_vectorized(
        robot, truth['q'].flatten(), truth['m1'], truth['m2'])
    
    assert (np.allclose(Fb, Fb_vec))
    assert (np.allclose(Jb, Jb_vec, rtol=1e-2))

def test_get_fb_jb_vectorized_cantilever_n51(softrobot_cantilever_n51):
    robot = softrobot_cantilever_n51
    valid_data = scipy.io.loadmat(
        rel_path('resources/rod_cantilever/rod_cantilever_n51_get_fb_jb_q.mat'))
    fb_jb_vectorized_helper(robot, valid_data)
    valid_data = scipy.io.loadmat(
        rel_path('resources/rod_cantilever/rod_cantilever_n51_get_fb_jb_q0.mat'))
    fb_jb_vectorized_helper(robot, valid_data)
