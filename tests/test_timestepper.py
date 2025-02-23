import scipy
import numpy as np
import pytest

from dismech import TimeStepper


def test_compute_reference_twist_cantilever_n51(softrobot_cantilever_n51):
    robot = softrobot_cantilever_n51
    TimeStepper(robot, np.array([0, 1, 2, 3, 4, 5]))
