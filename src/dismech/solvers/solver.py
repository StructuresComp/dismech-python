import abc

import numpy as np
import scipy
import pypardiso


class Solver(metaclass=abc.ABCMeta):

    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def solve(self, J: np.ndarray, F: np.ndarray):
        pass


class NumpySolver(Solver):

    def __init__(self, **kwargs):
        pass

    def solve(self, J: np.ndarray, F: np.ndarray):
        return np.linalg.solve(J, F)


class PardisoSolver(Solver):

    def __init__(self, **kwargs):
        pass

    def solve(self, J: np.ndarray, F: np.ndarray):
        J_sparse = scipy.sparse.csr_matrix(J)
        return pypardiso.spsolve(J_sparse, F)
