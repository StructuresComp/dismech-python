import abc

import numpy as np
import scipy.sparse as sp


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
        if isinstance(J, sp.csr_matrix):
            print("[WARNING] Using numpy (a dense solver) for a sparse matrix")
            J = J.toarray()
        return np.linalg.solve(J, F)


class PardisoSolver(Solver):

    def __init__(self, **kwargs):
        try:
            import pypardiso
        except ImportError:
            raise ImportError("pypardiso is required for PardisoSolver but not installed. Please install it using:\n"
                              "pip install pypardiso"
                              )
        else:
            self.pardiso = pypardiso

    def solve(self, J: np.ndarray | sp.csr_matrix, F: np.ndarray):
        if isinstance(J, np.ndarray):
            print("[WARNING] Using Pardiso (a sparse solver) for a dense matrix")
            J = sp.csr_matrix(J)
        return self.pardiso.spsolve(J, F)
