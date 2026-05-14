import abc
import warnings

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


class RobustSolver(Solver):
    """Catch-on-failure Tikhonov regularization wrapper.

    On the happy path this is a thin pass-through to the base solver. When the
    base solver raises LinAlgError/ValueError on a singular or near-singular J,
    we retry once with J + reg_factor*I, then once with J + max_reg_factor*I,
    and finally fall back to an SVD pseudo-inverse. We never preemptively
    compute cond(J), so there is no O(n^3) overhead on well-conditioned solves.

    Sparse matrices stay sparse through the regularized retries. Only the
    pseudo-inverse fallback densifies.

    increase_regularization() / reset_regularization() let an outer
    time-stepper escalate the cap after Newton non-convergence and undo it
    once a step succeeds.
    """

    def __init__(self, base_solver=None, reg_factor=1e-8, max_reg_factor=1e-4,
                 verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.base_solver = base_solver if base_solver is not None else NumpySolver()
        self.reg_factor = reg_factor
        self.max_reg_factor = max_reg_factor
        self._original_max_reg_factor = max_reg_factor
        self.verbose = verbose
        self._regularization_count = 0

    def solve(self, J: np.ndarray | sp.csr_matrix, F: np.ndarray):
        try:
            return self.base_solver.solve(J, F)
        except (np.linalg.LinAlgError, ValueError):
            return self._solve_regularized(J, F)

    def _solve_regularized(self, J, F):
        is_sparse = sp.issparse(J)
        # Two-step ladder: light reg first, then the cap.
        for lam in (self.reg_factor, self.max_reg_factor):
            J_reg = self._add_diag(J, lam, is_sparse)
            self._regularization_count += 1
            if self.verbose:
                warnings.warn(
                    f"Singular J; retrying with Tikhonov lambda={lam:.2e}.",
                    RuntimeWarning,
                )
            try:
                return self.base_solver.solve(J_reg, F)
            except (np.linalg.LinAlgError, ValueError):
                continue

        # Last resort: truncated SVD pseudo-inverse on the regularized matrix.
        if self.verbose:
            warnings.warn(
                "Regularized solve failed; falling back to SVD pseudo-inverse.",
                RuntimeWarning,
            )
        J_dense = J_reg.toarray() if is_sparse else J_reg
        U, s, Vt = np.linalg.svd(J_dense, full_matrices=False)
        tol = np.max(s) * np.finfo(s.dtype).eps * max(J_dense.shape)
        s_inv = np.where(s > tol, 1.0 / s, 0.0)
        return Vt.T @ (s_inv * (U.T @ F))

    @staticmethod
    def _add_diag(J, lam, is_sparse):
        if is_sparse:
            return J + sp.identity(J.shape[0], format='csr') * lam
        J_reg = J.copy()
        np.fill_diagonal(J_reg, J_reg.diagonal() + lam)
        return J_reg

    def increase_regularization(self, factor=10.0):
        """Raise the lambda cap (used after Newton non-convergence)."""
        self.max_reg_factor = min(self._original_max_reg_factor * factor, 1e-2)
        if self.verbose:
            print(f"RobustSolver: max_reg_factor -> {self.max_reg_factor:.2e}")

    def reset_regularization(self):
        """Restore the original lambda cap (call after a successful step)."""
        self.max_reg_factor = self._original_max_reg_factor

    def get_stats(self):
        return {
            'regularization_count': self._regularization_count,
            'current_max_reg_factor': self.max_reg_factor,
        }
