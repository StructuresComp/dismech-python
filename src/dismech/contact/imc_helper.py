import numpy as np
import sympy as sp
from sympy.tensor.array import derive_by_array


def dot_product(x0, x1, x2, y0, y1, y2):
    return x0 * y0 + x1 * y1 + x2 * y2


def norm(x0, x1, x2):
    return sp.sqrt(x0**2 + x1**2 + x2**2)


def cross(x0, x1, x2, y0, y1, y2):
    return (
        x1 * y2 - x2 * y1,
        x2 * y0 - x0 * y2,
        x0 * y1 - x1 * y0,
    )


def piecewise_abs(x):
    return sp.Piecewise(
        (x, x > 0),
        (-x, x < 0),
        (0, True)  # At x = 0
    )


def delta_p_to_p(x0, x1, x2, y0, y1, y2, a0, a1, a2, b0, b1, b2):
    return norm(x0 - a0, x1 - a1, x2 - a2)


def delta_p_to_e(x0, x1, x2, y0, y1, y2, a0, a1, a2, b0, b1, b2):
    u0, u1, u2 = cross(b0 - a0, b1 - a1, b2 - a2, a0 - x0, a1 - x1, a2 - x2)
    return norm(u0, u1, u2) / norm(b0 - a0, b1 - a1, b2 - a2)


def delta_e_to_e(x0, x1, x2, y0, y1, y2, a0, a1, a2, b0, b1, b2):
    u0, u1, u2 = cross(y0 - x0, y1 - x1, y2 - x2, b0 - a0, b1 - a1, b2 - a2)
    u_norm = norm(u0, u1, u2)
    proj = dot_product(x0 - a0, x1 - a1, x2 - a2, u0 /
                       u_norm, u1 / u_norm, u2 / u_norm)
    return piecewise_abs(proj)


def get_lambda_fns(expr_fn, vars=sp.symbols('x0 x1 x2 y0 y1 y2 a0 a1 a2 b0 b1 b2', real=True)):
    expr = expr_fn(*vars).doit()

    def batchify(f):
        def batched(*args):
            args = np.stack(args).T
            ret = []
            for a in args:
                ret.append(f(*a))
            return np.stack(ret)
        return batched

    fn = sp.lambdify(vars, expr, modules='numpy', cse=True, docstring_limit=0)
    grad_fn = batchify(sp.lambdify(
        vars, derive_by_array(expr, vars), modules='numpy', cse=True, docstring_limit=0))
    hess_fn = batchify(sp.lambdify(vars, derive_by_array(
        derive_by_array(expr, vars), vars), modules='numpy', cse=True, docstring_limit=0))

    return fn, grad_fn, hess_fn
