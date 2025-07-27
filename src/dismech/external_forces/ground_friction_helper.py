import sympy as sp

import sympy as sp

def floor_static_friction(v0, v1, v2, f0, f1, f2, mu, K2):
    # Define vectors
    v = sp.Matrix([v0, v1, v2])
    f = sp.Matrix([f0, f1, f2])

    f_norm = sp.sqrt(f0**2 + f1**2 + f2**2)
    n = f / f_norm

    # Tangential component of velocity
    v_dot_n = v.dot(n)
    v_tangent = v - v_dot_n * n

    v_tangent_norm = sp.sqrt(v_tangent.dot(v_tangent))

    # Avoid division by zero
    v_tangent_hat = v_tangent / v_tangent_norm

    v_n_scaled = K2 * v_tangent_norm
    gamma = 2 / (sp.exp(-v_n_scaled)) - 1

    ffr_scalar = gamma * mu * f_norm
    ffr = ffr_scalar * v_tangent_hat

    return [ffr[0], ffr[1], ffr[2]]

def floor_slide_friction(v0, v1, v2, f0, f1, f2, mu, K2):
    # Define vectors
    v = sp.Matrix([v0, v1, v2])
    f = sp.Matrix([f0, f1, f2])

    f_norm = sp.sqrt(f0**2 + f1**2 + f2**2)
    n = f / f_norm

    # Tangential component of velocity
    v_dot_n = v.dot(n)
    v_tangent = v - v_dot_n * n

    v_tangent_norm = sp.sqrt(v_tangent.dot(v_tangent))

    # Avoid division by zero
    v_tangent_hat = v_tangent / v_tangent_norm

    ffr_scalar = mu * f_norm
    ffr = ffr_scalar * v_tangent_hat

    return [ffr[0], ffr[1], ffr[2]]


def get_floor_lambda_fns():
    vars = sp.symbols('v0 v1 v2 f0 f1 f2 mu K2')
    dof = vars[:3]
    fn = vars[3:6]

    static_expr = floor_static_friction(*vars)
    slide_expr = floor_slide_friction(*vars)

    static_jac_dof = sp.Matrix(static_expr).jacobian(dof)  # shape (2, 2)
    static_jac_fn  = sp.Matrix(static_expr).jacobian(fn)  # shape (2, 1)

    slide_jac_dof = sp.Matrix(slide_expr).jacobian(dof)
    slide_jac_fn  = sp.Matrix(slide_expr).jacobian(fn)

    static_jac_dof_lambda = sp.lambdify(vars, static_jac_dof, modules='numpy', cse=True, docstring_limit=0)
    static_jac_fn_lambda = sp.lambdify(vars, static_jac_fn, modules='numpy', cse=True, docstring_limit=0)
    slide_jac_dof_lambda = sp.lambdify(vars, slide_jac_dof, modules='numpy', cse=True, docstring_limit=0)
    slide_jac_fn_lambda = sp.lambdify(vars, slide_jac_fn, modules='numpy', cse=True, docstring_limit=0)

    return static_jac_dof_lambda, static_jac_fn_lambda, slide_jac_dof_lambda, slide_jac_fn_lambda

