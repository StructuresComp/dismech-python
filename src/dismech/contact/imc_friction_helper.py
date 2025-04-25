import sympy as sp


def norm(x0, x1, x2):
    return sp.sqrt(x0**2 + x1**2 + x2**2)


def dot_product(x0, x1, x2, y0, y1, y2):
    return x0 * y0 + x1 * y1 + x2 * y2


def imc_friction_vel_sym(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11,
                         xf0, xf1, xf2, yf0, yf1, yf2,
                         af0, af1, af2, bf0, bf1, bf2,
                         mu, vel_tol):
    
    # Constants
    K2 = 15 / vel_tol

    # Force norms
    f1s_n = norm(xf0, xf1, xf2)
    f1e_n = norm(yf0, yf1, yf2)
    f2s_n = norm(af0, af1, af2)
    f2e_n = norm(bf0, bf1, bf2)
    f1_n = norm(xf0 + yf0, xf1 + yf1, xf2 + yf2)
    f2_n = norm(af0 + bf0, af1 + bf1, af2 + bf2)

    # Contact point weights
    t1 = f1s_n / f1_n
    t2 = 1 - t1
    u1 = f2s_n / f2_n
    u2 = 1 - u1

    # Contact normal
    contact_nx = (xf0 + yf0) / f1_n
    contact_ny = (xf1 + yf1) / f1_n
    contact_nz = (xf2 + yf2) / f1_n

    # Interpolated velocities
    v1x = t1 * v0 + t2 * v3
    v1y = t1 * v1 + t2 * v4
    v1z = t1 * v2 + t2 * v5
    v2x = u1 * v6 + u2 * v9
    v2y = u1 * v7 + u2 * v10
    v2z = u1 * v8 + u2 * v11

    # Relative velocity and projection
    vrx = v1x - v2x
    vry = v1y - v2y
    vrz = v1z - v2z
    dot_vn = dot_product(vrx, vry, vrz, contact_nx, contact_ny, contact_nz)
    tx = vrx - dot_vn * contact_nx
    ty = vry - dot_vn * contact_ny
    tz = vrz - dot_vn * contact_nz
    tv_norm = norm(tx, ty, tz)

    # Unit tangent
    tx_u = tx / tv_norm
    ty_u = ty / tv_norm
    tz_u = tz / tv_norm

    # Gamma modulation
    gamma = (2 / (1 + sp.exp(-K2 * tv_norm))) - 1

    # Stick friction
    fx = gamma * mu * tx_u
    fy = gamma * mu * ty_u
    fz = gamma * mu * tz_u

    # Distribute over nodes
    stick_force = [
        fx * f1s_n, fy * f1s_n, fz * f1s_n,
        fx * f1e_n, fy * f1e_n, fz * f1e_n,
        -fx * f2s_n, -fy * f2s_n, -fz * f2s_n,
        -fx * f2e_n, -fy * f2e_n, -fz * f2e_n
    ]

    # Slide friction (γ = 1)
    fx_sl = mu * tx_u
    fy_sl = mu * ty_u
    fz_sl = mu * tz_u

    slide_force = [
        fx_sl * f1s_n, fy_sl * f1s_n, fz_sl * f1s_n,
        fx_sl * f1e_n, fy_sl * f1e_n, fz_sl * f1e_n,
        -fx_sl * f2s_n, -fy_sl * f2s_n, -fz_sl * f2s_n,
        -fx_sl * f2e_n, -fy_sl * f2e_n, -fz_sl * f2e_n
    ]

    return stick_force, slide_force

def generate_velocity_jacobian_funcs():
    # 12 velocities + 12 force components + 2 params
    vars = sp.symbols(
        'v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 '
        'xf0 xf1 xf2 yf0 yf1 yf2 af0 af1 af2 bf0 bf1 bf2 '
        'mu vel_tol', real=True
    )

    stick_force, slide_force = imc_friction_vel_sym(*vars)

    v_vars = vars[:12]
    f_vars = vars[12:24]

    # Jacobians of force wrt velocities and contact force vectors
    Jv_stick = sp.Matrix(stick_force).jacobian(v_vars)
    Jf_stick = sp.Matrix(stick_force).jacobian(f_vars)

    Jv_slide = sp.Matrix(slide_force).jacobian(v_vars)
    Jf_slide = sp.Matrix(slide_force).jacobian(f_vars)

    dfr_dv_func = sp.lambdify(vars, Jv_stick, modules='numpy', cse=True)
    dfr_df_func = sp.lambdify(vars, Jf_stick, modules='numpy', cse=True)

    dfr_dv_func2 = sp.lambdify(vars, Jv_slide, modules='numpy', cse=True)
    dfr_df_func2 = sp.lambdify(vars, Jf_slide, modules='numpy', cse=True)

    return dfr_dv_func, dfr_df_func, dfr_dv_func2, dfr_df_func2
