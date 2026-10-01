"""
Two helical springs in series: full DER helix vs coarse rods of LoveSprings.

Coil A (r = 10 mm, 4 turns) and coil B (r = 6 mm, 6 turns) share the z axis and meet
at a rigid junction bar through the axis. Both outer ends are clamped and the top is
pulled along z by D. Each coil unwinds by a different amount under tension, so the
free junction rotates until the axial torques of the two coils balance.

    DER helix         full helices with the standard stretch/bend/twist springs (truth)
    Uncoupled rod     coarse straight rod, standard springs with Love's rest (EA, GJ)
    Love rod          coarse straight rod of LoveSprings (nonlinear stretch-twist coupling)
    Love (continuum)  Love's law solved directly for the two-coil chain

Run from this directory: python series.py -> figs/series.png, figs/series.pdf
"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from scipy.optimize import fsolve

import dismech
from dismech.elastics import love_constitutive

FIGS = pathlib.Path(__file__).parent / "figs"

# Coils, wire and load path
rA, rB, turnsA, turnsB = 10e-3, 6e-3, 4, 6
alpha = np.deg2rad(10)                              # pitch angle
hA = turnsA * 2 * np.pi * rA * np.tan(alpha)        # rest axial lengths
hB = turnsB * 2 * np.pi * rB * np.tan(alpha)
wire_r, E, nu = 1e-3, 1e9, 0.3
B = E * np.pi * wire_r ** 4 / 4                     # wire bending stiffness
C = E / (2 * (1 + nu)) * np.pi * wire_r ** 4 / 2    # wire torsional stiffness
NODES_PER_TURN, N_EL = 60, 8                        # truth resolution, coarse elements per coil
STIFF = 100.0                                       # rigid bar stiffness relative to the wire
D = np.linspace(0, 1.5 * (hA + hB), 31)[1:]


def love(eps, vth, r):
    return love_constitutive(eps, vth, r, alpha, B, C)


def make_robot(nodes, edges):
    sim = dismech.SimParams(static_sim=True, two_d_sim=False, use_mid_edge=False,
                            use_line_search=False, show_floor=False, log_data=False,
                            log_step=1, dt=1.0, max_iter=60, total_time=1.0, plot_step=1,
                            tol=1e-6, ftol=1e-12, dtol=1e-14, solver='pardiso', sparse=True)
    material = dismech.Material(density=1000, youngs_rod=E, youngs_shell=0,
                                poisson_rod=nu, poisson_shell=0)
    geo = dismech.Geometry(nodes, edges, np.empty(0))
    return dismech.SoftRobot(dismech.GeomParams(rod_r0=wire_r, shell_h=0), material, geo,
                             sim, dismech.Environment())


def chain(n):
    return np.column_stack([np.arange(n - 1), np.arange(1, n)])


def build_helix():
    """Full helices joined by rigid bars through the axis (end bars, junction bar)."""
    def coil(r, turns, z0, phase):
        phi = np.arange(turns * NODES_PER_TURN + 1) * 2 * np.pi / NODES_PER_TURN
        return np.column_stack([r * np.cos(phase + phi), r * np.sin(phase + phi),
                                z0 + r * np.tan(alpha) * phi])

    xa, xb = coil(rA, turnsA, 0, 0), coil(rB, turnsB, hA, np.pi)   # B starts opposite A's end
    nodes = np.vstack([[0, 0, 0], xa, [0, 0, hA], xb, [0, 0, hA + hB]])
    n, J = len(nodes), len(xa) + 1
    rigid = np.isin(np.arange(n - 1), [0, J - 1, J, n - 2])
    robot = make_robot(nodes, chain(n))
    robot.stretch_springs.EA[rigid] *= STIFF
    touch = rigid[robot.bend_springs.edges_ind].any(axis=1)
    robot.bend_springs.EI[touch] *= STIFF
    robot.twist_springs.GJ[touch] *= STIFF
    return robot, dict(bottom=[0, 1], top=[n - 2, n - 1], J=J, coils=(np.arange(1, J), np.arange(J + 1, n - 1)),
                       x0=nodes)


def build_rod(love_springs):
    """Straight rod along z. Twist DOFs sit on edges, so short end and junction edges keep
    the twist measured over (almost) the full coil lengths."""
    s = 0.05 * min(hA, hB) / N_EL
    zA = np.r_[0, np.linspace(s, hA - s / 2, N_EL + 1)]
    zB = np.r_[np.linspace(hA + s / 2, hA + hB - s, N_EL + 1), hA + hB]
    z = np.r_[zA, zB]
    n, J_edge = z.size, zA.size - 1
    robot = make_robot(np.column_stack([0 * z, 0 * z, z]), chain(n))

    # Love's rest tangent (EA = dR/deps, GJ = dK/dvtheta) and the coil flexural rigidity
    (_, _, _, TA, _, GA), (_, _, _, TB, _, GB) = love(0, 0, rA), love(0, 0, rB)
    in_A = robot.bend_springs.edges_ind[:, 0] < J_edge      # triplet in coil A
    robot.stretch_springs.EA[:] = np.where(np.arange(n - 1) < J_edge, TA, TB)
    robot.stretch_springs.EA[J_edge] = 2 / (1 / TA + 1 / TB)
    robot.twist_springs.GJ[:] = np.where(in_A, GA, GB)
    robot.bend_springs.EI[:] = 2 * np.sin(alpha) * B * C / (B + C)
    if love_springs:
        robot = robot.add_love_springs(coil_r=np.where(in_A, rA, rB), pitch=alpha, EI=B, GJ=C)
    return robot, dict(bottom=[0], top=[n - 1], fixed_edges=[0, n - 2], J_edge=J_edge, z=z)


def pull(robot, info):
    """Quasi-static pull: the top nodes move rigidly along z through D."""
    robot = robot.fix_nodes(np.array(info["bottom"] + info["top"]))
    if "fixed_edges" in info:
        robot = robot.fix_edges(info["fixed_edges"])
    stepper = dismech.ImplicitEulerTimeStepper(robot)
    n, q_ref = robot.n_nodes, robot.state.q.copy()
    top = 3 * np.array(info["top"]) + 2
    out, d_now, prev = dict(Fz=[], phi=[], x=[]), 0.0, None
    for d in D:
        q = robot.state.q.copy()
        if prev is None:                                    # uniform stretch predictor
            q[2:3 * n:3] *= 1 + d / q[2:3 * n:3].max()
        else:                                               # secant predictor
            q += (q - prev[1]) * (d - d_now) / (d_now - prev[0])
        q[top] = q_ref[top] + d
        prev, d_now = (d_now, robot.state.q.copy()), d
        robot, _, _ = stepper.step(stepper._finalize_update(robot, q))
        q = robot.state.q
        force = -sum(e.grad_hess_energy_linear_elastic(robot.state, sparse=True)[0]
                     for e in stepper.elastic_energies.values())
        x = q[:3 * n].reshape(-1, 3)
        if "J_edge" in info:
            phi = q[3 * n + info["J_edge"]]
        else:
            v = x[info["J"] + 1] - x[info["J"]]
            phi = np.arctan2(v[1], v[0]) - np.pi
        out["Fz"].append(force[top].sum())
        out["phi"].append(np.angle(np.exp(1j * phi)))
        out["x"].append(x.copy())
    return {k: np.array(v) for k, v in out.items()}


def solve_continuum():
    """Love's law on the chain: hA epsA + hB epsB = D, R_A = R_B, K_A = K_B."""
    out, x = dict(Fz=[], phi=[]), np.array([D[0] / (hA + hB), 0.0])
    for d in D:
        def residual(x):
            eA, phi = x
            _, RA, KA, *_ = love(eA, phi / hA, rA)
            _, RB, KB, *_ = love((d - hA * eA) / hB, -phi / hB, rB)
            return [RA - RB, (KA - KB) / rA]
        x = fsolve(residual, x, xtol=1e-13)
        out["Fz"].append(love(x[0], x[1] / hA, rA)[1])
        out["phi"].append(x[1])
    return {k: np.array(v) for k, v in out.items()}


def simulate():
    helix, helix_info = build_helix()
    rod_info = build_rod(False)[1]
    return dict(helix=pull(helix, helix_info), uncoupled=pull(build_rod(False)[0], rod_info),
                love=pull(build_rod(True)[0], rod_info), continuum=solve_continuum(),
                helix_info=helix_info, rod_info=rod_info)


# Figure

BLACK, BLUE, ORANGE, GRAY = "#222222", "#2a78d6", "#eb6834", "#8a8983"
RADIAL = 1.5                                        # drawn coil width exaggeration in (a)
AZIM, ELEV = np.deg2rad(-30), np.deg2rad(12)        # view of (a)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
})


def project(x):
    """Orthographic view: screen (u, v) and depth from the axis (larger is farther)."""
    u = x[:, 0] * np.cos(AZIM) - x[:, 1] * np.sin(AZIM)
    y = x[:, 0] * np.sin(AZIM) + x[:, 1] * np.cos(AZIM)
    return np.column_stack([u, x[:, 2] * np.cos(ELEV) + y * np.sin(ELEV)]), y * np.cos(ELEV)


def draw_helix(ax, x, info, rmax):
    """Depth-shaded helix: near wire dark and thick, far wire light and thin."""
    uv, w = project(x)
    i = np.concatenate([c[:-1] for c in info["coils"]])
    t = np.clip((w[i] + w[i + 1]) / (4 * rmax) + 0.5, 0, 1)
    order = np.argsort(-t)                              # far segments first
    t = t[order, None]
    near, far = np.array(to_rgb(BLACK)), np.array(to_rgb("#c4c3bd"))
    ax.add_collection(LineCollection(np.stack([uv[i], uv[i + 1]], axis=1)[order],
                                     colors=near + t * (far - near), linewidths=1.0 - 0.5 * t[:, 0],
                                     capstyle="round"))
    for ends in (info["bottom"], info["top"]):
        ax.plot(*uv[ends].T, color=BLACK, lw=1.0, solid_capstyle="round")
    J = info["J"]
    ax.plot(*uv[[J - 1, J, J + 1]].T, color=ORANGE, lw=1.8, solid_capstyle="round", zorder=5)


def draw_rod(ax, z, J_edge):
    """Love rod skeleton on the helix axis, junction edge marked orange."""
    v = z * np.cos(ELEV)
    ax.plot(0 * v, v, "-o", color=BLUE, lw=0.7, ms=1.6, zorder=4)
    ax.plot(0, v[J_edge:J_edge + 2].mean(), "o", color=ORANGE, ms=2.8, zorder=6)


def main():
    res = simulate()
    hi, ri = res["helix_info"], res["rod_info"]
    Dmm = D * 1e3
    fig = plt.figure(figsize=(3.4, 4.2))
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.05, 1], hspace=0.02)

    # (a) rest and two stretched states, hung from the fixed top end, same scale
    mid = len(D) // 2
    states = [(hi["x0"], ri["z"], 0.0)] + [
        (res["helix"]["x"][k], res["love"]["x"][k][:, 2], Dmm[k]) for k in (mid, -1)]
    rmax = 1.3e3 * RADIAL * max(rA, rB)
    scale = np.array([RADIAL, RADIAL, 1]) * 1e3
    shapes = [(x - x[-1] * [0, 0, 1]) * scale for x, _, _ in states]
    vmin = min(project(x)[0][:, 1].min() for x in shapes)
    for k, (x, (_, z, d)) in enumerate(zip(shapes, states)):
        ax = top.add_axes([0.03 + k * 0.32, 0.20, 0.30, 0.76])
        draw_helix(ax, x, hi, rmax)
        draw_rod(ax, (z - z[-1]) * 1e3, ri["J_edge"])
        ax.set(xlim=(-rmax, rmax), ylim=(vmin - 4, 4), aspect="equal")
        ax.set_axis_off()
        top.text(0.18 + k * 0.32, 0.14, f"$D$ = {d:.0f} mm", ha="center", va="center")
    top.text(0.5, 0.04, "(a)", ha="center", va="center")

    # (b) force and (c) junction rotation, with a shared legend
    styles = dict(helix=dict(color=BLACK, lw=1.5), uncoupled=dict(color=GRAY, lw=1.2, ls="--"),
                  love=dict(color=BLUE, marker="o", ms=2.6, mew=0.7, mfc="none", ls="none"),
                  continuum=dict(color=BLUE, lw=0.7, alpha=0.6))
    bottom.legend([Line2D([], [], **s) for s in styles.values()],
                  ["DER helix", "Uncoupled rod", "Love rod", "Love (continuum)"],
                  loc="upper center", bbox_to_anchor=(0.5, 0.99), ncol=2, fancybox=False,
                  edgecolor=BLACK, columnspacing=1.2, handlelength=1.8, fontsize=7.5)
    axes = bottom.subplots(1, 2, gridspec_kw=dict(left=0.13, right=0.97, top=0.70, bottom=0.25,
                                                  wspace=0.5))
    for ax, key, f, label, tag in zip(axes, ("Fz", "phi"), (1, 180 / np.pi),
                                      ("Axial force, $F$ [N]", r"Junction rotation, $\varphi$ [deg]"),
                                      ("(b)", "(c)")):
        for name, s in styles.items():
            ax.plot(Dmm, f * res[name][key], **s)
        ax.set(xlabel="Displacement, $D$ [mm]", ylabel=label, xlim=(0, Dmm[-1] * 1.03))
        ax.text(0.5, -0.4, tag, transform=ax.transAxes, ha="center", va="top")
    axes[0].set_ylim(bottom=0)

    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "series.png", dpi=300)
    fig.savefig(FIGS / "series.pdf")
    for name in styles:
        F, phi = res[name]["Fz"][-1], np.rad2deg(res[name]["phi"][-1])
        print(f"{name:9s} F(D_max) = {F:.3f} N   phi(D_max) = {phi:+.2f} deg")


if __name__ == "__main__":
    main()
