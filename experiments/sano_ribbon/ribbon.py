"""
Compressed and sheared ribbon: discrete shell vs Kirchhoff DER vs Sano rod.

A ribbon (L x W x b) is clamped at both ends and the far clamp is translated along
a two-phase path: compress by dx = -0.2 L (it buckles into an arch), then shear by
dy up to 0.3 L at fixed dx (the arch twists and snaps). All models start from the
same small bump so the buckling direction is well defined.

    Shell   mid-edge discrete shell on a structured 41 x 5 mesh (reference)
    DER     anisotropic Kirchhoff rod on the shell's node columns,
            EI1 = E b W^3/12, EI2 = E W b^3/(12 (1 - nu^2)), GJ = G W b^3/3
    Sano    same rod with Sano's bend-twist coupling, zeta^2 = (1 - nu) W^4 / (60 b^2)

Each load step is a static Newton solve from a secant predictor. Past a fold no
nearby equilibrium exists, so the step falls back to overdamped implicit-Euler
dynamics until the ribbon settles on the next stable branch; converged equilibria
with a negative Hessian eigenvalue are nudged off along that mode and relaxed.

Run from this directory: python ribbon.py -> figs/ribbon.png, figs/ribbon.pdf
"""
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import dismech

FIGS = pathlib.Path(__file__).parent / "figs"

# Ribbon, mesh and load path
L, W, b, E, nu = 0.2, 0.02, 5e-4, 1e7, 0.3
NX, NY = 41, 5                                      # node columns, nodes across the width (odd)
ZETA = np.sqrt((1 - nu) * W ** 4 / (60 * b ** 2))
N_COMPRESS = 20
PATH = np.vstack([np.column_stack([-np.linspace(0, 0.2, N_COMPRESS + 1)[1:] * L, np.zeros(N_COMPRESS)]),
                  np.column_stack([np.full(60, -0.2 * L), np.linspace(0, 0.3, 61)[1:] * L])])


def rest_z(x):
    """Rest-shape imperfection shared by every model: a small +z bump."""
    return 5e-3 * L * np.sin(np.pi * x / L) ** 2


def sim_params(**kw):
    return dismech.SimParams(**{**dict(
        static_sim=True, two_d_sim=False, use_mid_edge=False, use_line_search=False, show_floor=False,
        log_data=False, log_step=1, dt=1.0, max_iter=100, total_time=1.0, plot_step=1,
        tol=1e-9, ftol=1e-10, dtol=1e-12), **kw})


class Ribbon:
    """Quasi-static displacement control of the far clamp."""

    def __init__(self, robot, clamp_a, clamp_b):
        x0 = robot.state.q[:3 * robot.n_nodes].reshape(-1, 3)
        self.x_rest = x0[:, 0].copy()
        self.clamp_a, self.clamp_b = clamp_a, clamp_b
        self.x_a0, self.x_b0 = x0[clamp_a].copy(), x0[clamp_b].copy()
        self.interior = np.setdiff1d(np.arange(robot.n_nodes), np.r_[clamp_a, clamp_b])
        self.robot = robot.fix_nodes(np.r_[clamp_a, clamp_b])
        self.stepper = dismech.ImplicitEulerTimeStepper(self.robot)
        self.u, self.prev, self.jumps = np.zeros(2), None, 0    # jumps: branch changes so far

    def predict(self, du):
        """Secant extrapolation along the path; rigid-shift predictor at the start of a phase."""
        q, n = self.robot.state.q.copy(), self.robot.n_nodes
        if self.prev is not None:
            q_prev, du_prev = self.prev
            step = np.linalg.norm(du_prev)
            if step > 0 and np.isclose(du @ du_prev, np.linalg.norm(du) * step):
                return q + (q - q_prev) * np.linalg.norm(du) / step
        w = (self.x_rest / L)[:, None]
        w[self.clamp_a], w[self.clamp_b] = 0.0, 1.0
        x = q[:3 * n].reshape(-1, 3)
        x[:, :2] += w * du
        # beyond the Euler load a straight predictor lands on the unstable flat branch:
        # lift the bump to the inextensible buckling amplitude instead
        shortening = -(self.u[0] + du[0])
        if du[0] < 0 and shortening > 0:
            A, zmax = 2 / np.pi * np.sqrt(shortening * L), x[self.interior, 2].max()
            if 0 < zmax < 0.5 * A:
                x[self.interior, 2] *= A / zmax
        q[:3 * n] = x.ravel()
        return q

    def solve(self, u):
        du = u - self.u
        q = self.predict(du)
        x = q[:3 * self.robot.n_nodes].reshape(-1, 3)
        x[self.clamp_a], x[self.clamp_b] = self.x_a0, self.x_b0
        x[self.clamp_b, :2] += u
        start, q_old = self.stepper._finalize_update(self.robot, q), self.robot.state.q.copy()
        try:
            self.robot, _, _ = self.stepper.step(start)
        except ValueError:                          # past a fold: let the ribbon snap
            self.robot = self.relax(start)
            self.jumps += 1
        self.stabilize()
        self.prev, self.u = (q_old, du), u

    def solve_adaptive(self, u, depth=0):
        """solve, bisecting the load increment when Newton fails."""
        try:
            self.solve(u)
        except ValueError:
            if depth == 5:
                raise
            self.solve_adaptive(0.5 * (self.u + u), depth + 1)
            self.solve_adaptive(u, depth + 1)

    def stabilize(self, max_tries=3):
        """Static Newton also converges to unstable equilibria: nudge along the
        negative-curvature mode of the Hessian and relax."""
        for _ in range(max_tries):
            r = self.robot
            q, fd = r.state.q, r.state.free_dof
            _, J = self.stepper._compute_forces_and_jacobian(np.zeros(q.size), np.zeros((q.size, q.size)),
                                                             r, q, np.zeros_like(q))
            H = (J.toarray() if hasattr(J, "toarray") else J)[np.ix_(fd, fd)]
            lam, vec = np.linalg.eigh(0.5 * (H + H.T))
            if lam[0] > -1e-9:
                return
            v = vec[:, 0]
            q = q.copy()
            q[fd] += 1e-3 * L * v / np.abs(v[fd < 3 * r.n_nodes]).max()
            self.robot = self.relax(self.stepper._finalize_update(r, q))
            self.jumps += 1

    def relax(self, robot, dt=1e-2, dt_max=1e4, max_steps=400):
        """Overdamped implicit-Euler dynamics (velocity reset every step) with dt grown
        until the step is static, so the result is a stable equilibrium."""
        sim = robot.sim_params
        sim.static_sim = False
        try:
            for _ in range(max_steps):
                sim.dt = dt
                try:
                    robot, _, _ = self.stepper.step(robot.update(u=np.zeros_like(robot.state.u)))
                except ValueError:
                    dt /= 4
                    continue
                if dt >= dt_max:
                    return robot
                dt *= 2
            raise ValueError("damped relaxation did not settle")
        finally:
            sim.static_sim, sim.dt = True, 1.0


class Shell(Ribbon):
    def __init__(self):
        X, Y = np.meshgrid(np.linspace(0, L, NX), np.linspace(-W / 2, W / 2, NY), indexing="ij")
        nodes = np.column_stack([X.ravel(), Y.ravel(), rest_z(X.ravel())])
        self.idx = idx = np.arange(NX * NY).reshape(NX, NY)
        faces = []
        for i in range(NX - 1):
            for j in range(NY - 1):
                a, b_, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
                # alternate diagonals so the mesh has no preferred shear direction
                faces += [[a, b_, c], [a, c, d]] if (i + j) % 2 == 0 else [[a, b_, d], [b_, c, d]]
        self.faces = np.array(faces)
        robot = dismech.SoftRobot(
            dismech.GeomParams(rod_r0=0, shell_h=b),
            dismech.Material(density=1000, youngs_rod=0, youngs_shell=E, poisson_rod=0, poisson_shell=nu),
            dismech.Geometry(nodes, np.empty((0, 2), dtype=np.int64), self.faces),
            # mid-edge Newton converges linearly near the round-off floor
            sim_params(use_mid_edge=True, sparse=True, solver="pardiso", tol=1e-7, ftol=1e-7, max_iter=300),
            dismech.Environment())
        super().__init__(robot, idx[:2].ravel(), idx[-2:].ravel())

    def grid(self):
        return self.robot.state.q[:3 * self.robot.n_nodes].reshape(-1, 3)[self.idx]

    def centerline(self):
        return self.grid()[:, NY // 2]

    def director(self):
        """Unit width direction per node column (sign-free): the chord across it."""
        d = self.grid()[:, -1] - self.grid()[:, 0]
        return d / np.linalg.norm(d, axis=1, keepdims=True)


class Rod(Ribbon):
    def __init__(self, sano):
        x = np.linspace(0, L, NX)
        robot = dismech.SoftRobot(
            dismech.GeomParams(rod_r0=b, shell_h=0, axs=W * b, ixs1=b * W ** 3 / 12,   # hard: in-plane
                               ixs2=W * b ** 3 / 12, jxs=W * b ** 3 / 3),              # easy: out-of-plane
            dismech.Material(density=1000, youngs_rod=E, youngs_shell=0, poisson_rod=nu, poisson_shell=0),
            dismech.Geometry(np.column_stack([x, 0 * x, rest_z(x)]),
                             np.column_stack([np.arange(NX - 1), np.arange(1, NX)]), np.empty(0)),
            sim_params(use_line_search=sano), dismech.Environment())
        robot.bend_springs.EI[:, 1] /= 1 - nu ** 2      # plate bending stiffness: the ribbon is wide
        # the space-parallel frame puts m1 along the normal (z); rotate it onto the width (y)
        q = robot.state.q.copy()
        q[3 * NX:] = np.pi / 2
        m1, m2 = robot.compute_material_directors(q, robot.state.a1, robot.state.a2)
        robot = robot.update(q=q, m1=m1, m2=m2)
        if sano:
            robot = robot.add_sano_springs(zeta=ZETA)
        super().__init__(robot, np.array([0, 1]), np.array([NX - 2, NX - 1]))

    def centerline(self):
        return self.robot.state.q[:3 * NX].reshape(-1, 3)

    def director(self):
        """m1 averaged onto the nodes (end nodes take their only edge)."""
        m1 = self.robot.state.m1
        d = np.vstack([m1[:1], 0.5 * (m1[:-1] + m1[1:]), m1[-1:]])
        return d / np.linalg.norm(d, axis=1, keepdims=True)


def simulate():
    """Walk all three models along PATH; record every step (index 0 is the rest state)."""
    models = dict(shell=Shell(), der=Rod(sano=False), sano=Rod(sano=True))
    out = {name: dict(x=[], d=[], jumps=[]) for name in models}
    grids = []
    for k in range(len(PATH) + 1):
        if k:
            for m in models.values():
                m.solve_adaptive(PATH[k - 1])
            print(f"[{k:2d}/{len(PATH)}] dx/L = {PATH[k - 1, 0] / L:+.3f}  dy/L = {PATH[k - 1, 1] / L:+.3f}",
                  flush=True)
        grids.append(models["shell"].grid().copy())
        for name, m in models.items():
            out[name]["x"].append(m.centerline().copy())
            out[name]["d"].append(m.director().copy())
            out[name]["jumps"].append(m.jumps)
    res = {name: {k: np.array(v) for k, v in r.items()} for name, r in out.items()}
    return res, np.array(grids), models["shell"]


def on_shell_branch(x, d, xs, ds, u):
    """Past a symmetry-breaking bifurcation a rod may take the mirror-image branch:
    map it by the problem's C2 symmetry (rotation by pi about the vertical axis
    through the clamps' midpoint) when that is closer to the shell, and align the
    sign-free width directors with the shell's."""
    xm = x[::-1] * [-1, -1, 1] + [L + u[0], u[1], 0]
    if np.linalg.norm(xm - xs, axis=1).max() < np.linalg.norm(x - xs, axis=1).max():
        x, d = xm, d[::-1] * [-1, -1, 1]
    return x, d * np.sign(np.sum(d * ds, axis=1, keepdims=True))


# Figure

BLACK, BLUE, ORANGE, GRAY = "#222222", "#2a78d6", "#eb6834", "#8a8983"
TOP, UNDER, CLAMP, SHADOW = "#d6d4cd", "#8c897f", "#4a4a4a", "#ebeae5"
ELEV, AZIM = 32, -58
LIGHT = np.array([-0.35, -0.55, 1.0]) / np.linalg.norm([-0.35, -0.55, 1.0])
STYLES = dict(shell=dict(color=BLACK, lw=1.8), der=dict(color=BLUE, lw=1.2, ls=(0, (4, 2))),
              sano=dict(color=ORANGE, lw=1.2))
LABELS = ["Shell (reference)", "Kirchhoff DER", "Sano rod"]
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 11, "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
})


def shade(tri, clamped):
    """Two-sided Lambert shading: the rest top face light, the underside dark, so
    twist shows as the underside turning into view."""
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    e, a = np.deg2rad(ELEV), np.deg2rad(AZIM)
    facing = n @ [np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)] > 0
    base = np.where(facing[:, None], to_rgb(TOP), to_rgb(UNDER))
    base[clamped] = to_rgb(CLAMP)
    return base * (0.55 + 0.45 * np.abs(n @ LIGHT))[:, None]


def draw_3d(ax, x, rods, faces, clamped, lims, u, rest_clamp):
    """Shell with its floor shadow, the moving clamp's rest outline and travel, and
    the rod centerlines drawn on top."""
    ax.computed_zorder = False
    tri = x[faces]
    floor = tri.copy()
    floor[..., 2] = lims[2, 0]
    ax.add_collection3d(Poly3DCollection(floor, facecolors=SHADOW, edgecolors=SHADOW, linewidths=0.3, zorder=0))
    c = rest_clamp.mean(axis=0)
    for line in (rest_clamp[[0, 1, 2, 3, 0]], np.array([c, c + [*u, 0]])):
        ax.plot(*line.T, color=BLACK, lw=0.6, ls=(0, (2, 1.5)), zorder=1)
    ax.add_collection3d(Poly3DCollection(tri, facecolors=shade(tri, clamped), edgecolors=(0, 0, 0, 0.15),
                                         linewidths=0.2, zorder=2))
    for name, xr in rods.items():
        ax.plot(*xr.T, **{**STYLES[name], "lw": 1.1}, zorder=3)
    ax.set(xlim=lims[0], ylim=lims[1], zlim=lims[2])
    ax.set_box_aspect(np.ptp(lims, axis=1), zoom=1.05)
    ax.set_proj_type("ortho")
    ax.view_init(ELEV, AZIM)
    ax.set_axis_off()


def main():
    res, grids, shell = simulate()
    u = np.vstack([[0, 0], PATH]) / L

    # end of compression, just before and after the shell's twist snap, after the rods snap, end
    shell_snap = np.flatnonzero(np.diff(res["shell"]["jumps"]))[0]
    rod_snap = np.flatnonzero(np.diff(res["der"]["jumps"]))[-1] + 1
    steps = [N_COMPRESS, shell_snap, shell_snap + 2, rod_snap, len(u) - 1]
    states = [{name: on_shell_branch(r["x"][k], r["d"][k], res["shell"]["x"][k], res["shell"]["d"][k], u[k] * L)
               for name, r in res.items()} for k in steps]

    clamped = np.isin(shell.faces, np.r_[shell.idx[:2].ravel(), shell.idx[-2:].ravel()]).all(axis=1)
    g0 = grids[0] / L
    rest_clamp = np.array([g0[-2, 0], g0[-1, 0], g0[-1, -1], g0[-2, -1]])
    xs = [grids[k].reshape(-1, 3) / L for k in steps]
    allx = np.vstack(xs + [rest_clamp])
    lims = np.column_stack([allx.min(axis=0) - 0.02, allx.max(axis=0) + 0.02])
    lims[2, 0] = -0.04                              # floor a little below the ribbon

    fig = plt.figure(figsize=(7.0, 3.6))
    top, bottom = fig.subfigures(2, 1, height_ratios=[0.8, 1.15])
    w = 1 / len(steps)

    # (a) shell snapshots with the rod centerlines
    for k, (step, x, state) in enumerate(zip(steps, xs, states)):
        ax = top.add_axes([k * w, 0.06, w, 1.12], projection="3d")
        rods = {name: state[name][0] / L for name in ("der", "sano")}
        draw_3d(ax, x, rods, shell.faces, clamped, lims, u[step], rest_clamp)
        dy = rf"{u[step, 1]:.3g}L" if u[step, 1] else "0"
        top.text((k + 0.5) * w, 0.13, rf"({'i ii iii iv v'.split()[k]}) $\Delta y = {dy}$",
                 ha="center", va="center", fontsize=9.5)
    top.text(0.5, 0.0, "(a)", ha="center", va="center")

    # (b) width tilt out of the xy-plane along the ribbon (the twist)
    axes = bottom.subplots(1, len(steps), sharey=True,
                           gridspec_kw=dict(left=0.085, right=0.99, top=0.72, bottom=0.28, wspace=0.12))
    s = np.linspace(0, 1, NX)
    for ax, state in zip(axes, states):
        ax.axhline(0, color=GRAY, lw=0.5, zorder=0)
        for name, (_, d) in state.items():
            ax.plot(s, np.rad2deg(np.arcsin(d[:, 2].clip(-1, 1))), **STYLES[name])
        ax.set(xlim=(0, 1), xticks=[0, 0.5, 1], xticklabels=["0", "0.5", "1"])
        ax.set_xlabel("$s / L$", labelpad=1)
        ax.tick_params(labelsize=9)
    axes[0].set(ylabel="Width tilt [deg]", ylim=(-60, 60), yticks=[-45, 0, 45])
    bottom.legend([Line2D([], [], **st) for st in STYLES.values()], LABELS, loc="upper center",
                  bbox_to_anchor=(0.5, 0.93), ncol=3, fancybox=False, edgecolor=BLACK, columnspacing=1.4,
                  fontsize=9.5)
    bottom.text(0.5, 0.04, "(b)", ha="center", va="center")

    FIGS.mkdir(exist_ok=True)
    fig.savefig(FIGS / "ribbon.png", dpi=300)
    fig.savefig(FIGS / "ribbon.pdf")
    print("panels at dy/L =", u[steps, 1].round(3).tolist())


if __name__ == "__main__":
    main()
