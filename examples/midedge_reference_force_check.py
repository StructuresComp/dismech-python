"""Demonstrate why VisuoShellForceEstimator subtracts a bending reference force
for the midedge (TriangleEnergy) path but not for the hinge (HingeEnergy) path.

For the hinge path, springs.nat_strain == get_strain(reference_state) by
construction, so the raw bending gradient at the reference is exactly zero —
the estimator skips the reference-force computation entirely.

For the midedge path, TriangleEnergy bypasses the base-class strain machinery
and uses cached init_ts/init_fs/init_cs/init_xis from compute_tfc_midedge —
these do NOT generally agree with the on-the-fly _get_t_f_c output at the
reference state, so the raw bending gradient at the reference is nonzero. The
estimator calibrates this away by subtracting reference_force.

This script reports the raw and calibrated forces for both paths so you can
see the difference.
"""
import numpy as np

from dismech.visuoshell import VisuoShellForceEstimator


def make_asymmetric_shell(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_phi, n_theta = 6, 5
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    theta = np.linspace(0.2, np.pi - 0.2, n_theta)
    P, T = np.meshgrid(phi, theta)
    pts = np.stack(
        [np.sin(T) * np.cos(P), np.sin(T) * np.sin(P), np.cos(T)], axis=-1
    ).reshape(-1, 3)
    pts += 0.03 * rng.standard_normal(pts.shape)
    pts[:, 2] *= 1.4
    return pts


def report(label: str, estimator: VisuoShellForceEstimator, points: np.ndarray) -> None:
    state = estimator._state_from_nodes(points)
    raw_bend, _ = estimator.energy.grad_hess_energy_linear_elastic(state)
    raw_stretch, _ = estimator.stretch_energy.grad_hess_energy_linear_elastic(state)
    calibrated = estimator.elastic_force(points)

    n_node_dofs = 3 * estimator.n_nodes
    print(f"--- {label} ---")
    print(f"  total DOFs in state.q              = {state.q.shape[0]}")
    print(f"  raw bending grad at reference")
    print(f"      |.|_max (all DOFs)             = {np.abs(raw_bend).max():.3e}")
    print(f"      |.|_max (node DOFs only)       = {np.abs(raw_bend[:n_node_dofs]).max():.3e}")
    if raw_bend.shape[0] > n_node_dofs:
        print(f"      |.|_max (xi DOFs only)         = {np.abs(raw_bend[n_node_dofs:]).max():.3e}")
    if estimator.reference_force.size == 0:
        print(f"  reference_force stored on estimator = (not computed; hinge path skips it)")
    else:
        print(f"  reference_force stored on estimator = {np.abs(estimator.reference_force).max():.3e}")
    print(f"  raw stretch grad at reference       = {np.abs(raw_stretch).max():.3e}")
    print(f"  calibrated elastic_force(reference) = {np.abs(calibrated).max():.3e}")
    print()


def main() -> None:
    points = make_asymmetric_shell()
    print(f"reference cloud: {points.shape[0]} nodes\n")

    hinge = VisuoShellForceEstimator.from_reference_points(points, use_midedge=False)
    midedge = VisuoShellForceEstimator.from_reference_points(points, use_midedge=True)

    report("hinge bending", hinge, points)
    report("midedge bending", midedge, points)

    print("Conclusion:")
    print("  - Hinge raw bending grad at reference is 0 -> reference subtraction is a no-op.")
    print("  - Midedge raw bending grad at reference is NOT 0 -> reference subtraction is required.")
    print("  - Stretch raw grad at reference is 0 in both cases (ref_len matches node spacing).")
    print("  - calibrated elastic_force(reference) is 0 for both, as expected.")


if __name__ == "__main__":
    main()
