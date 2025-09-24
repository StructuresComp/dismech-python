#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tomato/Pepper pollination simulation (clean .py version).

This script wraps the simulation logic from a Jupyter notebook into a clean,
runnable Python module. It uses `dismech` to set up geometry, materials,
environment, runs a time-stepper while prescribing motion at selected nodes,
and then computes:
  - amp_x_flower_hilbert (Hilbert-envelope amplitude of flower-node x-motion)
  - f_est (dominant frequency estimate via FFT)

You can choose amplitude and frequency via CLI:
    python run_pollination_sim.py --A_main 0.004 --freq 1.6

Requirements:
    pip install numpy scipy

Notes:
- Assumes `dismech` is importable in your environment.
- Adjust paths to your plant .txt as needed.
"""

import argparse
import os
from typing import Sequence

import numpy as np
from scipy.signal import hilbert, detrend

import dismech


# ---------------------------- Helpers ----------------------------------------

def avg_cycle_amplitude_hilbert(x: np.ndarray) -> float:
    """
    Robust average oscillation amplitude using the analytic-signal envelope.
    Returns the median envelope value (less sensitive to outliers).
    """
    x = np.asarray(x).ravel()
    x = detrend(x)  # remove slow drift
    envelope = np.abs(hilbert(x))
    return float(np.median(envelope))


def estimate_frequency(signal: np.ndarray, dt: float) -> float:
    """
    Estimate dominant frequency of a periodic signal using FFT.

    Parameters
    ----------
    signal : 1D np.ndarray
        The time-varying signal (samples).
    dt : float
        Time step between samples.

    Returns
    -------
    freq : float
        Dominant frequency (Hz).
    """
    signal = np.asarray(signal).ravel()
    n = len(signal)
    if n < 2:
        return float("nan")

    # Remove mean to avoid 0 Hz dominance
    signal = signal - np.mean(signal)

    # FFT
    freqs = np.fft.rfftfreq(n, d=dt)   # frequency bins (Hz)
    fft_vals = np.fft.rfft(signal)     # FFT coefficients
    magnitude = np.abs(fft_vals)

    if magnitude.size <= 1:
        return float("nan")

    # Find the peak frequency (excluding DC = 0 Hz)
    peak_idx = np.argmax(magnitude[1:]) + 1
    return float(freqs[peak_idx])


def build_amps_from_xyz0(
    xyz0: np.ndarray,
    main_grip_index: int,
    other_grip_nodes_below: Sequence[int],
    other_grip_nodes_above: Sequence[int],
    A_main: float,
    all_grip_nodes: np.ndarray,
) -> np.ndarray:
    """
    Create per-node amplitudes for all prescribed nodes based on initial geometry,
    scaling relative to the first node in each group (below/above).
    """
    amps = np.empty(all_grip_nodes.size, float)
    amps[0] = A_main  # main_grip

    # Below group
    below = np.array(other_grip_nodes_below, dtype=int)
    if below.size:
        ref = below[0]
        L = max(np.linalg.norm(xyz0[main_grip_index] - xyz0[ref]), 1e-12)
        for j, node in enumerate(below, start=1):
            amps[j] = A_main * (np.linalg.norm(xyz0[ref] - xyz0[node]) / L)

    # Above group
    above = np.array(other_grip_nodes_above, dtype=int)
    if above.size:
        ref = above[0]
        offset = 1 + below.size
        L = max(np.linalg.norm(xyz0[main_grip_index] - xyz0[ref]), 1e-12)
        for j, node in enumerate(above, start=offset):
            amps[j] = A_main * (np.linalg.norm(xyz0[ref] - xyz0[node]) / L)

    return amps


# ---------------------------- Main routine -----------------------------------

def run_simulation(
    plant_name: str = "pepper0",
    position: str = "3",
    A_main: float = 4e-3,
    freq: float = 1.6,
    flower_node: int = 143,
):
    # --- Geometry / materials / sim params ---
    geom = dismech.GeomParams(
        rod_r0=0.059,
        shell_h=0.0,
    )

    material = dismech.Material(
        density=1500,
        youngs_rod=3.16e7,
        youngs_shell=0,
        poisson_rod=0.5,
        poisson_shell=0,
    )

    dyn_sim = dismech.SimParams(
        static_sim=False,
        two_d_sim=True,   # no twisting
        use_mid_edge=False,
        use_line_search=False,
        show_floor=False,
        log_data=True,
        log_step=1,
        dt=1e-3,
        max_iter=50,
        total_time=5,
        plot_step=100,
        tol=1e-4,
        ftol=1e-4,
        dtol=1e-2,
    )

    env = dismech.Environment()
    env.add_force("gravity", g=np.array([0.0, 0.0, -9.81]))
    env.add_force("damping", eta=100)

    input_txt = os.path.join("../tests/resources/tomato_pollination/plants/", f"{plant_name}.txt")
    geo = dismech.Geometry.from_txt(input_txt)

    robot = dismech.SoftRobot(geom, material, geo, dyn_sim, env)

    # --- grip nodes ---
    main_grip = [71]  # as list to match original usage
    other_grip_nodes_below = np.array([57, 65, 69], dtype=int)
    other_grip_nodes_above = np.array([81, 79, 77, 76], dtype=int)

    # Fix only main_grip and base nodes
    base_nodes = np.where(
        (robot.state.q[robot.node_dof_indices].reshape(-1, 3)[:, 2]) < 0.22
    )[0]
    robot = robot.fix_nodes(np.unique(np.r_[main_grip, base_nodes]))

    # Gather all grip nodes (main first!)
    all_grip_nodes = np.r_[main_grip, other_grip_nodes_below, other_grip_nodes_above].astype(int)

    # Initial positions and targets
    xyz0 = robot.state.q[robot.node_dof_indices].reshape(-1, 3).copy()
    x0 = xyz0[all_grip_nodes, 0].copy()     # initial x of all prescribed nodes
    prev_target_x = x0.copy()               # last absolute target (starts at initial)

    # Build per-node amplitudes
    main_grip_index = int(main_grip[0])
    amps = build_amps_from_xyz0(
        xyz0=xyz0,
        main_grip_index=main_grip_index,
        other_grip_nodes_below=other_grip_nodes_below,
        other_grip_nodes_above=other_grip_nodes_above,
        A_main=A_main,
        all_grip_nodes=all_grip_nodes,
    )

    print("Main grip node:", main_grip)
    print("grasping points:", all_grip_nodes)
    print("Amplitudes (m):", amps)

    # Prescribed motion callback
    def shake_distributed(robot_obj: "dismech.SoftRobot", t: float):
        # absolute target centered at initial pose
        target_x = x0 + amps * np.sin(2 * np.pi * freq * t)
        # increment only
        delta_x = target_x - prev_target_x
        prev_target_x[:] = target_x
        return robot_obj.move_nodes(all_grip_nodes, delta_x.reshape(-1, 1).flatten(), axis=0)

    stepper = dismech.ImplicitEulerTimeStepper(robot)
    stepper.before_step = shake_distributed

    # Simulate
    robots = stepper.simulate()

    # Stack q's
    qs = np.stack([r.state.q for r in robots])

    # Extract flower node trajectory
    flower_node_dofs = robot.map_node_to_dof(flower_node)  # [ix, iy, iz]
    logged = qs[:, flower_node_dofs].squeeze()             # (n_timesteps, 3)

    # Remove value at t=0
    logged = logged - robot.nodes[flower_node]

    # Time array
    t = np.arange(qs.shape[0]) * robot.sim_params.dt

    # Metrics
    amp_x_flower_hilbert = avg_cycle_amplitude_hilbert(logged[:, 0])
    amp_x_flower_peak = (np.max(logged[:, 0]) - np.min(logged[:, 0])) / 2.0
    f_est = estimate_frequency(logged[:, 0], robot.sim_params.dt)

    print(f"Amplitude of flower node {flower_node} in x (peak-to-peak): {amp_x_flower_peak:.6g} m")
    print(f"Amplitude of flower node {flower_node} in x (hilbert): {amp_x_flower_hilbert:.6g} m")
    print(f"Estimated frequency: {f_est:.6g} Hz")

    return {
        "amp_x_flower_hilbert": float(amp_x_flower_hilbert),
        "amp_x_flower_peak": float(amp_x_flower_peak),
        "f_est": float(f_est),
        "A_main": float(A_main),
        "freq": float(freq),
        "flower_node": int(flower_node),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Run the pollination simulation once and print key metrics.")
    p.add_argument("--plant_name", type=str, default="pepper0", help="Plant base name (without .txt)")
    p.add_argument("--position", type=str, default="3", help="Position tag (unused, kept for parity)")
    p.add_argument("--A_main", type=float, default=4e-3, help="Main-grip amplitude in meters")
    p.add_argument("--freq", type=float, default=1.6, help="Excitation frequency in Hz")
    p.add_argument("--flower_node", type=int, default=143, help="Node index for 'flower' tracking")
    return p.parse_args()


def main():
    args = parse_args()
    _ = run_simulation(
        plant_name=args.plant_name,
        position=args.position,
        A_main=args.A_main,
        freq=args.freq,
        flower_node=args.flower_node,
    )


if __name__ == "__main__":
    main()
