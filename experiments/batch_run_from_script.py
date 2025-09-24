#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner for run_pollination_sim.py

Sweeps A_main and freq over predefined grids (or CLI overrides),
calls run_simulation() from run_pollination_sim, and saves:
  - amp_x_flower_hilbert_matrix.csv
  - f_est_matrix.csv
  - results_summary.xlsx  (two sheets)

Usage:
    python batch_run_from_script.py \
        --script /path/to/run_pollination_sim.py \
        --outdir ./batch_results

Optional overrides:
    --a-list "[0.004, 0.006]" --f-list "[0.7, 0.9]"
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


A_MAIN_LIST_DEFAULT = [4e-3, 6e-3, 8e-3, 10e-3, 12e-3]
FREQ_LIST_DEFAULT   = [0.7, 1.0, 1.3, 1.6, 1.9]


def load_run_module(script_path: Path):
    """Dynamically import run_pollination_sim.py from a given path."""
    spec = importlib.util.spec_from_file_location("run_pollination_sim", str(script_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", type=str, required=True, help="Path to run_pollination_sim.py")
    ap.add_argument("--outdir", type=str, default="batch_results", help="Output directory")
    ap.add_argument("--plant_name", type=str, default="tomato2", help="Plant base name (without .txt)")
    ap.add_argument("--position", type=str, default="3", help="Position tag (unused)")
    ap.add_argument("--flower_node", type=int, default=73, help="Flower node index")
    ap.add_argument("--a-list", type=str, default="", help="JSON list for A_main values (override default)")
    ap.add_argument("--f-list", type=str, default="", help="JSON list for freq values (override default)")
    ap.add_argument("--dt-initial", type=float, default=1e-2, help="Initial dt to try")
    ap.add_argument("--dt-min", type=float, default=1e-3, help="Smallest dt to allow")
    ap.add_argument("--dt-factor", type=float, default=0.1, help="Multiply dt by this on each retry")
    ap.add_argument("--max-retries", type=int, default=1, help="Max retries per (A,f) combo")

    args = ap.parse_args()

    script_path = Path(args.script).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Load the run module
    run_mod = load_run_module(script_path)

    # Prepare sweep lists
    A_vals = A_MAIN_LIST_DEFAULT if not args.a_list else json.loads(args.a_list)
    F_vals = FREQ_LIST_DEFAULT if not args.f_list else json.loads(args.f_list)

    # Results matrices
    amp_df = pd.DataFrame(index=F_vals, columns=A_vals, dtype=float)
    f_df   = pd.DataFrame(index=F_vals, columns=A_vals, dtype=float)
    amp_df.index.name = "freq"
    f_df.index.name = "freq"

    # Sweep
    # Sweep with retry-on-failure (shrink dt)
    for f in F_vals:
        for A in A_vals:
            dt_try = float(args.dt_initial)
            tried = 0
            success = False
            last_err = None

            while tried <= args.max_retries and dt_try >= args.dt_min:
                print(f"Running A_main={A}, freq={f}, dt={dt_try} ...")
                try:
                    result = run_mod.run_simulation(
                        plant_name=args.plant_name,
                        position=args.position,
                        A_main=float(A),
                        freq=float(f),
                        flower_node=int(args.flower_node),
                        dt=dt_try,  # << pass dt
                    )
                    amp_df.at[f, A] = result.get("amp_x_flower_hilbert", np.nan)
                    f_df.at[f, A]   = result.get("f_est", np.nan)
                    print(f"  -> SUCCESS (dt_used={result.get('dt_used', dt_try)}): "
                        f"amp_x_flower_hilbert={amp_df.at[f, A]}, f_est={f_df.at[f, A]}")
                    success = True
                    break
                except Exception as e:
                    last_err = e
                    tried += 1
                    # shrink dt and retry
                    dt_try *= float(args.dt_factor)
                    print(f"  -> FAIL (attempt {tried}): {e}\n"
                        f"     Retrying with smaller dt={dt_try} ...")

            if not success:
                # give up on this (A,f); record NaNs so you can see gaps in the grid
                amp_df.at[f, A] = np.nan
                f_df.at[f, A]   = np.nan
                print(f"  -> GAVE UP for A_main={A}, freq={f} after {tried} attempts. "
                    f"Last error: {last_err}")

    # Save CSVs
    amp_csv = outdir / "amp_x_flower_hilbert_matrix.csv"
    f_csv   = outdir / "f_est_matrix.csv"
    amp_df.to_csv(amp_csv, float_format="%.10g")
    f_df.to_csv(f_csv, float_format="%.10g")
    print(f"Saved CSVs:\n  {amp_csv}\n  {f_csv}")

    # Save Excel with two sheets
    xlsx = outdir / "results_summary.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        amp_df.to_excel(writer, sheet_name="amp_x_flower_hilbert")
        f_df.to_excel(writer,   sheet_name="f_est")
    print(f"Saved Excel summary: {xlsx}")


if __name__ == "__main__":
    main()
