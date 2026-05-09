#!/usr/bin/env python3
"""Extract reduced slinky training data from raw trajectory npz files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def output_stem(raw_path: Path) -> str:
    stem = raw_path.stem
    if stem.endswith("_raw"):
        return stem[:-4]
    if stem.endswith("raw"):
        return stem[:-3].rstrip("_-")
    return stem.replace("raw", "").rstrip("_-")


def load_raw(raw_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(raw_path)
    required = {"qs", "F", "t"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"{raw_path} is missing required arrays: {sorted(missing)}")

    qs = data["qs"]
    forces = data["F"]
    times = data["t"]

    if qs.ndim != 2 or forces.ndim != 2 or times.ndim != 1:
        raise ValueError(f"{raw_path} must contain 2D qs/F arrays and a 1D t array")
    if qs.shape != forces.shape or qs.shape[0] != times.shape[0]:
        raise ValueError(f"{raw_path} has inconsistent qs/F/t shapes")
    if (qs.shape[1] + 1) % 4 != 0:
        raise ValueError(f"{raw_path} does not look like [3*N node dofs, N-1 theta dofs]")

    return qs, forces, times


def prepare_raw_arrays(
    qs: np.ndarray,
    forces: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    n_nodes = (qs.shape[1] + 1) // 4
    print(f"Detected {n_nodes} nodes from qs shape {qs.shape}")

    first_idx = 1
    last_idx = min(3*n_nodes, qs.shape[0])
    if last_idx <= first_idx:
        raise ValueError("raw trajectory is too short after dropping the first force frame")

    q_filtered = qs[first_idx:last_idx]
    f_filtered = forces[first_idx:last_idx]
    t_filtered = times[first_idx:last_idx]

    # print shapes for debugging
    print(f"q_filtered shape: {q_filtered.shape}")
    print(f"f_filtered shape: {f_filtered.shape}")
    print(f"t_filtered shape: {t_filtered.shape}")

    positions = q_filtered[:, : 3 * n_nodes]
    positions_reshaped = positions.reshape(q_filtered.shape[0], n_nodes, 3)
    return q_filtered, f_filtered, t_filtered, positions_reshaped, n_nodes


def extract_3_noded(
    f_filtered: np.ndarray,
    positions_reshaped: np.ndarray,
) -> dict[str, np.ndarray]:
    num_t = positions_reshaped.shape[0]
    node0 = positions_reshaped[:, 0]
    node_last = positions_reshaped[:, -1]
    centroid = np.mean(positions_reshaped, axis=1)

    qs_to_train = np.zeros((num_t, 11))
    qs_to_train[:, 0:3] = node0
    qs_to_train[:, 3] = 0.0
    qs_to_train[:, 4] = centroid[:, 0]
    qs_to_train[:, 6] = centroid[:, 2]
    qs_to_train[:, 7] = 0.0
    qs_to_train[:, 8:11] = node_last

    idx_b = np.array([0, 1, 2, 3, 7, 8, 9, 10])
    xb = np.zeros((num_t, len(idx_b)))
    xb[:, 5:8] = node_last

    f_train = np.column_stack(
        [
            f_filtered[:, 0] + f_filtered[:, 3],
            np.zeros(num_t),
            f_filtered[:, 2] + f_filtered[:, 5],
        ]
    )

    return with_trajectory_dim(qs_to_train, f_train, xb, idx_b)


def extract_5_noded(
    f_filtered: np.ndarray,
    positions_reshaped: np.ndarray,
) -> dict[str, np.ndarray]:
    num_t = positions_reshaped.shape[0]
    n_nodes = positions_reshaped.shape[1]

    node0 = positions_reshaped[:, 0]
    node_last = positions_reshaped[:, -1]
    centroid = np.mean(positions_reshaped, axis=1)

    half_idx = n_nodes // 2
    centroid_first_half = np.mean(positions_reshaped[:, :half_idx], axis=1)
    centroid_second_half = np.mean(positions_reshaped[:, half_idx:], axis=1)

    qs_to_train = np.zeros((num_t, 19))
    qs_to_train[:, 0:3] = node0
    qs_to_train[:, 3] = 0.0
    qs_to_train[:, 4] = centroid_first_half[:, 0]
    qs_to_train[:, 6] = centroid_first_half[:, 2]
    qs_to_train[:, 7] = 0.0
    qs_to_train[:, 8] = centroid[:, 0]
    qs_to_train[:, 10] = centroid[:, 2]
    qs_to_train[:, 11] = 0.0
    qs_to_train[:, 12] = centroid_second_half[:, 0]
    qs_to_train[:, 14] = centroid_second_half[:, 2]
    qs_to_train[:, 15] = 0.0
    qs_to_train[:, 16:19] = node_last

    idx_b = np.array([0, 1, 2, 3, 15, 16, 17, 18])
    xb = np.zeros((num_t, len(idx_b)))
    xb[:, 5:8] = node_last

    f_train = np.column_stack(
        [
            f_filtered[:, 0],
            np.zeros(num_t),
            f_filtered[:, 2],
        ]
    )

    return with_trajectory_dim(qs_to_train, f_train, xb, idx_b)


def with_trajectory_dim(
    qs: np.ndarray,
    forces: np.ndarray,
    xb: np.ndarray,
    idx_b: np.ndarray,
) -> dict[str, np.ndarray]:
    qs = qs[np.newaxis]
    forces = forces[np.newaxis]
    xb = xb[np.newaxis]
    lambdas = np.linspace(0, 1, qs.shape[1])[np.newaxis]
    valid = np.full(qs.shape[:2], True)

    return {
        "qs": qs,
        "F": forces,
        "xb": xb,
        "idx_b": idx_b,
        "lambdas": lambdas,
        "valid": valid,
    }


def extract_file(raw_path: Path) -> tuple[Path, Path]:
    qs, forces, times = load_raw(raw_path)
    _, f_filtered, _, positions_reshaped, _ = prepare_raw_arrays(qs, forces, times)

    base_stem = output_stem(raw_path)
    out_3 = raw_path.with_name(f"{base_stem}.npz")
    out_5 = raw_path.with_name(f"{base_stem}_5_noded.npz")

    np.savez(out_3, **extract_3_noded(f_filtered, positions_reshaped))
    np.savez(out_5, **extract_5_noded(f_filtered, positions_reshaped))

    return out_3, out_5


def find_raw_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.npz")
        if "raw" in path.stem and path.stem.endswith("raw")
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract 3-node and 5-node centerline training data from raw slinky npz files.",
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory to search recursively. Defaults to the experiments directory.",
    )
    args = parser.parse_args()

    raw_files = find_raw_files(args.root)
    if not raw_files:
        print(f"No raw npz files found under {args.root}")
        return

    for raw_path in raw_files:
        out_3, out_5 = extract_file(raw_path)
        print(f"{raw_path} -> {out_3}, {out_5}")


if __name__ == "__main__":
    main()
