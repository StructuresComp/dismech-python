import pathlib

import numpy as np


def load_nodes(folder: str | pathlib.Path) -> dict[str, np.ndarray]:
    """Load VisuoShell tracked 3D node CSVs from a folder."""
    folder = pathlib.Path(folder)
    files = sorted(folder.glob("*.csv"))

    nodes_by_frame: dict[str, np.ndarray] = {}
    expected_n = None

    for path in files:
        for frame_name, nodes in _load_csv_nodes(path).items():
            if expected_n is None:
                expected_n = nodes.shape[0]
            elif nodes.shape[0] != expected_n:
                continue

            nodes_by_frame[frame_name] = nodes

    return nodes_by_frame


def _load_csv_nodes(path: pathlib.Path) -> dict[str, np.ndarray]:
    with path.open() as f:
        header = f.readline().strip().lower().split(",")
    columns = {name: i for i, name in enumerate(header)}

    if {"frame", "point_id", "x", "y", "z"}.issubset(columns):
        return _load_trajectory_csv(path, columns)

    if {"x", "y", "z"}.issubset(columns):
        nodes = np.loadtxt(
            path,
            delimiter=",",
            skiprows=1,
            usecols=(columns["x"], columns["y"], columns["z"]),
        )
        return {path.name: np.atleast_2d(nodes)}

    return {}


def _load_trajectory_csv(path: pathlib.Path, columns: dict[str, int]) -> dict[str, np.ndarray]:
    data = np.loadtxt(
        path,
        delimiter=",",
        skiprows=1,
        usecols=(
            columns["frame"],
            columns["point_id"],
            columns["x"],
            columns["y"],
            columns["z"],
        ),
    )
    data = np.atleast_2d(data)

    frames: dict[str, np.ndarray] = {}
    frame_ids = np.unique(data[:, 0].astype(np.int64))
    for frame_id in frame_ids:
        frame_data = data[data[:, 0] == frame_id]
        frame_data = frame_data[np.argsort(frame_data[:, 1])]
        frames[f"{path.stem}_frame_{frame_id:06d}.csv"] = frame_data[:, 2:5]
    return frames
