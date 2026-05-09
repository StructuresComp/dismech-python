import argparse
import pathlib

import numpy as np

from dismech.visuoshell import (
    VisuoShellForceEstimator,
    VisuoShellVisualizationOptions,
    get_force_animation_plotly,
    load_nodes,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, default=pathlib.Path("forces.txt"))
    parser.add_argument("--html-output", type=pathlib.Path)
    parser.add_argument("--kb", type=float, default=1.0e9)
    parser.add_argument("--nu", type=float, default=0.5)
    parser.add_argument("--use-midedge", action="store_true")
    parser.add_argument(
        "--mesh-method",
        choices=("convex_hull", "delaunay"),
        default="convex_hull",
    )
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--plot-step", type=int, default=1)
    parser.add_argument("--arrow-scale", type=float, default=1.0)
    args = parser.parse_args()

    nodes_by_frame = load_nodes(args.data_dir)
    if not nodes_by_frame:
        raise ValueError(f"no CSV frames found in {args.data_dir}")

    reference_nodes = next(iter(nodes_by_frame.values()))
    estimator = VisuoShellForceEstimator.from_reference_points(
        reference_nodes,
        kb=args.kb,
        nu=args.nu,
        mesh_method=args.mesh_method,
        use_midedge=args.use_midedge,
    )

    forces_by_frame = {}
    with args.output.open("w") as f:
        for frame_name, nodes in nodes_by_frame.items():
            force = estimator.external_balance_force(nodes)
            forces_by_frame[frame_name] = force
            f.write(f"{frame_name}\n")
            np.savetxt(f, force, delimiter=",")
            f.write("\n")

    if args.show or args.html_output:
        options = VisuoShellVisualizationOptions(
            plot_step=args.plot_step,
            arrow_scale=args.arrow_scale,
        )
        fig = get_force_animation_plotly(
            nodes_by_frame,
            forces_by_frame,
            estimator.triangles,
            options=options,
        )
        if args.html_output:
            fig.write_html(args.html_output)
        if args.show:
            fig.show()


if __name__ == "__main__":
    main()
