import dataclasses
import math
import pathlib
import time
import typing

import numpy as np
import plotly.graph_objects as go


@dataclasses.dataclass
class VisuoShellVisualizationOptions:
    """Options for visualizing tracked VisuoShell marker forces."""

    title: str = "VisuoShell Force Estimation"
    colorscale: str = "Hot"
    marker_size: int = 4
    edge_color: str = "#333333"
    arrow_color: str = "#1f78b4"
    arrow_scale: float = 1.0
    pyvista_pause: float = 0.5
    pyvista_show_edges: bool = True
    pyvista_off_screen: bool = False
    pyvista_image_dir: str | pathlib.Path | None = None
    pyvista_gif_path: str | pathlib.Path | None = None
    pyvista_gif_duration_ms: int = 100
    camera_view: tuple[float, float] = (30.0, 45.0)
    x_lim: tuple[float, float] | None = None
    y_lim: tuple[float, float] | None = None
    z_lim: tuple[float, float] | None = None
    plot_step: int = 1


def get_force_figure(
    nodes: np.ndarray,
    forces: np.ndarray,
    triangles: np.ndarray,
    title: str | None = None,
    options: VisuoShellVisualizationOptions | None = None,
) -> go.Figure:
    """Return a Plotly figure for one VisuoShell frame."""
    options = options or VisuoShellVisualizationOptions()
    nodes, forces, triangles = _validate_inputs(nodes, forces, triangles)
    force_magnitude = np.linalg.norm(forces, axis=1)
    arrow_vectors = _arrow_vectors(forces, options.arrow_scale)
    edge_x, edge_y, edge_z = _edge_lines(nodes, triangles)
    i, j, k = _triangle_indices(triangles)

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=nodes[:, 0],
                y=nodes[:, 1],
                z=nodes[:, 2],
                i=i,
                j=j,
                k=k,
                intensity=force_magnitude,
                colorscale=options.colorscale,
                showscale=True,
                colorbar=dict(title="|F|"),
                name="Force magnitude",
            ),
            go.Scatter3d(
                x=edge_x,
                y=edge_y,
                z=edge_z,
                mode="lines",
                line=dict(color=options.edge_color, width=2),
                name="Edges",
                showlegend=False,
            ),
            go.Scatter3d(
                x=nodes[:, 0],
                y=nodes[:, 1],
                z=nodes[:, 2],
                mode="markers",
                marker=dict(
                    size=options.marker_size,
                    color=force_magnitude,
                    colorscale=options.colorscale,
                    showscale=False,
                ),
                name="Markers",
            ),
            go.Cone(
                x=nodes[:, 0],
                y=nodes[:, 1],
                z=nodes[:, 2],
                u=arrow_vectors[:, 0],
                v=arrow_vectors[:, 1],
                w=arrow_vectors[:, 2],
                sizemode="absolute",
                sizeref=1.0,
                anchor="tail",
                colorscale=[[0, options.arrow_color], [1, options.arrow_color]],
                showscale=False,
                name="Force direction",
            ),
        ]
    )

    _apply_layout(fig, nodes[None, :, :], title or options.title, options)
    return fig


def get_force_animation_plotly(
    nodes_by_frame: typing.Mapping[str, np.ndarray],
    forces_by_frame: typing.Mapping[str, np.ndarray],
    triangles: np.ndarray,
    options: VisuoShellVisualizationOptions | None = None,
) -> go.Figure:
    """Return an interactive Plotly animation of VisuoShell force frames."""
    options = options or VisuoShellVisualizationOptions()
    frame_names = sorted(nodes_by_frame)
    if not frame_names:
        raise ValueError("nodes_by_frame must contain at least one frame")
    if options.plot_step < 1:
        raise ValueError("plot_step must be at least 1")

    frame_names = frame_names[:: options.plot_step]
    if frame_names[-1] != sorted(nodes_by_frame)[-1]:
        frame_names.append(sorted(nodes_by_frame)[-1])

    triangles = np.asarray(triangles, dtype=np.int64)
    nodes_all = np.asarray([nodes_by_frame[name] for name in frame_names], dtype=np.float64)
    first_name = frame_names[0]
    fig = get_force_figure(
        nodes_by_frame[first_name],
        forces_by_frame[first_name],
        triangles,
        title=first_name,
        options=options,
    )

    frames = []
    for frame_idx, frame_name in enumerate(frame_names):
        nodes, forces, triangles = _validate_inputs(
            nodes_by_frame[frame_name],
            forces_by_frame[frame_name],
            triangles,
        )
        force_magnitude = np.linalg.norm(forces, axis=1)
        arrow_vectors = _arrow_vectors(forces, options.arrow_scale)
        edge_x, edge_y, edge_z = _edge_lines(nodes, triangles)
        i, j, k = _triangle_indices(triangles)

        frames.append(
            go.Frame(
                name=str(frame_idx),
                data=[
                    go.Mesh3d(
                        x=nodes[:, 0],
                        y=nodes[:, 1],
                        z=nodes[:, 2],
                        i=i,
                        j=j,
                        k=k,
                        intensity=force_magnitude,
                    ),
                    go.Scatter3d(x=edge_x, y=edge_y, z=edge_z),
                    go.Scatter3d(
                        x=nodes[:, 0],
                        y=nodes[:, 1],
                        z=nodes[:, 2],
                        marker=dict(color=force_magnitude),
                    ),
                    go.Cone(
                        x=nodes[:, 0],
                        y=nodes[:, 1],
                        z=nodes[:, 2],
                        u=arrow_vectors[:, 0],
                        v=arrow_vectors[:, 1],
                        w=arrow_vectors[:, 2],
                    ),
                ],
                layout=go.Layout(title=f"{options.title}: {frame_name}"),
            )
        )

    fig.frames = frames
    _apply_layout(fig, nodes_all, f"{options.title}: {first_name}", options)
    fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {
                                "frame": {"duration": 150, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 70},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top",
            }
        ],
        sliders=[
            {
                "steps": [
                    {
                        "args": [
                            [str(i_frame)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                            },
                        ],
                        "label": frame_name,
                        "method": "animate",
                    }
                    for i_frame, frame_name in enumerate(frame_names)
                ],
                "transition": {"duration": 0},
                "x": 0.1,
                "y": 0,
                "currentvalue": {"prefix": "Frame: "},
            }
        ],
    )
    return fig


def visualize(
    nodes_by_frame: typing.Mapping[str, np.ndarray],
    forces_by_frame: typing.Mapping[str, np.ndarray],
    triangles: np.ndarray,
    options: VisuoShellVisualizationOptions | None = None,
) -> go.Figure:
    """Mirror VisuoShell's visualize entry point with an interactive Plotly figure."""
    fig = get_force_animation_plotly(nodes_by_frame, forces_by_frame, triangles, options)
    fig.show()
    return fig


def get_force_polydata_pyvista(
    nodes: np.ndarray,
    forces: np.ndarray,
    triangles: np.ndarray,
):
    """Return a PyVista PolyData frame using VisuoShell's original mesh style."""
    pv = _import_pyvista()
    nodes, forces, triangles = _validate_inputs(nodes, forces, triangles)

    mesh = pv.PolyData(nodes, _pyvista_faces(triangles))
    mesh["force_magnitude"] = np.linalg.norm(forces, axis=1)
    return mesh


def visualize_pyvista(
    nodes_by_frame: typing.Mapping[str, np.ndarray],
    forces_by_frame: typing.Mapping[str, np.ndarray],
    triangles: np.ndarray,
    options: VisuoShellVisualizationOptions | None = None,
):
    """Show frames with the original VisuoShell PyVista rendering style.

    Set ``pyvista_image_dir`` or ``pyvista_gif_path`` in the options to save
    off-screen PNG frames and/or an animated GIF.
    """
    pv = _import_pyvista()
    options = options or VisuoShellVisualizationOptions()
    frame_names = sorted(nodes_by_frame)
    if not frame_names:
        raise ValueError("nodes_by_frame must contain at least one frame")
    if options.plot_step < 1:
        raise ValueError("plot_step must be at least 1")

    selected_frame_names = frame_names[:: options.plot_step]
    if frame_names[-1] != selected_frame_names[-1]:
        selected_frame_names.append(frame_names[-1])

    image_dir = (
        pathlib.Path(options.pyvista_image_dir)
        if options.pyvista_image_dir is not None
        else None
    )
    gif_path = (
        pathlib.Path(options.pyvista_gif_path)
        if options.pyvista_gif_path is not None
        else None
    )
    if image_dir is None and gif_path is not None:
        image_dir = gif_path.with_suffix("")
    if image_dir is not None:
        image_dir.mkdir(parents=True, exist_ok=True)

    saved_images: list[pathlib.Path] = []
    off_screen = options.pyvista_off_screen or image_dir is not None or gif_path is not None
    last_plotter = None
    for frame_name in selected_frame_names:
        nodes, forces, frame_triangles = _validate_inputs(
            nodes_by_frame[frame_name],
            forces_by_frame[frame_name],
            triangles,
        )
        mesh = get_force_polydata_pyvista(nodes, forces, frame_triangles)
        arrow_vectors = _arrow_vectors(forces, options.arrow_scale)

        plotter = pv.Plotter(off_screen=off_screen)
        plotter.add_mesh(
            mesh,
            scalars="force_magnitude",
            cmap=options.colorscale.lower(),
            show_edges=options.pyvista_show_edges,
        )
        plotter.add_arrows(nodes, arrow_vectors, mag=1.0)
        plotter.add_title(frame_name)
        screenshot_path = (
            image_dir / f"{pathlib.Path(frame_name).stem}.png"
            if image_dir is not None
            else None
        )
        plotter.show(screenshot=str(screenshot_path) if screenshot_path else None)
        last_plotter = plotter
        if screenshot_path is not None:
            saved_images.append(screenshot_path)

        if options.pyvista_pause > 0:
            time.sleep(options.pyvista_pause)

        if off_screen:
            plotter.close()

    if gif_path is not None:
        _save_gif(saved_images, gif_path, options.pyvista_gif_duration_ms)

    return last_plotter


def _validate_inputs(
    nodes: np.ndarray,
    forces: np.ndarray,
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes = np.asarray(nodes, dtype=np.float64)
    forces = np.asarray(forces, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)

    if nodes.ndim != 2 or nodes.shape[1] != 3:
        raise ValueError("nodes must have shape (N, 3)")
    if forces.shape != nodes.shape:
        raise ValueError("forces must have the same shape as nodes")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (M, 3)")
    return nodes, forces, triangles


def _arrow_vectors(forces: np.ndarray, scale: float) -> np.ndarray:
    norms = np.linalg.norm(forces, axis=1, keepdims=True)
    directions = np.divide(forces, norms, out=np.zeros_like(forces), where=norms > 0)
    return scale * directions


def _triangle_indices(triangles: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    return (
        triangles[:, 0].astype(int).tolist(),
        triangles[:, 1].astype(int).tolist(),
        triangles[:, 2].astype(int).tolist(),
    )


def _edge_lines(nodes: np.ndarray, triangles: np.ndarray) -> tuple[list[float], list[float], list[float]]:
    edges = set()
    for tri in triangles:
        edges.add(tuple(sorted((int(tri[0]), int(tri[1])))))
        edges.add(tuple(sorted((int(tri[1]), int(tri[2])))))
        edges.add(tuple(sorted((int(tri[0]), int(tri[2])))))

    edge_x: list[float] = []
    edge_y: list[float] = []
    edge_z: list[float] = []
    for n0, n1 in sorted(edges):
        edge_x += [nodes[n0, 0], nodes[n1, 0], None]
        edge_y += [nodes[n0, 1], nodes[n1, 1], None]
        edge_z += [nodes[n0, 2], nodes[n1, 2], None]
    return edge_x, edge_y, edge_z


def _pyvista_faces(triangles: np.ndarray) -> np.ndarray:
    triangles = np.asarray(triangles, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape (M, 3)")
    return np.column_stack(
        [np.full(triangles.shape[0], 3, dtype=np.int64), triangles]
    ).reshape(-1)


def _import_pyvista():
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError(
            "PyVista rendering requires the optional 'pyvista' package. "
            "Install it to use visualize_pyvista()."
        ) from exc
    return pv


def _save_gif(
    image_paths: typing.Sequence[pathlib.Path],
    gif_path: pathlib.Path,
    duration_ms: int,
) -> None:
    if not image_paths:
        raise ValueError("no PyVista screenshots were generated for GIF output")

    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "GIF output requires the optional 'Pillow' package. "
            "Install it to use pyvista_gif_path."
        ) from exc

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames = [Image.open(path) for path in image_paths]
    try:
        frames[0].save(
            gif_path,
            format="GIF",
            append_images=frames[1:],
            save_all=True,
            duration=duration_ms,
            loop=0,
        )
    finally:
        for frame in frames:
            frame.close()


def _apply_layout(
    fig: go.Figure,
    nodes_all: np.ndarray,
    title: str,
    options: VisuoShellVisualizationOptions,
) -> None:
    elev, azim = options.camera_view
    radius = 2.0
    camera_eye = dict(
        x=radius * math.cos(math.radians(elev)) * math.cos(math.radians(azim)),
        y=radius * math.cos(math.radians(elev)) * math.sin(math.radians(azim)),
        z=radius * math.sin(math.radians(elev)),
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X Position",
            yaxis_title="Y Position",
            zaxis_title="Z Position",
            camera=dict(eye=camera_eye),
            aspectmode="data",
            xaxis=dict(range=options.x_lim or _axis_range(nodes_all[:, :, 0])),
            yaxis=dict(range=options.y_lim or _axis_range(nodes_all[:, :, 1])),
            zaxis=dict(range=options.z_lim or _axis_range(nodes_all[:, :, 2])),
        ),
    )


def _axis_range(values: np.ndarray) -> list[float]:
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if np.isclose(min_value, max_value):
        pad = 0.5 if np.isclose(min_value, 0.0) else abs(min_value) * 0.1
    else:
        pad = 0.05 * (max_value - min_value)
    return [min_value - pad, max_value + pad]
