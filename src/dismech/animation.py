import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from dataclasses import dataclass


@dataclass
class AnimationOptions:
    free_node_color: str = '#1f78b4'
    fixed_node_color: str = '#e31a1c'
    edge_color: str = '#333333'
    face_color: tuple = (166/255, 206/255, 227/255, 0.4)
    title: str = "Dismech Simulation"
    x_lim: tuple = None
    y_lim: tuple = None
    z_lim: tuple = None
    camera_view: tuple = (30, 45)  # (elevation, azimuth) angles in degrees
    follow_data: bool = False


def get_animation(robot, t, qs, options: AnimationOptions):
    n_frames = qs.shape[0]
    n_nodes = len(robot.node_dof_indices)

    # Precompute mapping from node index to its dof indices.
    node_dof_list = [robot.map_node_to_dof(i) for i in range(n_nodes)]

    # Pre-compute node positions for each frame; shape (n_frames, n_nodes, 3)
    nodes_all = np.array([[q[dofs] for dofs in node_dof_list] for q in qs])

    # Determine which nodes are fixed
    fixed_set = set(robot.fixed_dof)
    node_is_fixed = np.array([all(d in fixed_set for d in dofs)
                             for dofs in node_dof_list])

    # Create figure and 3D axis.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title(options.title)

    ax.view_init(*options.camera_view)

    scatter_free = ax.scatter([], [], [], color=options.free_node_color, s=15)
    scatter_fixed = ax.scatter(
        [], [], [], color=options.fixed_node_color, s=15)

    edge_collection = Line3DCollection(
        [[(0, 0, 0), (0, 0, 0)]], colors=options.edge_color, linewidths=1.5)
    ax.add_collection3d(edge_collection)
    face_collection = Poly3DCollection(
        [[(0, 0, 0), (0, 0, 0), (0, 0, 0)]], facecolors=options.face_color, edgecolors='none')
    ax.add_collection3d(face_collection)

    time_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    if not options.follow_data:
        ax.set_xlim(
            *((options.x_lim or (nodes_all[:, :, 0].min(), nodes_all[:, :, 0].max()))))
        ax.set_ylim(
            *((options.y_lim or (nodes_all[:, :, 1].min(), nodes_all[:, :, 1].max()))))
        ax.set_zlim(
            *((options.z_lim or (nodes_all[:, :, 2].min(), nodes_all[:, :, 2].max()))))

    def init():
        scatter_free._offsets3d = ([], [], [])
        scatter_fixed._offsets3d = ([], [], [])
        edge_collection.set_segments([])
        face_collection.set_verts([])
        time_text.set_text("")
        return scatter_free, scatter_fixed, edge_collection, face_collection, time_text

    def update(frame):
        nodes = nodes_all[frame]
        scatter_free._offsets3d = (
            nodes[~node_is_fixed, 0], nodes[~node_is_fixed, 1], nodes[~node_is_fixed, 2])
        scatter_fixed._offsets3d = (
            nodes[node_is_fixed, 0], nodes[node_is_fixed, 1], nodes[node_is_fixed, 2])

        edge_collection.set_segments(
            [[nodes[i], nodes[j]] for i, j in robot.edges])
        face_collection.set_verts(
            [[nodes[i], nodes[j], nodes[k]] for i, j, k in robot.face_nodes_shell])

        if options.follow_data:
            ax.set_xlim(nodes[:, 0].min(), nodes[:, 0].max())
            ax.set_ylim(nodes[:, 1].min(), nodes[:, 1].max())
            ax.set_zlim(nodes[:, 2].min(), nodes[:, 2].max())

        time_text.set_text(
            f"Time: {t[frame]:.2f}s (Step: {frame+1}/{n_frames})")
        return scatter_free, scatter_fixed, edge_collection, face_collection, time_text

    ani = FuncAnimation(fig, update, frames=n_frames,
                        init_func=init, interval=50, blit=False)
    return ani
