import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
import plotly.graph_objects as go

from dataclasses import dataclass


@dataclass
class AnimationOptions:
    free_node_color: str = '#1f78b4'
    fixed_node_color: str = '#e31a1c'
    edge_color: str = '#333333'
    face_color: tuple = (166/255, 206/255, 227/255, 0.7)
    title: str = "Dismech Simulation"
    x_lim: tuple = None
    y_lim: tuple = None
    z_lim: tuple = None
    camera_view: tuple = (30, 45)  # (elevation, azimuth) angles in degrees
    follow_data: bool = False


def get_animation(robot, t, qs, options: AnimationOptions):
    n_frames = qs.shape[0]
    n_nodes = len(robot.node_dof_indices)

    # Precompute nodes
    node_dof_list = [robot.map_node_to_dof(i) for i in range(n_nodes)]
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

def rgba_to_str(color):
    """Convert an RGBA tuple to a Plotly rgba string."""
    if isinstance(color, tuple) and len(color) == 4:
        r, g, b, a = color
        return f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {a})"
    return color

def get_interactive_animation_plotly(robot, t, qs, options: AnimationOptions):
    n_frames = qs.shape[0]
    n_nodes = len(robot.node_dof_indices)
    
    # Precompute nodes for each frame: nodes_all[frame][node] gives a (x,y,z)
    node_dof_list = [robot.map_node_to_dof(i) for i in range(n_nodes)]
    nodes_all = np.array([[q[dofs] for dofs in node_dof_list] for q in qs])
    
    # Identify fixed nodes based on fixed dofs
    fixed_set = set(robot.fixed_dof)
    node_is_fixed = np.array([all(d in fixed_set for d in dofs)
                              for dofs in node_dof_list])
    
    # Precompute indices for free and fixed nodes
    free_indices = np.where(~node_is_fixed)[0]
    fixed_indices = np.where(node_is_fixed)[0]
    
    # Create an initial frame (frame 0)
    nodes0 = nodes_all[0]
    free_nodes0 = nodes0[free_indices]
    fixed_nodes0 = nodes0[fixed_indices]
    
    # Scatter for free nodes
    free_scatter = go.Scatter3d(
        x=free_nodes0[:, 0],
        y=free_nodes0[:, 1],
        z=free_nodes0[:, 2],
        mode='markers',
        marker=dict(size=5, color=options.free_node_color),
        name='Free Nodes'
    )

    
    # Scatter for fixed nodes
    fixed_scatter = go.Scatter3d(
        x=fixed_nodes0[:, 0],
        y=fixed_nodes0[:, 1],
        z=fixed_nodes0[:, 2],
        mode='markers',
        marker=dict(size=5, color=options.fixed_node_color),
        name='Fixed Nodes'
    )
    
    # Create one trace for all edges (using None to separate segments)
    edge_x, edge_y, edge_z = [], [], []
    for (i, j) in robot.edges:
        edge_x += [nodes0[i][0], nodes0[j][0], None]
        edge_y += [nodes0[i][1], nodes0[j][1], None]
        edge_z += [nodes0[i][2], nodes0[j][2], None]
    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode='lines',
        line=dict(color=options.edge_color, width=2),
        name='Edges'
    )
    
    # Use a Mesh3d trace for triangular faces
    # Note: Mesh3d uses a single set of vertices, so the connectivity (I, J, K) remains the same.
    I, J, K = [], [], []
    for (i, j, k) in robot.face_nodes_shell:
        I.append(i)
        J.append(j)
        K.append(k)
    face_trace = go.Mesh3d(
        x=nodes0[:, 0],
        y=nodes0[:, 1],
        z=nodes0[:, 2],
        i=I,
        j=J,
        k=K,
        color=rgba_to_str(options.face_color),
        opacity=options.face_color[3] if len(options.face_color) >= 4 else 1.0,
        name='Faces',
        showscale=False
    )
    
    # Set up camera using the provided camera_view (conversion from elevation/azimuth)
    elev, azim = options.camera_view
    r = 2  # Radius for camera distance (adjust as needed)
    camera_eye = dict(
        x=r * math.cos(math.radians(elev)) * math.cos(math.radians(azim)),
        y=r * math.cos(math.radians(elev)) * math.sin(math.radians(azim)),
        z=r * math.sin(math.radians(elev))
    )
    
    # Build animation frames
    frames = []
    for frame_idx in range(n_frames):
        nodes = nodes_all[frame_idx]
        free_nodes_frame = nodes[free_indices]
        fixed_nodes_frame = nodes[fixed_indices]
        
        # Update edges for current frame
        edge_x_frame, edge_y_frame, edge_z_frame = [], [], []
        for (i, j) in robot.edges:
            edge_x_frame += [nodes[i][0], nodes[j][0], None]
            edge_y_frame += [nodes[i][1], nodes[j][1], None]
            edge_z_frame += [nodes[i][2], nodes[j][2], None]
        
        frame_data = [
            dict(type='scatter3d',
                 name='Free Nodes',
                 x=free_nodes_frame[:, 0],
                 y=free_nodes_frame[:, 1],
                 z=free_nodes_frame[:, 2]),
            dict(type='scatter3d',
                 name='Fixed Nodes',
                 x=fixed_nodes_frame[:, 0],
                 y=fixed_nodes_frame[:, 1],
                 z=fixed_nodes_frame[:, 2]),
            dict(type='scatter3d',
                 name='Edges',
                 x=edge_x_frame,
                 y=edge_y_frame,
                 z=edge_z_frame),
            dict(type='mesh3d',
                 name='Faces',
                 x=nodes[:, 0],
                 y=nodes[:, 1],
                 z=nodes[:, 2],
                 i=I,
                 j=J,
                 k=K)
        ]
        frames.append(dict(name=str(frame_idx),
                           data=frame_data,
                           layout=dict(annotations=[dict(
                               text=f"Time: {t[frame_idx]:.2f}s (Step: {frame_idx+1}/{n_frames})",
                               showarrow=False,
                               x=0.05, y=0.95, xref="paper", yref="paper"
                           )])))
    
    # Define layout with slider and play/pause buttons
    layout = go.Layout(
        title=options.title,
        scene=dict(
            xaxis_title='X Position',
            yaxis_title='Y Position',
            zaxis_title='Z Position',
            camera=dict(eye=camera_eye),
            xaxis=dict(range=options.x_lim if options.x_lim is not None else [nodes_all[:, :, 0].min(), nodes_all[:, :, 0].max()]),
            yaxis=dict(range=options.y_lim if options.y_lim is not None else [nodes_all[:, :, 1].min(), nodes_all[:, :, 1].max()]),
            zaxis=dict(range=options.z_lim if options.z_lim is not None else [nodes_all[:, :, 2].min(), nodes_all[:, :, 2].max()])
        ),
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 50, "redraw": True}, "fromcurrent": True}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }],
        sliders=[{
            "steps": [
                {
                    "args": [[str(frame_idx)], {"frame": {"duration": 50, "redraw": True}, "mode": "immediate"}],
                    "label": str(frame_idx),
                    "method": "animate"
                }
                for frame_idx in range(n_frames)
            ],
            "transition": {"duration": 0},
            "x": 0.1,
            "y": 0,
            "currentvalue": {"prefix": "Frame: "}
        }]
    )

    data = [free_scatter, fixed_scatter, edge_trace, face_trace]

    x_min = nodes_all[:, :, 0].min()
    x_max = nodes_all[:, :, 0].max()
    y_min = nodes_all[:, :, 1].min()
    y_max = nodes_all[:, :, 1].max()

    if hasattr(robot.env, 'ground_z'):
        floor = go.Surface(
            z=np.full((2, 2), robot.env.ground_z),  # Flat z plane
            x=np.array([[x_min-0.5, x_max+0.5], [x_min-0.5, x_max+0.5]]),
            y=np.array([[y_min-0.5, y_min-0.5], [y_max+0.5, y_max+0.5]]),
            showscale=False,
            colorscale=[[0, 'lightgray'], [1, 'lightgray']],  # Flat color
            opacity=0.9
        )
        data.append(floor)
    
    
    # Create the figure with initial data and frames
    fig = go.Figure(
        data=data,
        layout=layout,
        frames=frames
    )
    
    return fig