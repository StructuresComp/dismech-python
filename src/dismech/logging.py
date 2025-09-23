import numpy as np

def logDataForRendering(
    dofs,
    time_array,
    softRobot,
    Nsteps,
    static_sim,
    mapNodetoDOF,
    log_step=1
):
    dof_with_time = np.hstack([time_array, dofs])
    n_rod_nodes = len(np.unique(softRobot.rod_edges))
    n_faces = len(softRobot.face_nodes_shell)

    if static_sim:
        rod_data = np.zeros((n_rod_nodes, 4))
        for j in range(n_rod_nodes):
            rod_data[j, 0] = dof_with_time[0, -1]
            rod_data[j, 1:] = dof_with_time[-1, 1 + mapNodetoDOF(j)]

        shell_data = np.zeros((3 * n_faces, 3))
        for j in range(n_faces):
            n1, n2, n3 = softRobot.face_nodes_shell[j]
            shell_data[3*j:3*j+3, :] = np.vstack([
                dof_with_time[-1, 1 + mapNodetoDOF(n1)],
                dof_with_time[-1, 1 + mapNodetoDOF(n2)],
                dof_with_time[-1, 1 + mapNodetoDOF(n3)]
            ])

        np.savetxt('rawDataRod.txt', rod_data, fmt='%.6e')
        np.savetxt('rawDataShell.txt', shell_data, fmt='%.6e')
        return rod_data, shell_data

    # indices to log
    frame_indices = list(range(0, Nsteps, log_step))
    n_frames = len(frame_indices)

    rod_data = np.zeros((n_rod_nodes * n_frames, 4))
    shell_data = np.zeros((3 * n_faces * n_frames, 3))

    for k, i in enumerate(frame_indices):
        # rod data
        for j in range(n_rod_nodes):
            rod_data[k * n_rod_nodes + j, 0] = dof_with_time[i, 0]
            rod_data[k * n_rod_nodes + j, 1:] = dof_with_time[i, 1 + mapNodetoDOF(j)]

        # shell data
        for j in range(n_faces):
            n1, n2, n3 = softRobot.face_nodes_shell[j]
            idx = k * 3 * n_faces + 3 * j
            shell_data[idx:idx+3, :] = np.vstack([
                dof_with_time[i, 1 + mapNodetoDOF(n1)],
                dof_with_time[i, 1 + mapNodetoDOF(n2)],
                dof_with_time[i, 1 + mapNodetoDOF(n3)]
            ])

    np.savetxt('rawDataRod.txt', rod_data, fmt='%.6e')
    np.savetxt('rawDataShell.txt', shell_data, fmt='%.6e')

    return rod_data, shell_data


def export_rod_shell_data(robot, rod_file='rawDataRod.txt', shell_file='rawDataShell.txt',
                          rod_js='rodData.js', shell_js='shellData.js',
                          rod_radius=0.1, scaleFactor=100):
    """
    Export rod and shell data to .js files for visualization.

    Parameters
    ----------
    robot : object
        Object with attributes `rod_edges` and `face_nodes_shell`.
    rod_file : str
        Path to raw rod data (.txt).
    shell_file : str
        Path to raw shell data (.txt).
    rod_js : str
        Output JS file path for rod data.
    shell_js : str
        Output JS file path for shell data.
    rod_radius : float
        Radius of rods.
    scaleFactor : float
        Scale factor for coordinates.
    """

    # === Load rod data ===
    df = np.loadtxt(rod_file)
    n_rod_nodes = len(np.unique(robot.rod_edges))
    n_Tri = len(robot.face_nodes_shell)

    # Write rod data
    with open(rod_js, 'w') as fileID:
        fileID.write(f'nNodes = {n_rod_nodes};\n')
        fileID.write(f'rodRadius = {rod_radius};\n')
        fileID.write('nodesRod = [\n')

        for row in df:
            t, x, y, z = row
            x, y, z = x * scaleFactor, y * scaleFactor, z * scaleFactor
            fileID.write(f'{t}, 1, {x}, {y}, {z},\n')

        fileID.write(']\n;\n')

    # === Load shell data ===
    ds = np.loadtxt(shell_file)

    # Write shell data
    with open(shell_js, 'w') as shell_fileID:
        shell_fileID.write(f'nTri = {n_Tri},\n')
        shell_fileID.write('nodes = [\n')

        for row in ds:
            x, y, z = row * scaleFactor
            shell_fileID.write(f'{x}, {y}, {z},\n')

        shell_fileID.write('];\n')

def export_parachute_data(dof_with_time, elStretchRod, MultiRod,
                          shell_file='rawDataShell.txt',
                          rod_js='rodData.js', shell_js='shellData.js',
                          scaleFactor=10, rod_radius=0.32):
    """
    Export rod + shell data for parachute visualization into JS files.

    Parameters
    ----------
    dof_with_time : np.ndarray
        DOF with time array (shape: (3*n_nodes+1, steps)).
    elStretchRod : np.ndarray
        Rod element connectivity matrix.
    MultiRod : object
        Object with attributes: n_nodes, n_rod_nodes, n_faces.
    shell_file : str
        Path to raw shell node data (.txt).
    rod_js : str
        Output JS file path for rod data.
    shell_js : str
        Output JS file path for shell data.
    scaleFactor : float
        Factor to scale node coordinates.
    rod_radius : float
        Radius of rod.
    """

    # === Rod data ===
    node_data = scaleFactor * dof_with_time[:, 0:3*MultiRod.n_nodes]
    # connectivity_matrix = elStretchRod - np.ones_like(elStretchRod, dtype=int)
    connectivity_matrix = elStretchRod
    n_Tri = MultiRod.face_nodes_shell.shape[0]

    with open(rod_js, 'w') as f:
        f.write('var rodData = {\n')
        # Uncomment if you want to include:
        # f.write(f'  nNodes : {MultiRod.n_rod_nodes},\n')
        # f.write(f'  rodRadius : {rod_radius},\n')

        # Node positions
        f.write('  nodePositions : [\n')
        for row in node_data:
            row_str = ', '.join(map(str, row))
            f.write(f'    [{row_str}],\n')
        f.write('  ],\n')

        # Connectivity
        f.write('  connectivity : [\n')
        for row in connectivity_matrix:
            row_str = ', '.join(map(str, row))
            f.write(f'    [{row_str}],\n')
        f.write('  ]\n')

        f.write('};\n')

    # === Shell data ===
    ds = np.loadtxt(shell_file)

    with open(shell_js, 'w') as f:
        f.write('var shellData = {\n')
        f.write(f'  nTri : {n_Tri},\n')
        f.write('  nodes : [\n')
        for row in ds:
            x, y, z = row * scaleFactor
            f.write(f'    {x}, {y}, {z},\n')
        f.write('  ]\n')
        f.write('};\n')

