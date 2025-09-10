import numpy as np

def logDataForRendering(dofs, time_array, softRobot, Nsteps, static_sim, mapNodetoDOF):
    dof_with_time = np.hstack([time_array, dofs])
    print(np.shape(dof_with_time), np.shape(time_array), np.shape(dofs))
    n_rod_nodes = len(np.unique(softRobot.rod_edges))
    n_faces = len(softRobot.face_nodes_shell)
    print(np.shape(softRobot.face_nodes_shell))
    print(n_faces)

    if static_sim:
        rod_data = np.zeros((n_rod_nodes, 4))
        for j in range(n_rod_nodes):
            rod_data[j, 0] = dof_with_time[0, -1]
            rod_data[j, 1:] = dof_with_time[-1, 1 + mapNodetoDOF(j)]

        shell_data = np.zeros((3 * n_faces, 3))
        for j in range(n_faces):
            n1 = softRobot.face_nodes_shell[j, 0]
            n2 = softRobot.face_nodes_shell[j, 1]
            n3 = softRobot.face_nodes_shell[j, 2]
            shell_data[3*j:3*j+3, :] = np.vstack([
                dof_with_time[-1, 1 + mapNodetoDOF(n1)],
                dof_with_time[-1, 1 + mapNodetoDOF(n2)],
                dof_with_time[-1, 1 + mapNodetoDOF(n3)]
            ])

        np.savetxt('rawDataRod.txt', rod_data, fmt='%.6e')
        np.savetxt('rawDataShell.txt', shell_data, fmt='%.6e')
        return rod_data, shell_data

    # For dynamic case
    rod_data = np.zeros((n_rod_nodes * Nsteps, 4))
    for i in range(Nsteps-1):
        for j in range(n_rod_nodes):
            rod_data[i * n_rod_nodes + j, 0] = dof_with_time[i, 0]
            rod_data[i * n_rod_nodes + j, 1:] = dof_with_time[i, 1 + mapNodetoDOF(j)]

    shell_data = np.zeros((3 * n_faces * Nsteps, 3))
    for i in range(Nsteps-1):
        for j in range(n_faces):
            n1 = softRobot.face_nodes_shell[j, 0]
            n2 = softRobot.face_nodes_shell[j, 1]
            n3 = softRobot.face_nodes_shell[j, 2]
            idx = i * 3 * n_faces + 3 * j
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
