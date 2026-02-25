import numpy as np


def add_point_forces(robot, node_indices, force_vectors):
    fpt = np.zeros_like(robot.state.q)

    # Ensure numpy array
    force_vectors = np.asarray(force_vectors)

    dof_indices = robot.map_node_to_dof(node_indices) # returns 1D array of dof indices for node list
    
    fpt[dof_indices] = force_vectors.reshape(-1) # reshape to 1D if needed
    # print(f"Adding point forces at nodes {node_indices} with force vectors {force_vectors}")
    return fpt
