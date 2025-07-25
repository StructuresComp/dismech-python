import numpy as np


def compute_gravity_forces(robot):
    fg = np.zeros_like(robot.mass_matrix)
    mass_diag = robot.mass_matrix
    fg[robot.node_dof_indices] = mass_diag[robot.node_dof_indices] * robot.env.g
    return fg
