import numpy as np
from scipy.linalg import solve, LinAlgError

from dismech import SoftRobot, fs


class TimeStepper:
    def __init__(self, robot: SoftRobot, fixed_nodes):
        self.robot = robot.initialize(fixed_nodes)
        self.epsilon = 1e-8  # Regularization parameter
        self.min_force = 1e-8  # Threshold for negligible forces

    def step(self, robot: SoftRobot = None) -> SoftRobot:
        robot = robot or self.robot
        params = robot.sim_params

        # Initialize iteration variables
        q = robot.q0.copy()
        alpha = 1.0
        iteration = 1
        err_history = []

        while True:
            # Compute forces and Jacobians
            forces, jacobian = self._compute_forces_and_jacobian(robot, q)

            # Handle free DOF components
            free_idx = robot.free_dof
            f_free = -forces[free_idx]
            j_free = -jacobian[np.ix_(free_idx, free_idx)]

            # Regularized matrix solver
            dq_free = self._safe_solve(j_free, f_free)

            # Adaptive damping and update
            alpha = self._adaptive_damping(alpha, iteration)
            q[free_idx] -= alpha * dq_free

            # Convergence checks
            err = np.linalg.norm(f_free)
            err_history.append(err)

            # Compute displacement increment
            dq = np.zeros(robot.n_dof)
            dq[free_idx] = alpha * dq_free

            # Check all convergence criteria
            disp_converged = np.max(np.abs(dq[1:robot.end_node_dof_index])) / robot.sim_params.dt < robot.sim_params.dtol
            force_converged = err < params.tol
            relative_converged = err < err_history[0] * params.ftol
            iteration_limit = iteration >= params.max_iter

            if any([force_converged, relative_converged, disp_converged, iteration_limit]):
                break

            iteration += 1

        # Final update and return
        return self._finalize_update(robot, q)

    def _compute_forces_and_jacobian(self, robot, q):
        forces = np.zeros(robot.n_dof)
        jacobian = np.zeros((robot.n_dof, robot.n_dof))

        # Compute reference frames and material directors
        a1_iter, a2_iter = robot.compute_time_parallel(robot.a1, robot.q0, q)
        theta = robot.get_theta(q)
        m1, m2 = robot.compute_material_directors(a1_iter, a2_iter, theta)

        # Add stretch spring contributions
        if robot.stretch_springs:
            Fs, Js = fs.get_fs_js_vectorized(robot, q)
            forces += Fs
            jacobian += Js

        # Add bend/twist contributions
        if robot.bend_twist_springs:
            Fb, Jb = fs.get_fb_jb_vectorized(robot, q, m1, m2)
            forces += Fb
            jacobian += Jb

        # Add gravity forces
        if "gravity" in robot.env.ext_force_list:
            forces += self._compute_gravity_forces(robot)

        return forces, jacobian

    def _safe_solve(self, J, F):
        """Regularized matrix solver with fallback strategies"""
        if np.linalg.norm(F) < self.min_force:
            return np.zeros_like(F)

        try:
            # Add regularization to Jacobian
            J_reg = J + self.epsilon * np.eye(J.shape[0])
            return solve(J_reg, F, assume_a='pos')
        except LinAlgError:
            # Fallback to least squares solution
            return np.linalg.lstsq(J_reg, F, rcond=None)[0]

    def _adaptive_damping(self, alpha, iteration):
        if iteration < 10:
            return 1.0

        return max(alpha * 0.9, 0.1)

    def _compute_gravity_forces(self, robot):
        fg = np.zeros_like(robot.mass_matrix[0])
        mass_diag = np.diag(robot.mass_matrix)
        fg[robot.node_dof_indices] = mass_diag[robot.node_dof_indices] * robot.env.g
        return fg

    def _finalize_update(self, robot: SoftRobot, q):
        u = (q - robot.q0) / robot.sim_params.dt
        a1, a2 = robot.compute_time_parallel(robot.a1, robot.q0, q)

        return robot.update(
            q, u, a1, a2,
            *robot.compute_material_directors(a1, a2, robot.get_theta(q)),
            robot.compute_reference_twist(
                robot.bend_twist_springs, a1, robot.compute_tangent(q), robot.ref_twist)
        )
