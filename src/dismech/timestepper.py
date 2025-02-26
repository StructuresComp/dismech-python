import copy

import numpy as np
from scipy.linalg import solve, LinAlgError

from dismech import SoftRobot, fs


class TimeStepper:
    def __init__(self, robot: SoftRobot, fixed_nodes):
        self.robot = robot.initialize(fixed_nodes)
        self.epsilon = 1e-8  # Regularization parameter
        self.min_force = 1e-8  # Threshold for negligible forces

    def step(self, robot: SoftRobot = None, debug=False) -> SoftRobot:
        robot = robot or self.robot
        params = robot.sim_params
        free_idx = robot.free_dof

        # Initialize iteration variables
        q0 = copy.deepcopy(robot.q)
        q = copy.deepcopy(robot.q)

        alpha = 1.0
        iteration = 1
        err_history = []

        while True:
            # Compute forces and Jacobians
            forces, jacobian = self._compute_forces_and_jacobian(robot, q, q0)

            # Inertial force vs equilibrium
            if params.static_sim:
                forces = -forces
                jacobian = -jacobian
            else:
                forces = (robot.mass_matrix / params.dt) @ \
                    ((q - q0) / params.dt - robot.u) - forces
                jacobian = robot.mass_matrix / params.dt ** 2 - jacobian

            # Handle free DOF components
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
            disp_converged = np.max(np.abs(
                dq[1:robot.end_node_dof_index])) / robot.sim_params.dt < robot.sim_params.dtol
            force_converged = err < params.tol
            relative_converged = err < err_history[0] * params.ftol
            iteration_limit = iteration >= params.max_iter

            if any([force_converged, relative_converged, disp_converged, iteration_limit]):
                if iteration_limit:
                    raise ValueError
                break
            if debug:
                print("iter: {}, error: {:.3f}".format(iteration, err))

            iteration += 1

        # Final update and return
        self.robot = self._finalize_update(robot, q)
        return self.robot

    def _compute_forces_and_jacobian(self, robot: SoftRobot, q, q0):
        forces = np.zeros(robot.n_dof)
        jacobian = np.zeros((robot.n_dof, robot.n_dof))

        # Compute reference frames and material directors
        a1_iter, a2_iter = robot.compute_time_parallel(robot.a1, robot.q, q)
        theta = robot.get_theta(q)
        m1, m2 = robot.compute_material_directors(a1_iter, a2_iter, theta)
        ref_twist = robot.compute_reference_twist(
            robot.bend_twist_springs, a1_iter, robot.compute_tangent(q), robot.ref_twist)

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
            if not robot.sim_params.two_d_sim:
                Ft, Jt = fs.get_ft_jt_vectorized(robot, q, ref_twist)
                forces += Ft
                jacobian += Jt

        if robot.hinge_springs:
            Fb, Jb = fs.get_fb_jb_shell_vectorized(robot, q)
            forces += Fb
            jacobian += Jb

        # Add gravity forces
        if "gravity" in robot.env.ext_force_list:
            forces += self._compute_gravity_forces(robot)
        if "aerodynamics" in robot.env.ext_force_list:
            F, J = self._compute_aerodynamic_forces_vectorized(robot, q, q0)
            forces += F
            jacobian += J

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

    def _compute_aerodynamic_forces(self, robot: SoftRobot, q, q0):
        cd = robot.env.cd
        rho_med = robot.env.rho
        dt = robot.sim_params.dt
        u = (q - q0) / dt
        Fd = np.zeros(robot.n_dof)
        Jd = np.zeros((robot.n_dof, robot.n_dof))
        face_as = robot.face_area
        face_nodes = robot.face_nodes_shell  # (n_faces, 3)

        def cross_mat(v):
            return np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])

        def gradient_of_unit_normal(normal, q1, q2, q3, i):
            norm_normal = np.linalg.norm(normal)
            if norm_normal == 0:
                return np.zeros((3, 3))
            term = (np.dot(normal, normal) * np.eye(3) -
                    np.outer(normal, normal)) / (norm_normal ** 3)
            if i == 1:
                v_edge = q3 - q2
            elif i == 2:
                v_edge = q1 - q3
            elif i == 3:
                v_edge = q2 - q1
            else:
                raise ValueError("i must be 1, 2, or 3")
            cm = cross_mat(v_edge)
            return term @ cm

        n_faces = face_nodes.shape[0]
        for c in range(n_faces):
            node1ind = face_nodes[c, 0]
            node2ind = face_nodes[c, 1]
            node3ind = face_nodes[c, 2]

            dof1 = robot.map_node_to_dof(node1ind)
            dof2 = robot.map_node_to_dof(node2ind)
            dof3 = robot.map_node_to_dof(node3ind)

            q1 = q[dof1]
            q2 = q[dof2]
            q3 = q[dof3]

            edge1 = q2 - q1
            edge2 = q3 - q2
            face_normal = np.cross(edge1, edge2)
            face_norm = np.linalg.norm(face_normal)
            if face_norm == 0:
                continue
            face_unit_normal = face_normal / face_norm

            u1 = u[dof1]
            u2 = u[dof2]
            u3 = u[dof3]

            # Compute signs
            dot_u1 = np.dot(u1, face_unit_normal)
            sign1 = -1 if dot_u1 > 0 else 1 if dot_u1 < 0 else 0
            dot_u2 = np.dot(u2, face_unit_normal)
            sign2 = -1 if dot_u2 > 0 else 1 if dot_u2 < 0 else 0
            dot_u3 = np.dot(u3, face_unit_normal)
            sign3 = -1 if dot_u3 > 0 else 1 if dot_u3 < 0 else 0

            face_A = face_as[c]

            # Forces
            if sign1 != 0:
                Fd[dof1] += sign1 * (0.5 * rho_med * cd *
                                     face_A / 3) * (dot_u1 ** 2) * face_unit_normal
            if sign2 != 0:
                Fd[dof2] += sign2 * (0.5 * rho_med * cd *
                                     face_A / 3) * (dot_u2 ** 2) * face_unit_normal
            if sign3 != 0:
                Fd[dof3] += sign3 * (0.5 * rho_med * cd *
                                     face_A / 3) * (dot_u3 ** 2) * face_unit_normal

            # Jacobian
            # Node 1 contributions
            if sign1 != 0 and dot_u1 != 0:
                # Jd[dof1, dof1]
                grad1 = gradient_of_unit_normal(face_normal, q1, q2, q3, 1)
                term_part = (1/dt) * face_unit_normal + u1 @ grad1
                term1 = 2 * dot_u1 * np.outer(face_unit_normal, term_part)
                term2 = (dot_u1 ** 2) * grad1
                Jd[np.ix_(dof1, dof1)] += sign1 * \
                    (rho_med * cd * face_A / 3) * (term1 + term2)

                # Jd[dof1, dof2]
                grad2 = gradient_of_unit_normal(face_normal, q1, q2, q3, 2)
                term_j = (2 * np.outer(face_unit_normal, u1) +
                          dot_u1 * np.eye(3)) @ grad2
                Jd[np.ix_(dof1, dof2)] += sign1 * \
                    (rho_med * cd * face_A / 3) * dot_u1 * term_j

                # Jd[dof1, dof3]
                grad3 = gradient_of_unit_normal(face_normal, q1, q2, q3, 3)
                term_j = (2 * np.outer(face_unit_normal, u1) +
                          dot_u1 * np.eye(3)) @ grad3
                Jd[np.ix_(dof1, dof3)] += sign1 * \
                    (rho_med * cd * face_A / 3) * dot_u1 * term_j

            # Node 2 contributions
            if sign2 != 0 and dot_u2 != 0:
                # Jd[dof2, dof1]
                grad1 = gradient_of_unit_normal(face_normal, q1, q2, q3, 1)
                term_j = (2 * np.outer(face_unit_normal, u2) +
                          dot_u2 * np.eye(3)) @ grad1
                Jd[np.ix_(dof2, dof1)] += sign2 * \
                    (rho_med * cd * face_A / 3) * dot_u2 * term_j

                # Jd[dof2, dof2]
                grad2 = gradient_of_unit_normal(face_normal, q1, q2, q3, 2)
                term_part = (1/dt) * face_unit_normal + u2 @ grad2
                term1 = 2 * dot_u2 * np.outer(face_unit_normal, term_part)
                term2 = (dot_u2 ** 2) * grad2
                Jd[np.ix_(dof2, dof2)] += sign2 * \
                    (rho_med * cd * face_A / 3) * (term1 + term2)

                # Jd[dof2, dof3]
                grad3 = gradient_of_unit_normal(face_normal, q1, q2, q3, 3)
                term_j = (2 * np.outer(face_unit_normal, u2) +
                          dot_u2 * np.eye(3)) @ grad3
                Jd[np.ix_(dof2, dof3)] += sign2 * \
                    (rho_med * cd * face_A / 3) * dot_u2 * term_j

            # Node 3 contributions
            if sign3 != 0 and dot_u3 != 0:
                # Jd[dof3, dof1]
                grad1 = gradient_of_unit_normal(face_normal, q1, q2, q3, 1)
                term_j = (2 * np.outer(face_unit_normal, u3) +
                          dot_u3 * np.eye(3)) @ grad1
                Jd[np.ix_(dof3, dof1)] += sign3 * \
                    (rho_med * cd * face_A / 3) * dot_u3 * term_j

                # Jd[dof3, dof2]
                grad2 = gradient_of_unit_normal(face_normal, q1, q2, q3, 2)
                term_j = (2 * np.outer(face_unit_normal, u3) +
                          dot_u3 * np.eye(3)) @ grad2
                Jd[np.ix_(dof3, dof2)] += sign3 * \
                    (rho_med * cd * face_A / 3) * dot_u3 * term_j

                # Jd[dof3, dof3]
                grad3 = gradient_of_unit_normal(face_normal, q1, q2, q3, 3)
                term_part = (1/dt) * face_unit_normal + u3 @ grad3
                term1 = 2 * dot_u3 * np.outer(face_unit_normal, term_part)
                term2 = (dot_u3 ** 2) * grad3
                Jd[np.ix_(dof3, dof3)] += sign3 * \
                    (rho_med * cd * face_A / 3) * (term1 + term2)

        return Fd, Jd

    def _compute_aerodynamic_forces_vectorized(self, robot, q, q0):
        cd = robot.env.cd
        rho_med = robot.env.rho
        dt = robot.sim_params.dt
        u = (q - q0) / dt
        n_dof = robot.n_dof
        Fd = np.zeros(n_dof)
        Jd = np.zeros((n_dof, n_dof))

        face_as = robot.face_area 
        face_nodes = robot.face_nodes_shell
        n_faces = face_nodes.shape[0]

        dofs = [np.array([robot.map_node_to_dof(n) for n in face_nodes[:, i]])
                for i in range(3)]
        qs = [q[dofs[i]] for i in range(3)]
        us = [u[dofs[i]] for i in range(3)]

        edge1 = qs[1] - qs[0]
        edge2 = qs[2] - qs[1]
        face_normal = np.cross(edge1, edge2)
        face_norm = np.linalg.norm(face_normal, axis=1)
        valid = face_norm > 0
        face_unit_normal = np.zeros_like(face_normal)
        face_unit_normal[valid] = face_normal[valid] / face_norm[valid, None]

        dot_us = [np.einsum('ij,ij->i', us[i], face_unit_normal)
                  for i in range(3)]
        signs = [np.where(dot_us[i] > 0, -1, np.where(dot_us[i] < 0, 1, 0))
                 for i in range(3)]

        force_coeff = 0.5 * rho_med * cd * face_as / 3.0
        forces = [signs[i][:, None] * (force_coeff * (dot_us[i]**2))[:, None] * face_unit_normal
                  for i in range(3)]
        for i in range(3):
            np.add.at(Fd, dofs[i], forces[i])

        def cross_mat_vec(v):
            zeros = np.zeros(v.shape[0])
            return np.stack([
                np.stack([zeros, -v[:, 2], v[:, 1]], axis=1),
                np.stack([v[:, 2], zeros, -v[:, 0]], axis=1),
                np.stack([-v[:, 1], v[:, 0], zeros], axis=1)
            ], axis=1)  # (n_faces, 3, 3)

        # Mode mapping:
        #   mode 1: v_edge = qs[2] - qs[1]
        #   mode 2: v_edge = qs[0] - qs[2]
        #   mode 3: v_edge = qs[1] - qs[0]
        grad_modes = [None, None, None]
        for mode in range(1, 4):
            if mode == 1:
                v_edge = qs[2] - qs[1]
            elif mode == 2:
                v_edge = qs[0] - qs[2]
            else:
                v_edge = qs[1] - qs[0]
            norm_normal = np.linalg.norm(face_normal, axis=1)
            grad = np.zeros((n_faces, 3, 3))
            valid_norm = norm_normal > 0
            if np.any(valid_norm):
                nvec = face_normal[valid_norm]
                norm_sq = np.sum(nvec**2, axis=1)
                I = np.tile(np.eye(3), (np.sum(valid_norm), 1, 1))
                n_outer = nvec[:, :, None] * nvec[:, None, :]
                term = (norm_sq[:, None, None] * I - n_outer) / \
                    (norm_normal[valid_norm][:, None, None]**3)
                v_edge_v = v_edge[valid_norm]
                cm = cross_mat_vec(v_edge_v)
                grad[valid_norm] = np.matmul(term, cm)
            grad_modes[mode - 1] = grad  # grad_modes[0] for mode 1, etc.

        # For self terms, use:
        #   Node 1 → grad_modes[0], Node 2 → grad_modes[1], Node 3 → grad_modes[2]
        # For cross terms, use mapping:
        #   (0,1): grad_modes[1], (0,2): grad_modes[2],
        #   (1,0): grad_modes[0], (1,2): grad_modes[2],
        #   (2,0): grad_modes[0], (2,1): grad_modes[1]
        cross_grad = {(0, 1): grad_modes[1],
                      (0, 2): grad_modes[2],
                      (1, 0): grad_modes[0],
                      (1, 2): grad_modes[2],
                      (2, 0): grad_modes[0],
                      (2, 1): grad_modes[1]}

        I_faces = np.tile(np.eye(3), (n_faces, 1, 1))

        def add_block(Jd, rows, cols, blocks):
            r, c = np.broadcast_arrays(rows[:, :, None], cols[:, None, :])
            np.add.at(Jd, (r.flatten(), c.flatten()), blocks.flatten())

        # --- Compute Jacobian blocks ---
        # We'll collect blocks in a dictionary keyed by (i,j) for node i's contribution to node j.
        J_blocks = {}
        for i in range(3):
            for j in range(3):
                J_blocks[(i, j)] = np.zeros((n_faces, 3, 3))

        coeff = (rho_med * cd * face_as / 3.0)[:, None, None]

        # For self contributions (i == j) and cross contributions (i != j)
        for i in range(3):
            # Create mask for valid contributions for node i:
            mask = (signs[i] != 0) & (dot_us[i] != 0)
            # Self term for node i: use grad_modes[i]
            term_part = (1/dt) * face_unit_normal + \
                np.einsum('ij,ijk->ik', us[i], grad_modes[i])
            term1 = 2 * dot_us[i][:, None, None] * \
                np.einsum('ij,ik->ijk', face_unit_normal, term_part)
            term2 = (dot_us[i]**2)[:, None, None] * grad_modes[i]
            J_blocks[(i, i)][mask] = signs[i][mask][:, None, None] * \
                coeff[mask] * (term1[mask] + term2[mask])
            # Cross contributions for node i:
            for j in range(3):
                if i == j:
                    continue
                # Use appropriate gradient for cross term:
                grad_cross = cross_grad[(i, j)]
                term_j = (2 * np.einsum('ij,ik->ijk', face_unit_normal, us[i]) +
                          dot_us[i][:, None, None] * I_faces)
                term_j = np.matmul(term_j, grad_cross)
                J_blocks[(i, j)][mask] = signs[i][mask][:, None, None] * coeff[mask] * \
                    dot_us[i][mask][:, None, None] * term_j[mask]

        # --- Scatter all block contributions into global Jacobian ---
        for i in range(3):
            for j in range(3):
                add_block(Jd, dofs[i], dofs[j], J_blocks[(i, j)])

        return Fd, Jd

    def _finalize_update(self, robot: SoftRobot, q):
        u = (q - robot.q) / robot.sim_params.dt
        a1, a2 = robot.compute_time_parallel(robot.a1, robot.q, q)

        return robot.update(
            q, u, a1, a2,
            *robot.compute_material_directors(a1, a2, robot.get_theta(q)),
            robot.compute_reference_twist(
                robot.bend_twist_springs, a1, robot.compute_tangent(q), robot.ref_twist)
        )
