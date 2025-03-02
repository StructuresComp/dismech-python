import copy

import numpy as np
import pypardiso
from scipy.sparse import csr_matrix

from . import fs
from .softrobot import SoftRobot
from .elastics import StretchEnergy, HingeEnergy
from .external_forces import compute_gravity_forces, compute_aerodynamic_forces_vectorized


class TimeStepper:

    def __init__(self, robot: SoftRobot, fixed_nodes):
        self.robot = robot.fix_nodes(fixed_nodes)
        self.fixed_nodes = fixed_nodes
        self.epsilon = 1e-8  # Regularization parameter
        self.min_force = 1e-8  # Threshold for negligible forces

        # Initialize elastics
        self.__elastic_energies = []
        if robot.stretch_springs:
            self.__elastic_energies.append(StretchEnergy(robot.stretch_springs))
        if robot.hinge_springs:
            self.__elastic_energies.append(HingeEnergy(robot.hinge_springs))

        self.__energies = []

    def step(self, robot: SoftRobot = None, debug=False) -> SoftRobot:
        robot = robot or self.robot
        params = robot.sim_params
        free_idx = robot.free_dof

        # Initialize iteration variables
        q0 = robot.q.view()
        q = copy.deepcopy(robot.q)

        alpha = 1.0
        iteration = 1
        err_history = []
        solved = False
        iteration_limit = False  # pull out for error

        while not solved:
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
            f_free = forces[free_idx]
            j_free = jacobian[np.ix_(free_idx, free_idx)]

            # Linear system solver
            if robot.sim_params.solver == 'np':
                dq_free = self._safe_solve(j_free, f_free)
            else:
                dq_free = self._pardiso_solve(j_free, f_free)

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

            solved = any([force_converged, relative_converged,
                         disp_converged, iteration_limit])
            if debug:
                print("iter: {}, error: {:.3f}".format(iteration, err))

            iteration += 1

        if iteration_limit:
            raise ValueError(
                "Iteration limit {} reached before convergence".format(params.max_iter))

        # Final update and return
        self.robot = self._finalize_update(robot, q)
        return self.robot

    def _compute_forces_and_jacobian(self, robot: SoftRobot, q, q0):
        forces = np.zeros(robot.n_dof)
        jacobian = np.zeros((robot.n_dof, robot.n_dof))

        # Compute reference frames and material directors
        a1_iter, a2_iter = robot.compute_time_parallel(robot.a1, q0, q)
        theta = robot.get_theta(q)
        m1, m2 = robot.compute_material_directors(a1_iter, a2_iter, theta)
        ref_twist = robot.compute_reference_twist(
            robot.bend_twist_springs, a1_iter, robot.compute_tangent(q), robot.ref_twist)

        # Add bend/twist contributions
        if robot.bend_twist_springs:
            Fb, Jb = fs.get_fb_jb_vectorized(robot, q, m1, m2)
            # Fb, Jb = fs.get_fb_jb(robot, q, m1, m2)
            forces += Fb
            jacobian += Jb
            if not robot.sim_params.two_d_sim:
                Ft, Jt = fs.get_ft_jt_vectorized(robot, q, ref_twist)
                forces += Ft
                jacobian += Jt

        # Add elastics
        for energy in self.__elastic_energies:
            F, J = energy.grad_hess_energy_linear_elastic(q, m1, m2, ref_twist)
            forces += F
            jacobian += J

        # Add gravity forces
        if "gravity" in robot.env.ext_force_list:
            forces += compute_gravity_forces(robot)
        if "aerodynamics" in robot.env.ext_force_list:
            F, J = compute_aerodynamic_forces_vectorized(robot, q, q0)
            forces += F
            jacobian += J

        return forces, jacobian

    def _safe_solve(self, J, F):
        if np.linalg.norm(F) < self.min_force:
            return np.zeros_like(F)
        return np.linalg.solve(J, F)

    def _pardiso_solve(self, J, F):
        """Pardiso solver (pypardiso)"""
        J_sparse = csr_matrix(J)
        if np.linalg.norm(F) < self.min_force:
            return np.zeros_like(F)

        return pypardiso.spsolve(J_sparse, F)

    def _adaptive_damping(self, alpha, iteration):
        if iteration < 10:
            return 1.0

        return max(alpha * 0.9, 0.1)

    def _finalize_update(self, robot: SoftRobot, q):
        u = (q - robot.q) / robot.sim_params.dt
        a1, a2 = robot.compute_time_parallel(robot.a1, robot.q, q)

        return robot.update(
            q, u, a1, a2,
            *robot.compute_material_directors(a1, a2, robot.get_theta(q)),
            robot.compute_reference_twist(
                robot.bend_twist_springs, a1, robot.compute_tangent(q), robot.ref_twist)
        )
