import abc
import copy
import typing

import scipy.sparse as sp
import numpy as np

from ..soft_robot import SoftRobot
from ..state import RobotState
from ..elastics import ElasticEnergy, StretchEnergy, HingeEnergy, BendEnergy, TriangleEnergy, TwistEnergy
from ..external_forces import compute_gravity_forces, compute_aerodynamic_forces_vectorized, compute_ground_contact, compute_ground_contact_friction, compute_rft, compute_damping_force, compute_surface_viscous_drag, compute_thrust_force_and_jacobian, add_point_forces, predictor_step_for_ground_contact, corrector_step_for_ground_contact
from ..solvers import Solver, NumpySolver, PardisoSolver
from ..visualizer import Visualizer
from ..contact import IMCEnergy, ShellContactEnergy, IMCFrictionEnergy

_SOLVERS: typing.Dict[str, Solver] = {
    'np': NumpySolver, 'pardiso': PardisoSolver}


STRETCH = 'stretch'
HINGE = 'hinge'
MIDEDGE = 'triangle'
BEND = 'bend'
TWIST = 'twist'


class TimeStepper(metaclass=abc.ABCMeta):

    def __init__(self, robot: SoftRobot, min_force=1e-8, dtype=np.float64):
        self.robot = robot
        self._min_force = min_force
        self.original_dt = robot.sim_params.dt

        # Initialize elastics
        self.elastic_energies: typing.Dict[str, ElasticEnergy] = {}
        if robot.stretch_springs.N != 0:
            self.elastic_energies[STRETCH] = StretchEnergy(
                robot.stretch_springs, robot.state)
        if robot.hinge_springs.N != 0:
            self.elastic_energies[HINGE] = HingeEnergy(
                robot.hinge_springs, robot.state)
        if robot.triangle_springs:
            self.elastic_energies[MIDEDGE] = TriangleEnergy(
                robot.triangle_springs, robot.state)
        if robot.bend_springs.N != 0:
            self.elastic_energies[BEND] = BendEnergy(
                robot.bend_springs, robot.state
            )
            if not robot.sim_params.two_d_sim:   # if 3d
                self.elastic_energies[TWIST] = TwistEnergy(
                    robot.twist_springs, robot.state)

        if robot.sim_params.use_predictor_corrector_for_ground_contact:
            if "floorContact" not in robot.env.ext_force_list:
                raise ValueError(
                    "use_predictor_corrector_for_ground_contact=True requires "
                    "the 'floorContact' env force to be configured (it provides "
                    "ground_z, ground_h, and ground_delta). The penalty contact "
                    "force will be skipped automatically when the predictor-"
                    "corrector is active."
                )

        if "selfContact" in robot.env.ext_force_list:
            self._contact_energy = IMCEnergy(robot.contact_pairs, robot.env.delta, robot.env.h, robot.env.imc_stiffness)
            # self._contact_energy = ShellContactEnergy(robot.tri_contact_pairs, robot.env.delta, robot.env.h, robot.env.imc_stiffness, None, True)
            # self._contact_energy = ShellContactEnergy(robot.tri_contact_pairs, robot.env.delta, robot.env.h, None, False)

        if "selfFriction" in robot.env.ext_force_list:
            self._imc_friction = IMCFrictionEnergy(robot.contact_pairs, robot.env.delta, robot.env.h, \
                                                   robot.sim_params.dt, robot.env.vel_tol, robot.env.mu, robot.env.imc_stiffness\
                                                   )

        # Set solver
        # TODO: figure out how to pass parameters
        self._solver = _SOLVERS.get(robot.sim_params.solver, NumpySolver)()

        # Simulate callbacks
        self.before_step = None

        # Contact-force bookkeeping. _contact_z_dofs is the cumulative set of
        # z-DOF indices pinned to the ground by the predictor (and not yet
        # released by the corrector); predictor adds, corrector removals are
        # detected via a fixed_dof diff so the same code handles both
        # no_future_freeing branches. _last_contact_info is the
        # (node_indices, F_xyz) snapshot for the most recent step — F_xyz has
        # shape (N, 3) with columns [Fx, Fy, Fz] at the node's x/y/z DOFs.
        # The z column is the normal reaction; x/y are ~0 for z-only pinned
        # (sliding) nodes and the friction reaction for fully stuck nodes.
        # simulate() appends _last_contact_info to contact_log at the same
        # cadence as Fs.
        self._contact_z_dofs: np.ndarray = np.array([], dtype=int)
        self._last_contact_info: typing.Tuple[np.ndarray, np.ndarray] = (
            np.array([], dtype=int), np.zeros((0, 3), dtype=np.float64))
        self.contact_log: typing.List[typing.Tuple[np.ndarray, np.ndarray]] = []

    def simulate(self, robot: SoftRobot = None, viz: Visualizer = None) -> typing.Tuple[typing.List[SoftRobot], typing.List[float], typing.List[float], typing.List[np.ndarray]]:
        robot = robot or self.robot
        steps = int(robot.sim_params.total_time / robot.sim_params.dt) + 1

        if viz is not None:
            viz.update(robot, 0)

        ret = []
        f_norms = []
        time_array = []
        Fs = []
        self.contact_log = []
        self._contact_z_dofs = np.array([], dtype=int)
        i = 0
        t = 0.0
        time_array.append(t)
        ret.append(robot)
        Fs.append(np.zeros_like(robot.state.q))
        f_norms.append(0.0)
        self.contact_log.append(
            (np.array([], dtype=int), np.zeros((0, 3), dtype=np.float64)))

        while t < robot.sim_params.total_time:
            # Handle user function
            if self.before_step is not None:
                robot = self.before_step(robot, i * robot.sim_params.dt)

            try:
                robot, f_norm, F = self.step(robot)
            except Exception as e:
                print("Error occurred during simulation step:", e)
                robot.sim_params.dt *= 0.1
                print("Reducing time step to:", robot.sim_params.dt)
                robot, f_norm, F = self.step(robot)

            i += 1
            t += robot.sim_params.dt

            # Update on step interval
            if viz is not None and i % robot.sim_params.plot_step == 0:
                viz.update(robot, t)
            if robot.sim_params.log_data and i % robot.sim_params.log_step == 0:
                ret.append(robot)
                f_norms.append(f_norm)
                Fs.append(F)
                self.contact_log.append(self._last_contact_info)
                time_array.append(t)

            # print current time
            print("current_time: ", t)
        return ret, time_array, f_norms, Fs

    def step(self, robot: SoftRobot = None, debug: bool = True) -> typing.Tuple[SoftRobot, float, np.ndarray]:
        robot = robot or self.robot

        q, f_norm, F = self._solve_step(robot, debug)

        if robot.sim_params.use_predictor_corrector_for_ground_contact:
            robot, revert_to_start, vertically_constrained_nodes = \
                predictor_step_for_ground_contact(robot, q)
            if revert_to_start:
                # Predictor mutated state.q to snap penetrating z-DOFs and
                # added them to fixed_dof; re-solve the step from the snapped
                # start-of-step state with the new constraints in place.
                q, f_norm, F = self._solve_step(robot, debug)
                # New z-DOFs the predictor just pinned join the cumulative set.
                new_z = vertically_constrained_nodes * 3 + 2
                self._contact_z_dofs = np.union1d(self._contact_z_dofs, new_z)
                # Run corrector and detect any z-DOFs it released via a
                # fixed_dof diff. With no_future_freeing=True the corrector
                # only adds (stuck x/y), so removed is empty and the set
                # grows monotonically. With the default behavior, sliding
                # nodes' z-DOFs are released and drop out of the set.
                fixed_before = robot.fixed_dof.copy()
                robot = corrector_step_for_ground_contact(
                    robot, q, vertically_constrained_nodes,
                    no_future_freeing=robot.sim_params.no_future_freeing_after_ground_contact,
                    frictionless=robot.sim_params.frictionless_pc_contact)
                removed = np.setdiff1d(fixed_before, robot.fixed_dof)
                self._contact_z_dofs = np.setdiff1d(
                    self._contact_z_dofs, removed)

        # F is the residual at the converged q for the most recent solve.
        # For every contact-active node we pull all three nodal DOFs: the z
        # entry is the normal contact reaction; the x/y entries are ~0 for
        # nodes whose x/y remain free (sliding / z-only state) and the
        # friction reactions for nodes the corrector fully constrained.
        if self._contact_z_dofs.size > 0:
            nodes = self._contact_z_dofs // 3
            xyz_dofs = nodes[:, None] * 3 + np.arange(3)[None, :]
            self._last_contact_info = (nodes.copy(), F[xyz_dofs].copy())
        else:
            self._last_contact_info = (
                np.array([], dtype=int),
                np.zeros((0, 3), dtype=np.float64))

        self.robot = self._finalize_update(robot, q)
        self.f_norm = f_norm
        return self.robot, self.f_norm, F

    def _compute_residual_at(self, robot: SoftRobot, q: np.ndarray, first_iter: bool = False) -> typing.Tuple[np.ndarray, typing.Any]:
        """Compute the full-size Newton residual F (and Jacobian J) at q.

        F is the residual we drive to zero on the free DOFs. On fixed DOFs F
        is the imbalance the constraint absorbs — i.e., the reaction force.
        """
        q_eval = self._compute_evaluation_position(robot, q)
        u_eval = self._compute_evaluation_velocity(robot, q)

        F = np.zeros(q.shape[0])
        if robot.sim_params.sparse:
            J = sp.csr_matrix((q.shape[0], q.shape[0]), dtype=np.float64)
        else:
            J = np.zeros((q.shape[0], q.shape[0]))

        if not robot.sim_params.static_sim:
            inertial_force, inertial_jacobian = self._compute_inertial_force_and_jacobian(
                robot, q)
            F += inertial_force
            if robot.sim_params.sparse:
                J += sp.diags(inertial_jacobian, format='csr')
            else:
                ndof_diag = np.arange(q.shape[0])
                J[ndof_diag, ndof_diag] += inertial_jacobian

        F, J = self._compute_forces_and_jacobian(
            F, J, robot, q_eval, u_eval, first_iter)
        return F, J

    def _solve_step(self, robot: SoftRobot, debug: bool) -> typing.Tuple[np.ndarray, float, np.ndarray]:
        """Run the Newton iteration to convergence and return the trial q,
        residual norm, and force vector without finalizing robot state."""
        # Initialize iteration variables
        q = copy.deepcopy(robot.state.q)
        alpha = 1.0
        iteration = 1
        err_history = []
        solved = False

        while not solved:
            F, J = self._compute_residual_at(
                robot, q, first_iter=(iteration == 1))

            # Handle free DOF components
            f_free = F[robot.state.free_dof]
            if robot.sim_params.sparse:
                j_free = J[robot.state.free_dof,
                           :][:, robot.state.free_dof]
            else:
                j_free = J[np.ix_(
                    robot.state.free_dof, robot.state.free_dof)]

            # For debugging:
            # print("Determinant of Jacobian: ", np.linalg.det(j_free.toarray() if robot.sim_params.sparse else j_free))

            # Linear system solver
            if np.linalg.norm(f_free) < self._min_force:
                dq_free = np.zeros_like(f_free)
            else:
                dq_free = self._solver.solve(j_free, f_free)

            # Adaptive damping and update
            if iteration > robot.sim_params.line_search_iters:
                if robot.sim_params.use_line_search:
                    alpha_orig = alpha
                    alpha = self._line_search(robot, q, dq_free, f_free, j_free)

                    if alpha<=0.0:
                        alpha = self._adaptive_damping(alpha_orig)
                else:
                    alpha = self._adaptive_damping(alpha)
            dq_free *= alpha
            q[robot.state.free_dof] -= dq_free

            # Error and convergence
            err = np.linalg.norm(f_free)
            err_history.append(err)

            solved = self._converged(
                err, err_history, dq_free, iteration, robot)

            if debug:
                print("iter: {}, error: {:.3f}".format(iteration, err))
            iteration += 1

        if iteration >= robot.sim_params.max_iter:
            raise ValueError(
                "Iteration limit {} reached before convergence".format(robot.sim_params.max_iter))

        # Re-evaluate F at the converged q. The in-loop F was computed at the
        # previous q (before the final dq update), so F[fixed_dof] would be one
        # Newton step stale. The fixed-DOF entries here are the constraint
        # reaction forces — including contact reactions at predictor-pinned
        # z-DOFs — so we want them at the actual converged configuration.
        F, _ = self._compute_residual_at(robot, q, first_iter=False)
        return q, np.linalg.norm(F[robot.state.free_dof]), F

    @abc.abstractmethod
    def _compute_inertial_force_and_jacobian(self, robot: SoftRobot, q: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
        pass

    def _compute_acceleration(self, robot: SoftRobot, q: np.ndarray) -> np.ndarray:
        return np.zeros_like(q)

    def _compute_velocity(self, robot: SoftRobot, q: np.ndarray) -> np.ndarray:
        return (q - robot.state.q) / robot.sim_params.dt

    def compute_total_elastic_energy(self, state: RobotState) -> np.ndarray:
        total = 0.0
        for energy in self.elastic_energies.values():
            total += energy.get_energy_linear_elastic(state)
        return np.array(total)

    def _compute_evaluation_position(self, robot: SoftRobot, q: np.ndarray) -> np.ndarray:
        return q

    def _compute_evaluation_velocity(self, robot: SoftRobot, q: np.ndarray) -> np.ndarray:
        return self._compute_velocity(robot, q)

    def _compute_forces_and_jacobian(self, forces, jacobian, robot: SoftRobot, q, u, first_iter=False):
        """ Computes forces and jacobian as sum of external and internal forces. """
        
        # Compute reference frames and material directors
        a1_iter, a2_iter = robot.compute_time_parallel(
            robot.state.a1, robot.state.q, q)
        m1, m2 = robot.compute_material_directors(q, a1_iter, a2_iter)
        ref_twist = robot.compute_reference_twist(
            robot.twist_springs, q, a1_iter, robot.state.ref_twist)
        tau = robot.update_pre_comp_shell(q)

        new_state = RobotState.init(
            q, a1_iter, a2_iter, m1, m2, ref_twist, tau)

        # Add elastic forces
        for energy in self.elastic_energies.values():
            F, J = energy.grad_hess_energy_linear_elastic(
                new_state, robot.sim_params.sparse)
            forces -= F
            jacobian -= J
        # Add external forces
        # TODO: Make this also a list
        if "gravity" in robot.env.ext_force_list:
            forces -= compute_gravity_forces(robot)
        # ignore for now
        if "aerodynamics" in robot.env.ext_force_list:
            F, J, = compute_aerodynamic_forces_vectorized(robot, q, u)
            forces -= F
            jacobian -= J  # FIXME: Sparse option
        if "damping" in robot.env.ext_force_list:
            F, J, = compute_damping_force(robot, q, u)
            forces -= F
            jacobian -= J  # FIXME: Sparse option
        if "hydrodynamics" in robot.env.ext_force_list:
            F, J, = compute_surface_viscous_drag(robot, q, u)
            forces -= F
            jacobian -= J  # FIXME: Sparse option
        if "thrust" in robot.env.ext_force_list:
            F, J, = compute_thrust_force_and_jacobian(robot, q, u)
            forces -= F
            jacobian -= J  # FIXME: Sparse option
        if "floorContact" in robot.env.ext_force_list and not robot.sim_params.use_predictor_corrector_for_ground_contact:
            if "floorFriction" in robot.env.ext_force_list:
                F, J = compute_ground_contact_friction(robot, q, u)
            else:
                F, J = compute_ground_contact(robot, q)
            forces -= F
            jacobian -= J
        if "rft" in robot.env.ext_force_list:
            F, J = compute_rft(robot, q, u)
            forces -= F
            jacobian -= J
        if "selfContact" in robot.env.ext_force_list:
            Fcon, Jcon = self._contact_energy.grad_hess_energy(new_state, robot, forces, first_iter)
            forces -= Fcon
            jacobian -= Jcon
        if "selfFriction" in robot.env.ext_force_list:
            F = self._imc_friction.grad_friction(new_state, robot, forces, first_iter)
            forces -= F
            jacobian -= J
        if "pointForces" in robot.env.ext_force_list:
            forces -= add_point_forces(robot, robot.env.point_force_node_indices, robot.env.point_force_vectors)

        return forces, jacobian

    def _converged(self,
                   err: float,
                   err_history: typing.List[float],
                   dq: np.ndarray,
                   iteration: int,
                   robot: SoftRobot):
        """ Check all convergence criteria """
        #print(err)
        if dq.size == 0:
            # No free DOFs — nothing left to solve, treat as converged.
            return True
        disp_converged = np.max(np.abs(dq)) / \
            robot.sim_params.dt < robot.sim_params.dtol
        force_converged = err < robot.sim_params.tol
        relative_converged = err < err_history[0] * robot.sim_params.ftol
        iteration_limit = iteration >= robot.sim_params.max_iter

        return any([force_converged, relative_converged, disp_converged, iteration_limit])

    def _adaptive_damping(self, alpha):
        """ Adaptive damping based on the current alpha value. """
        return max(alpha * 0.9, 0.1)

    def _line_search(self, robot, q, dq, F, J, m1=0.1, m2=0.9, alpha_low=0.0, alpha_high=1.0):
        assert F.shape[0] == len(robot.state.free_dof), "F must be reduced to free DOFs"
        d0 = -np.dot(F, J @ dq)  # Directional derivative (should be negative)

        if d0 >= 0:
            # Not a descent direction, don't take any step
            return 0.0

        alpha = alpha_high
        alpha_min = 1e-6
        alpha_max = 2.0
        max_iter = 100
        tol = 1e-6
        iteration = 0
        success = False

        ndof = q.shape[0]
        free_dof = robot.state.free_dof

        while not success:
            # Build full dq vector
            dq_full = np.zeros_like(q)
            dq_full[free_dof] = dq
            q_new = q - alpha * dq_full

            # Compute updated state
            q_eval = self._compute_evaluation_position(robot, q_new)
            u_eval = self._compute_evaluation_velocity(robot, q_new)

            # Recompute force at new configuration
            F_iter = np.zeros_like(q)
            if robot.sim_params.sparse:
                J_iter = sp.csr_matrix((ndof, ndof), dtype=np.float64)
            else:
                J_iter = np.zeros((ndof, ndof))

            if not robot.sim_params.static_sim:
                inertial_force, _ = self._compute_inertial_force_and_jacobian(robot, q_new)
                F_iter += inertial_force
                # Optional: add inertia Jacobian if needed

            F_new, _ = self._compute_forces_and_jacobian(F_iter, J_iter, robot, q_eval, u_eval)

            # Evaluate Wolfe conditions (on free DOFs only)
            F_norm_sq_old = 0.5 * np.linalg.norm(F)**2
            F_norm_sq_new = 0.5 * np.linalg.norm(F_new[free_dof])**2
            lhs = F_norm_sq_new - F_norm_sq_old
            rhs_low = alpha * m2 * d0
            rhs_high = alpha * m1 * d0

            if rhs_low <= lhs <= rhs_high:
                success = True
            elif lhs < rhs_low:
                alpha_low = alpha
            else:
                alpha_high = alpha

            # Update alpha
            if alpha_high < alpha_max:
                alpha = 0.5 * (alpha_low + alpha_high)
            else:
                alpha = 10 * alpha

            iteration += 1

            if (alpha < alpha_min) or (iteration > max_iter) or ((alpha_high - alpha_low) < tol):
                # Line search failed to find suitable step
                return 0.0

        return alpha


    def _finalize_update(self, robot: SoftRobot, q):
        u = self._compute_velocity(robot, q)
        a = self._compute_acceleration(robot, q)
        a1, a2 = robot.compute_time_parallel(robot.state.a1, robot.state.q, q)
        m1, m2 = robot.compute_material_directors(q, a1, a2)
        ref_twist = robot.compute_reference_twist(
            robot.twist_springs, q, a1, robot.state.ref_twist)
        return robot.update(q=q, u=u, a=a, a1=a1, a2=a2, m1=m1, m2=m2, ref_twist=ref_twist)
