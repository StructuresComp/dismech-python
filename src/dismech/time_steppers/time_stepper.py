import abc
import copy
import typing

import scipy.sparse as sp
import numpy as np

from ..soft_robot import SoftRobot
from ..state import RobotState
from ..elastics import ElasticEnergy, StretchEnergy, HingeEnergy, BendEnergy, TriangleEnergy, TwistEnergy
from ..external_forces import compute_gravity_forces, compute_aerodynamic_forces_vectorized
from ..solvers import Solver, NumpySolver, PardisoSolver
from ..visualizer import Visualizer
from ..contact import IMCEnergy

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

        # Initialize elastics
        self.elastic_energies: typing.Dict[str, ElasticEnergy] = {}
        if robot.stretch_springs:
            self.elastic_energies[STRETCH] = StretchEnergy(
                robot.stretch_springs, robot.state)
        if robot.hinge_springs:
            self.elastic_energies[HINGE] = HingeEnergy(
                robot.hinge_springs, robot.state)
        if robot.triangle_springs:
            self.elastic_energies[MIDEDGE] = TriangleEnergy(
                robot.triangle_springs, robot.state)
        if robot.bend_twist_springs:
            self.elastic_energies[BEND] = BendEnergy(
                robot.bend_twist_springs, robot.state)
            if not robot.sim_params.two_d_sim:   # if 3d
                self.elastic_energies[TWIST] = TwistEnergy(
                    robot.bend_twist_springs, robot.state)
                
        if "selfContact" in robot.env.ext_force_list:
            self._contact_energy = IMCEnergy(np.vstack([p.ind for p in robot.contact_pairs]), robot.env.delta, robot.env.h)

        # Set solver
        # TODO: figure out how to pass parameters
        self._solver = _SOLVERS.get(robot.sim_params.solver, NumpySolver)()

        # Simulate callbacks
        self.before_step = None

    def simulate(self, robot: SoftRobot = None, viz: Visualizer = None) -> typing.List[SoftRobot]:
        robot = robot or self.robot
        steps = int(robot.sim_params.total_time / robot.sim_params.dt) + 1

        if viz is not None:
            viz.update(robot, 0)

        ret = []
        for i in range(1, steps):
            # Handle user function
            if self.before_step is not None:
                robot = self.before_step(robot, i * robot.sim_params.dt)
            robot = self.step(robot)

            # Update on step interval
            if viz is not None and i % robot.sim_params.plot_step == 0:
                viz.update(robot, i * robot.sim_params.dt)
            if robot.sim_params.log_data and i % robot.sim_params.log_step == 0:
                ret.append(robot)
        return ret

    def step(self, robot: SoftRobot = None, debug: bool = False) -> SoftRobot:
        robot = robot or self.robot

        # Initialize iteration variables
        q = copy.deepcopy(robot.state.q)
        alpha = 1.0
        iteration = 1
        err_history = []
        solved = False

        # Preallocate matrices
        ndof_diag = np.arange(q.shape[0])

        while not solved:
            # Some integrators compute F and J not at q_{n+1} (midpoint)
            q_eval = self._compute_evaluation_position(robot, q)
            u_eval = self._compute_evaluation_velocity(robot, q)

            F, J = self._compute_forces_and_jacobian(robot, q_eval, u_eval)

            # Inertial force vs equilibrium
            if not robot.sim_params.static_sim:
                inertial_force, inertial_jacobian = self._compute_inertial_force_and_jacobian(
                    robot, q)
                F += inertial_force

                if robot.sim_params.sparse:
                    J += sp.diags(inertial_jacobian, format='csr')
                else:
                    J[ndof_diag, ndof_diag] += inertial_jacobian

            # Handle free DOF components
            f_free = F[robot.state.free_dof]
            if robot.sim_params.sparse:
                j_free = J[robot.state.free_dof,
                           :][:, robot.state.free_dof]
            else:
                j_free = J[np.ix_(
                    robot.state.free_dof, robot.state.free_dof)]

            # Linear system solver
            if np.linalg.norm(f_free) < self._min_force:
                dq_free = np.zeros_like(f_free)
            else:
                dq_free = self._solver.solve(j_free, f_free)

            # Adaptive damping and update
            if robot.sim_params.use_line_search:
                alpha = self._line_search(robot, q, dq_free, f_free, j_free)
            else:
                alpha = self._adaptive_damping(alpha, iteration)
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

        # Final update and return
        self.robot = self._finalize_update(robot, q)
        return self.robot

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
        return total

    def _compute_evaluation_position(self, robot: SoftRobot, q: np.ndarray) -> np.ndarray:
        return q

    def _compute_evaluation_velocity(self, robot: SoftRobot, q: np.ndarray) -> np.ndarray:
        return (q - robot.state.q) / robot.sim_params.dt

    def _compute_forces_and_jacobian(self, robot: SoftRobot, q, u):
        """ Computes forces and jacobian as sum of external and internal forces. """
        forces = np.zeros(q.shape[0])

        if robot.sim_params.sparse:
            jacobian = sp.csr_matrix(
                (q.shape[0], q.shape[0]), dtype=np.float64)
        else:
            jacobian = np.zeros((q.shape[0], q.shape[0]))

        # Compute reference frames and material directors
        a1_iter, a2_iter = robot.compute_time_parallel(
            robot.state.a1, robot.state.q, q)
        m1, m2 = robot.compute_material_directors(q, a1_iter, a2_iter)
        ref_twist = robot.compute_reference_twist(
            robot.bend_twist_springs, q, a1_iter, robot.state.ref_twist)
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
        if "selfContact" in robot.env.ext_force_list:
            F, J = self._contact_energy.grad_hess_energy(q)
            forces -= F
            jacobian -= J
        return forces, jacobian

    def _converged(self,
                   err: float,
                   err_history: typing.List[float],
                   dq: np.ndarray,
                   iteration: int,
                   robot: SoftRobot):
        """ Check all convergence criteria """
        disp_converged = np.max(np.abs(dq)) / \
            robot.sim_params.dt < robot.sim_params.dtol
        force_converged = err < robot.sim_params.tol
        relative_converged = err < err_history[0] * robot.sim_params.ftol
        iteration_limit = iteration >= robot.sim_params.max_iter

        return any([force_converged, relative_converged, disp_converged, iteration_limit])

    def _adaptive_damping(self, alpha, iteration):
        if iteration < 10:
            return 1.0

        return max(alpha * 0.9, 0.1)

    def _line_search(self, robot, q, dq, F, J, m1=0.1, m2=0.9, alpha_low=0.0, alpha_high=1.0, max_iter=10):
        d0 = np.dot(F, J @ dq)
        alpha = alpha_high
        iteration = 0
        while iteration < max_iter:
            # Construct full dq matrix
            dq_full = np.zeros_like(q)
            dq_full[robot.state.free_dof] = dq
            q_new = q - alpha * dq_full

            # Evaluate at same point as outer step
            q_eval = self._compute_evaluation_position(robot, q_new)
            u_eval = self._compute_evaluation_velocity(robot, q_new)
            F_new, _ = self._compute_forces_and_jacobian(robot, q_eval, u_eval)

            lhs = 0.5 * np.linalg.norm(F_new) ** 2 - \
                0.5 * np.linalg.norm(F) ** 2
            rhs_low = alpha * m2 * d0
            rhs_high = alpha * m1 * d0

            if rhs_low <= lhs <= rhs_high:
                return alpha
            elif lhs < rhs_low:
                alpha_low = alpha
            else:
                alpha_high = alpha
            alpha = 0.5 * (alpha_low + alpha_high)
            iteration += 1
        return alpha

    def _finalize_update(self, robot: SoftRobot, q):
        u = self._compute_velocity(robot, q)
        a = self._compute_acceleration(robot, q)
        a1, a2 = robot.compute_time_parallel(robot.state.a1, robot.state.q, q)
        m1, m2 = robot.compute_material_directors(q, a1, a2)
        ref_twist = robot.compute_reference_twist(
            robot.bend_twist_springs, q, a1, robot.state.ref_twist)
        return robot.update(q=q, u=u, a=a, a1=a1, a2=a2, m1=m1, m2=m2, ref_twist=ref_twist)
