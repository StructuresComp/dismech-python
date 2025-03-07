import abc
import copy
import typing

import numpy as np

from ..soft_robot import SoftRobot
from ..state import RobotState
from ..elastics import ElasticEnergy, StretchEnergy, HingeEnergy, BendEnergy, TwistEnergy
from ..external_forces import compute_gravity_forces, compute_aerodynamic_forces_vectorized
from ..solvers import Solver, NumpySolver, PardisoSolver

_SOLVERS: typing.Dict[str, Solver] = {
    'np': NumpySolver, 'pardiso': PardisoSolver}


class TimeStepper(metaclass=abc.ABCMeta):

    def __init__(self, robot: SoftRobot, min_force=1e-8, dtype=np.float64):
        self.robot = robot
        self._min_force = min_force

        # Initialize elastics
        self.__elastic_energies: typing.List[ElasticEnergy] = []
        if robot.stretch_springs:
            self.__elastic_energies.append(
                StretchEnergy(robot.stretch_springs, robot.state))
        if robot.hinge_springs:
            self.__elastic_energies.append(
                HingeEnergy(robot.hinge_springs, robot.state))
        if robot.bend_twist_springs:
            self.__elastic_energies.append(
                BendEnergy(robot.bend_twist_springs, robot.state))
            if not robot.sim_params.two_d_sim:   # if 3d
                self.__elastic_energies.append(
                    TwistEnergy(robot.bend_twist_springs, robot.state))

        # Set solver
        # TODO: figure out how to pass parameters
        self._solver = _SOLVERS.get(robot.sim_params.solver, NumpySolver)()

        # Simulate callbacks
        self.before_step = None

    def simulate(self, robot: SoftRobot = None) -> typing.List[SoftRobot]:
        robot = robot or self.robot
        steps = int(robot.sim_params.total_time / robot.sim_params.dt) + 1

        ret = []
        for i in range(1, steps):
            if self.before_step is not None:
                robot = self.before_step(robot, i * robot.sim_params.dt)
            robot = self.step(robot)
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
        iteration_limit = False

        # Preallocate matrices
        self._forces = np.empty(robot.state.q.shape[0])
        self._jacobian = np.empty(
            (robot.state.q.shape[0], robot.state.q.shape[0]))
        self._f_free = np.empty(robot.state.free_dof.shape[0])
        self._j_free = np.empty(
            (robot.state.free_dof.shape[0], robot.state.free_dof.shape[0]))
        self._dq_free = np.empty(robot.state.free_dof.shape[0])

        while not solved:
            # Updates private variables
            self._compute_forces_and_jacobian(robot, q, robot.state.q)

            # Inertial force vs equilibrium
            if robot.sim_params.static_sim:
                self._forces *= -1
                self._jacobian *= -1
            else:
                inertial_force, interial_jacobian = self._compute_inertial_force_and_jacobian(
                    robot, q)
                np.subtract(inertial_force, self._forces, out=self._forces)
                np.subtract(interial_jacobian, self._jacobian,
                            out=self._jacobian)

            # Handle free DOF components
            self._f_free[:] = self._forces[robot.state.free_dof]
            self._j_free[:] = self._jacobian[np.ix_(
                robot.state.free_dof, robot.state.free_dof)]

            # Linear system solver
            if np.linalg.norm(self._f_free) < self._min_force:
                self._dq_free.fill(0.0)
            else:
                self._dq_free[:] = self._solver.solve(
                    self._j_free, self._f_free)

            # Adaptive damping and update
            self._dq_free *= self._adaptive_damping(alpha, iteration)
            q[robot.state.free_dof] -= self._dq_free

            # Error and convergence
            err = np.linalg.norm(self._f_free)
            err_history.append(err)

            solved = self._converged(
                err, err_history, self._dq_free, iteration, robot)

            if debug:
                print("iter: {}, error: {:.3f}".format(iteration, err))
            iteration += 1

        if iteration_limit:
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

    def _compute_forces_and_jacobian(self, robot: SoftRobot, q, q0):
        """ Sets self._forces and self._jacobian to sum of external/internal forces """
        self._forces[:].fill(0.0)
        self._jacobian[:].fill(0.0)

        # Compute reference frames and material directors
        a1_iter, a2_iter = robot.compute_time_parallel(robot.state.a1, q0, q)
        m1, m2 = robot.compute_material_directors(q, a1_iter, a2_iter)
        ref_twist = robot.compute_reference_twist(
            robot.bend_twist_springs, q, a1_iter, robot.state.ref_twist)
        tau = robot.update_pre_comp_shell(q)

        new_state = RobotState.init(q, a1_iter, a2_iter, m1, m2, ref_twist, tau)

        # Add elastic forces
        for energy in self.__elastic_energies:
            F, J = energy.grad_hess_energy_linear_elastic(
                new_state)
            self._forces[:] += F
            self._jacobian[:] += J

        # Add external forces
        # TODO: Make this also a list
        if "gravity" in robot.env.ext_force_list:
            self._forces[:] += compute_gravity_forces(robot)
        if "aerodynamics" in robot.env.ext_force_list:
            F, J, = compute_aerodynamic_forces_vectorized(robot, q, q0)
            self._forces[:] += F
            self._jacobian[:] += J

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

    def _finalize_update(self, robot: SoftRobot, q):
        u = (q - robot.state.q) / robot.sim_params.dt
        a = self._compute_acceleration(robot, q)
        a1, a2 = robot.compute_time_parallel(robot.state.a1, robot.state.q, q)
        m1, m2 = robot.compute_material_directors(q, a1, a2)
        ref_twist = robot.compute_reference_twist(
            robot.bend_twist_springs, q, a1, robot.state.ref_twist)
        return robot.update(q=q, u=u, a=a, a1=a1, a2=a2, m1=m1, m2=m2, ref_twist=ref_twist)
