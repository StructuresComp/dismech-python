import numpy as np

from . import SoftRobot, fs


class TimeStepper:

    def __init__(self, robot: SoftRobot, fixed_nodes):
        self.__robot = robot.initialize(fixed_nodes)

    def step(self, robot: SoftRobot) -> SoftRobot:
        n_dof = robot.n_dof
        q0 = robot.q0
        ref_twist = robot.ref_twist

        q = robot.q0
        iteration = 1
        alpha = 1
        err = 10 * robot.sim_params.tol
        error0 = err
        solved = False

        while not solved:
            forces = np.zeros(n_dof)
            jforces = np.zeros((n_dof, n_dof))

            a1_iter, a2_iter = robot.compute_time_parallel(robot.a1, q0, q)

            tangent = robot.compute_tangent(q)
            ref_twist_iter = robot.compute_reference_twist(
                robot.bend_twist_springs, a1_iter, tangent, ref_twist)

            theta = robot.get_theta(q)
            m1, m2 = robot.compute_material_directors(a1_iter, a2_iter, theta)

            if len(robot.stretch_springs) > 0:
                Fs, Js = fs.get_fs_js(robot, q)

                forces += Fs
                jforces += Js

            if len(robot.bend_twist_springs) > 0:
                Fs, Js = fs.get_fb_jb(robot, q, m1, m2)

                forces += Fs
                jforces += Js

            # TODO: Not needed for rod cantilever
            if len(robot.face_nodes_shell) > 0:
                pass

            if "gravity" in robot.env.ext_force_list:
                pass

            if robot.sim_params.static_sim:
                f = - forces
                j = -jforces
            else:
                # TODO
                pass

    @staticmethod
    def newton_damper(alpha, iter):
        if iter < 10:
            return 1.0

        alpha *= 0.9
        if alpha < 0.1:
            return 0.1
        return alpha
