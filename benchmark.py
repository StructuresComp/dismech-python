import argparse
import numpy as np
import time
import dismech


def cantilever(EA: float, r0: float = 2e-2, dt: float = 1e-2, damping: float = 0.0):
    geom = dismech.GeomParams(rod_r0=r0,
                              shell_h=0.0)

    material = dismech.Material(density=1000,
                                youngs_rod=EA,
                                youngs_shell=0,
                                poisson_rod=0.5,
                                poisson_shell=0)

    static_2d_sim = dismech.SimParams(static_sim=False,
                                      two_d_sim=True,   # no twisting
                                      use_mid_edge=False,
                                      use_line_search=False,
                                      show_floor=False,
                                      log_data=True,
                                      log_step=1,
                                      dt=dt,
                                      max_iter=25,
                                      total_time=10.0,
                                      plot_step=1,
                                      tol=1e-4,
                                      ftol=1e-4,
                                      dtol=1e-2)

    env = dismech.Environment()
    env.add_force('gravity', g=np.array([0.0, 0.0, -9.81]))
    if damping != 0.0:
        env.add_force('viscous', eta=damping)

    geo = dismech.Geometry.from_txt(
        'tests/resources/rod_cantilever/horizontal_rod_n51.txt')
    robot = dismech.SoftRobot(geom, material, geo, static_2d_sim, env)

    fixed_points = np.array([0, 1])
    return robot.fix_nodes(fixed_points)


def helix(EA: float, r0: float = 1e-3, dt: float = 1e-2, damping: float = 0.0):
    geom = dismech.GeomParams(rod_r0=r0,
                              shell_h=0)

    material = dismech.Material(density=1273.52,
                                youngs_rod=EA,
                                youngs_shell=0,
                                poisson_rod=0.5,
                                poisson_shell=0)

    static_2d_sim = dismech.SimParams(static_sim=False,
                                      two_d_sim=False,
                                      use_mid_edge=False,
                                      use_line_search=False,
                                      show_floor=False,
                                      log_data=True,
                                      log_step=1,
                                      dt=dt,
                                      max_iter=25,
                                      total_time=10.0,
                                      plot_step=1,
                                      tol=1e-4,
                                      ftol=1e-4,
                                      dtol=1e-2,
                                      sparse=True,
                                      solver="pardiso")

    env = dismech.Environment()
    env.add_force('gravity', g=np.array([0.0, 0.0, -9.81]))
    if damping != 0.0:
        env.add_force('viscous', eta=damping)

    geo = dismech.Geometry.from_txt('tests/resources/helix/helix_n106.txt')
    robot = dismech.SoftRobot(geom, material, geo, static_2d_sim, env)

    fixed_points = np.array([0, 1])
    return robot.fix_nodes(fixed_points)


def snake():
    geom = dismech.GeomParams(rod_r0=0.35*0.005,
                              shell_h=0)

    material = dismech.Material(density=1000,
                                youngs_rod=1e6,
                                youngs_shell=0,
                                poisson_rod=0.5,
                                poisson_shell=0)

    static_2d_sim = dismech.SimParams(static_sim=False,
                                      two_d_sim=False,   # no twisting
                                      use_mid_edge=False,
                                      use_line_search=False,
                                      show_floor=False,
                                      log_data=True,
                                      log_step=1,
                                      dt=1e-2,
                                      max_iter=25,
                                      total_time=10.0,
                                      plot_step=1,
                                      tol=1e-4,
                                      ftol=1e-4,
                                      dtol=1e-2)

    env = dismech.Environment()
    env.add_force('rft', ct=0.5, cn=1.0)

    geo = dismech.Geometry.from_txt(
        'tests/resources/rod_cantilever/horizontal_rod_n101.txt')

    return dismech.SoftRobot(geom, material, geo, static_2d_sim, env)


def actuate_snake(robot, t,
                  amplitude=0.0393,
                  frequency=0.5,
                  spatial_wavelength=0.3,
                  phase_offset=-np.pi/2,
                  ramp_time=2.0,
                  direction=+1):
    """
    Apply traveling-wave bending strains to the snake robot.

    Parameters
    ----------
    robot : Robot instance (mutated in-place)
    t : float, current sim time [s]
    amplitude : float, peak strain magnitude
    frequency : float, drive frequency [Hz]
    spatial_wavelength : float, body-normalized wavelength (s in [0,1])
    phase_offset : float, phase shift between channels (if both used)
    ramp_time : float, duration over which actuation smoothly ramps from 0->1
    direction : +1 for forward (reversed from your previous, backward drift),
                -1 for backward (original behavior)
    """
    n_bends = robot.bend_springs.inc_strain.shape[0]
    s = np.linspace(0.0, 1.0, n_bends)

    omega = 2.0 * np.pi * frequency
    k = 2.0 * np.pi / spatial_wavelength

    tau = np.clip(t / ramp_time, 0.0, 1.0)
    coeff = 0.5 * (1.0 - np.cos(np.pi * tau))

    vertical_wave = amplitude * np.sin(omega * t + direction * k * s)
    # tangential_wave = amplitude * np.sin(omega * t + direction * k * s + phase_offset)

    robot.bend_springs.inc_strain[:, 1] = coeff * vertical_wave

    return robot


def _time(robot: dismech.SoftRobot, stepper_cls, iters: int = 5):
    def _run(step):
        start = time.time()
        step.simulate()
        end = time.time()
        return end - start
    total = 0.0
    for _ in range(iters):
        total += _run(stepper_cls(robot))
    return total / iters


def main():
    for stepper in [dismech.NewmarkBetaTimeStepper, dismech.ImplicitEulerTimeStepper]:
        cant_settings = [(1e5, 1e-2), (1e6, 5e-3), (1e7, 5e-3)]
        for (ea, dt) in cant_settings:
            cant = cantilever(EA=ea, dt=dt)
            print(
                f"Cantilever E={ea} dt={dt} stepper={stepper}: {_time(cant, stepper):.2f}")
        helix_settings = [(1e7, 5e-3, 5e-2), (1e9, 1e-3, 5e-3)]
        for (ea, r, dt) in helix_settings:
            h = helix(EA=ea, r0=r, dt=dt)
            print(
                f"Helix E={ea} R={r} dt={dt} stepper={stepper}: {_time(h, stepper):.2f}")

        def add_actuation(robot):
            step = stepper(robot)
            step.before_step = actuate_snake
            return step
        s = snake()
        print(
            f"Snake stepper={stepper}: {_time(s, add_actuation):.2f}")


if __name__ == "__main__":
    def add_actuation(robot):
        step = dismech.ImplicitEulerTimeStepper(robot)
        step.before_step = actuate_snake
        return step
    s = snake()
    print(
        f"Snake stepper={dismech.ImplicitEulerTimeStepper}: {_time(s, add_actuation):.2f}")
    def add_actuation(robot):
        step = dismech.NewmarkBetaTimeStepper(robot)
        step.before_step = actuate_snake
        return step
    s = snake()
    print(
        f"Snake stepper={dismech.NewmarkBetaTimeStepper}: {_time(s, add_actuation):.2f}")
    
    #main()
