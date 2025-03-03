import dataclasses
import typing


@dataclasses.dataclass
class GeomParams:
    rod_r0: float
    shell_h: float
    axs: typing.Optional[float] = None
    jxs: typing.Optional[float] = None
    ixs1: typing.Optional[float] = None
    ixs2: typing.Optional[float] = None


@dataclasses.dataclass
class Material:
    density: float
    youngs_rod: float
    youngs_shell: float
    poisson_rod: float
    poisson_shell: float


@dataclasses.dataclass
class SimParams:
    static_sim: bool
    two_d_sim: bool
    use_mid_edge: bool
    use_line_search: bool
    log_data: bool
    log_step: int
    show_floor: bool
    dt: float
    max_iter: int
    total_time: float
    plot_step: int
    tol: float
    ftol: float
    dtol: float
    solver: str = 'np'
