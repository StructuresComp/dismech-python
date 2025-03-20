import dataclasses
import numpy as np


@dataclasses.dataclass(frozen=True)
class RobotState:
    q: np.ndarray
    u: np.ndarray
    a: np.ndarray
    a1: np.ndarray
    a2: np.ndarray
    m1: np.ndarray
    m2: np.ndarray
    ref_twist: np.ndarray
    tau: np.ndarray
    free_dof: np.ndarray

    @classmethod
    def init(cls, q0: np.ndarray, a1: np.ndarray, a2: np.ndarray, m1: np.ndarray, m2: np.ndarray, ref_twist: np.ndarray, tau: np.ndarray):
        return cls(
            q=q0.copy(),
            u=np.zeros_like(q0),
            a=np.zeros_like(q0),
            a1=a1.copy(),
            a2=a2.copy(),
            m1=m1.copy(),
            m2=m2.copy(),
            ref_twist=ref_twist.copy(),
            tau=tau.copy(),
            free_dof=np.arange(q0.size)
        )
