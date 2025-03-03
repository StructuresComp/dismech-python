import typing
import numpy as np

from .params import GeomParams, Material


def compute_rod_stiffness(geom: GeomParams, material: Material) -> typing.Tuple[float, ...]:
    EA = material.youngs_rod * (geom.axs if geom.axs is not None else np.pi * geom.rod_r0 ** 2)

    if geom.ixs1 and geom.ixs2:
        EI1 = material.youngs_rod * geom.ixs1
        EI2 = material.youngs_rod * geom.ixs2
    else:
        EI1 = EI2 = material.youngs_rod * np.pi * geom.rod_r0 ** 4 / 4

    GJ = material.youngs_rod / \
        (2 * (1 + material.poisson_rod)) * \
        (geom.jxs if geom.jxs else np.pi * geom.rod_r0 ** 4 / 2)

    return EA, EI1, EI2, GJ


def compute_shell_stiffness(geom: GeomParams, material: Material, ref_len: np.ndarray, use_mid_edge: bool) -> typing.Tuple[float, float]:#
    if use_mid_edge:
        ks = 2 * material.youngs_shell * geom.shell_h / \
            (1 - material.poisson_shell**2) * ref_len
        kb = material.youngs_shell * geom.shell_h**3 / \
            (24 * (1 - material.poisson_shell**2))
    else:
        ks = (3**0.5 / 2) * material.youngs_shell * geom.shell_h * ref_len
        kb = (2 / 3**0.5) * material.youngs_shell * (geom.shell_h**3) / 12
    return ks, kb
