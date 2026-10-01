"""
Ribbon energy with Sano's regularized bend-twist coupling on rod triplets.

Constitutive law
----------------
A narrow ribbon is a rod with a hard (1) and an easy (2) bending axis whose
easy-axis bending and twist are coupled (Sadowsky). Sano regularizes the
coupling with a physical length zeta so that it stays bounded as the easy
curvature vanishes. Per triplet, with the DER integrated strains relative to
the natural state

    k1 = kappa1 - kappa1_bar,  k2 = kappa2 - kappa2_bar,  t = m - m_bar,

the energy is the discrete Kirchhoff energy plus the coupling term

    E = 1/2 EI1/l k1^2 + 1/2 EI2/l k2^2 + 1/2 GJ/l t^2
      + 1/2 EI2/l t^4 / ((l/zeta)^2 + k2^2)

with l the Voronoi length. The strains are integrated (unitless), so c =
(l/zeta)^2 is unitless for zeta a length and the scaling keeps the coupling
consistent across mesh resolutions. zeta -> 0 recovers the Kirchhoff rod.

Stretch stays with the rod's StretchSprings (it is linear and lives on edges).
The coupling makes the energy non-convex in k2 where 3 k2^2 < c, so Newton
steps may need a line search.
"""
import numpy as np

from .triplet_energy import TripletEnergy


def sano_coupling(t, k2, c):
    """f = t^4 / (c + k2^2) with first (f_t, f_k) and second (f_tt, f_tk, f_kk) derivatives.

    c = inf gives f = 0 with all derivatives zero (Kirchhoff limit)."""
    D = c + k2 ** 2
    t2, t3 = t ** 2, t ** 3
    t4 = t2 ** 2
    f = t4 / D
    f_t = 4.0 * t3 / D
    f_k = -2.0 * t4 * k2 / D ** 2
    f_tt = 12.0 * t2 / D
    f_tk = -8.0 * t3 * k2 / D ** 2
    f_kk = 2.0 * t4 * (4.0 * k2 ** 2 / D - 1.0) / D ** 2   # = 2 t^4 (3 k2^2 - c) / D^3
    return f, f_t, f_k, f_tt, f_tk, f_kk


class SanoEnergy(TripletEnergy):
    STRAINS = ('kappa1', 'kappa2', 'twist')

    def _regularization(self) -> np.ndarray:
        """Denominator offset c of the coupling term, (l / zeta)^2."""
        s = self._springs
        with np.errstate(divide='ignore'):
            return (s.voronoi_len / s.zeta) ** 2

    def _energy_derivatives(self, del_strain: np.ndarray):
        """E, dE/ds (N, 3) and d2E/ds2 (N, 3, 3) for s = [kappa1, kappa2, m]."""
        s = self._springs
        lv = s.voronoi_len
        B1, B2, C = s.EI[:, 0] / lv, s.EI[:, 1] / lv, s.GJ / lv

        k1, k2, t = del_strain[:, 0], del_strain[:, 1], del_strain[:, 2]
        f, f_t, f_k, f_tt, f_tk, f_kk = sano_coupling(t, k2, self._regularization())

        E = 0.5 * (B1 * k1 ** 2 + B2 * k2 ** 2 + C * t ** 2) + 0.5 * B2 * f

        grad = np.stack([B1 * k1,
                         B2 * k2 + 0.5 * B2 * f_k,
                         C * t + 0.5 * B2 * f_t], axis=1)

        hess = np.zeros((del_strain.shape[0], 3, 3))
        hess[:, 0, 0] = B1
        hess[:, 1, 1] = B2 + 0.5 * B2 * f_kk
        hess[:, 2, 2] = C + 0.5 * B2 * f_tt
        hess[:, 1, 2] = hess[:, 2, 1] = 0.5 * B2 * f_tk
        return E, grad, hess
