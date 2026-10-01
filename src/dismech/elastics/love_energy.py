"""
Helical coil spring energy (Love, Treatise §271) on rod triplets.

Constitutive law
----------------
A coil of rest radius r and pitch angle alpha, made of a round wire with bending
stiffness B and torsional stiffness C, loaded by a wrench about its axis stays a
uniform helix. Its energy per unit rest axial length is exact in closed form in
terms of the coil's axial stretch eps and twist per rest axial length vtheta:

    w(eps, vtheta) = (1/sin a) [ B/2 (k cos a1 - kappa0)^2 + C/2 (k sin a1 - tau0)^2 ]
    sin a1 = (1 + eps) sin a,   k = cos(a)/r + vtheta sin(a)
    kappa0 = cos^2(a)/r,        tau0 = sin(a) cos(a)/r

with axial force R = dw/deps and axial torque K = dw/dvtheta. The law is valid
while the coil stays a helix: (1 + eps) sin a < 1 (not pulled straight) and no
coil contact.

Discretization
--------------
The rod carries the coil axis. Twist lives on triplets (node between edges e, f)
but stretch lives on edges, and a Voronoi-averaged stretch alone would admit a
zero-energy zig-zag mode. The energy is therefore split exactly as

    w(eps, vtheta) = w(eps, 0) + [w(eps, vtheta) - w(eps, 0)]

with the pure stretch part on each edge (full edge length, like StretchEnergy)
and the coupled remainder on each triplet (Voronoi length, like TwistEnergy):

    E = sum_x  sw_x w(eps_x, 0)  +  l_v [w(eps_bar, vtheta) - w(eps_bar, 0)]

where x in {e, f}, sw_x is the share of edge x carried by this spring,
eps_bar = (|e| + |f|)/(l_e + l_f) - 1 and vtheta = chirality * dm / l_v with dm
the DER twist relative to rest. A uniform deformation gives exactly L w(eps, vtheta).
"""
import numpy as np

from .triplet_energy import TripletEnergy
from ..springs.love_spring import LoveSprings
from ..state import RobotState


def love_constitutive(eps, vtheta, r, alpha, B, C):
    """Energy density w, stress (R, K), and tangent (T_ee, T_ev, T_vv) of Love's law."""
    sa = np.sin(alpha)
    s1 = (1.0 + eps) * sa
    c1 = np.sqrt(1.0 - s1 ** 2)
    k = np.cos(alpha) / r + vtheta * sa
    kappa0 = np.cos(alpha) ** 2 / r
    tau0 = sa * np.cos(alpha) / r

    dk = k * c1 - kappa0     # bending strain of the wire
    dt = k * s1 - tau0       # twist strain of the wire
    w = (0.5 * B * dk ** 2 + 0.5 * C * dt ** 2) / sa

    # d/deps acts through (s1, c1), d/dvtheta through k
    dc1 = -s1 * sa / c1
    g = -s1 / c1
    dg = -(sa * c1 - s1 * dc1) / c1 ** 2
    R = B * dk * k * g + C * dt * k
    K = B * dk * c1 + C * dt * s1

    d_dk_de, d_dk_dv = k * dc1, c1 * sa
    d_dt_de, d_dt_dv = k * sa, s1 * sa
    T_ee = B * (d_dk_de * k * g + dk * k * dg) + C * d_dt_de * k
    T_ev = B * (d_dk_de * c1 + dk * dc1) + C * (d_dt_de * s1 + dt * sa)
    T_vv = B * d_dk_dv * c1 + C * d_dt_dv * s1
    return w, R, K, T_ee, T_ev, T_vv


class LoveEnergy(TripletEnergy):
    STRAINS = ('eps_e', 'eps_f', 'twist')

    def __init__(self, springs: LoveSprings, initial_state: RobotState):
        lv = springs.voronoi_len
        self._a = springs.ref_len / (2.0 * lv[:, None])   # eps_bar = a_e eps_e + a_f eps_f
        super().__init__(springs, initial_state)

    def _energy_derivatives(self, del_strain: np.ndarray):
        """E, dE/ds (N, 3) and d2E/ds2 (N, 3, 3) for s = [eps_e, eps_f, m]."""
        s = self._springs
        r, alpha, B, C = s.coil_r, s.pitch, s.EI, s.GJ
        lv, sw, a = s.voronoi_len, s.stretch_w, self._a
        chi = s.chirality

        eps_e, eps_f, dm = del_strain[:, 0], del_strain[:, 1], del_strain[:, 2]
        eps_bar = a[:, 0] * eps_e + a[:, 1] * eps_f
        vth = chi * dm / lv
        zero = np.zeros_like(vth)

        w_e, R_e, _, Tee_e, _, _ = love_constitutive(eps_e, zero, r, alpha, B, C)
        w_f, R_f, _, Tee_f, _, _ = love_constitutive(eps_f, zero, r, alpha, B, C)
        w_c, R_c, K_c, Tee_c, Tev_c, Tvv_c = love_constitutive(eps_bar, vth, r, alpha, B, C)
        w_0, R_0, _, Tee_0, _, _ = love_constitutive(eps_bar, zero, r, alpha, B, C)

        E = sw[:, 0] * w_e + sw[:, 1] * w_f + lv * (w_c - w_0)

        dRc = R_c - R_0
        grad = np.stack([sw[:, 0] * R_e + lv * a[:, 0] * dRc,
                         sw[:, 1] * R_f + lv * a[:, 1] * dRc,
                         chi * K_c], axis=1)

        dTc = Tee_c - Tee_0
        hess = np.empty((del_strain.shape[0], 3, 3))
        hess[:, 0, 0] = sw[:, 0] * Tee_e + lv * a[:, 0] ** 2 * dTc
        hess[:, 1, 1] = sw[:, 1] * Tee_f + lv * a[:, 1] ** 2 * dTc
        hess[:, 0, 1] = hess[:, 1, 0] = lv * a[:, 0] * a[:, 1] * dTc
        hess[:, 0, 2] = hess[:, 2, 0] = chi * a[:, 0] * Tev_c
        hess[:, 1, 2] = hess[:, 2, 1] = chi * a[:, 1] * Tev_c
        hess[:, 2, 2] = Tvv_c / lv
        return E, grad, hess
