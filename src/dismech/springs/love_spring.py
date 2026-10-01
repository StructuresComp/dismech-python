import numpy as np

from .triplet_spring import TripletSprings


class LoveSprings(TripletSprings):
    """
    Helical-spring (Love, Treatise §271) elements on rod triplets (n0, e, n1, f, n2).

    Each rod segment represents a coil spring whose axis follows the rod. The
    strains are [eps_e, eps_f, m]: the stretch of both edges and the DER twist at
    the middle node. See LoveEnergy for the constitutive law.

    Material fields per spring (see TripletSprings for the common ones):
        coil_r:      rest coil radius r
        pitch:       rest pitch angle alpha [rad] (h0 = l sin alpha)
        EI, GJ:      wire bending (B) and torsional (C) stiffness
        chirality:   +1 right-handed coil, -1 left-handed
    """

    n_strain = 3

    def _material_fields(self):
        return {name: np.empty(self.N, dtype=np.float64)
                for name in ('coil_r', 'pitch', 'EI', 'GJ', 'chirality')}
