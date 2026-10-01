import numpy as np

from .triplet_spring import TripletSprings


class SanoSprings(TripletSprings):
    """
    Ribbon (Sano) elements on rod triplets (n0, e, n1, f, n2).

    The strains are [kappa1, kappa2, m]: the DER curvatures about the major
    (hard) and minor (easy) axes and the DER twist at the middle node. See
    SanoEnergy for the constitutive law.

    Material fields per spring (see TripletSprings for the common ones):
        EI:   (EI1, EI2) bending stiffness about the hard and easy axes
        GJ:   torsional stiffness
        zeta: Sano regularization length (0 recovers the Kirchhoff rod)
    """

    n_strain = 3

    def _material_fields(self):
        return {
            'EI': np.empty((self.N, 2), dtype=np.float64),
            'GJ': np.empty(self.N, dtype=np.float64),
            'zeta': np.empty(self.N, dtype=np.float64),
        }
