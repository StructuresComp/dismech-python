import numpy as np

class CurvaturePI:
    """
    PI controller for bending natural strain (curvature).
    Shapes:
      kappa_ref : (Nb, 2)
      nat0      : (Nb, 2)
      kappa_meas: (Nb, 2)
      nat_arr   : (Nb, 2)  # updated in-place
    """
    def __init__(self, kappa_ref, nat0,
                 Kp=0.8, Ki=0.03, Kd=0.0,       # Kd usually 0; see notes
                 du_rate=0.10,                  # max per-step *magnitude* change of natural curvature
                 smooth_alpha=0.20,             # Laplacian smoothing along springs
                 kappa_max=np.inf):             # optional magnitude cap on nat curvature
        self.kappa_ref = np.asarray(kappa_ref, dtype=np.float64)  # (Nb,2)
        self.nat       = np.asarray(nat0,      dtype=np.float64).copy()  # (Nb,2)
        self.I         = np.zeros_like(self.kappa_ref)             # integral memory (Nb,2)
        self.prev_meas = None
        self.Kp, self.Ki, self.Kd = float(Kp), float(Ki), float(Kd)
        self.du_rate = float(du_rate)
        self.smooth_alpha = float(smooth_alpha)
        self.kappa_max = float(kappa_max)

    @staticmethod
    def _laplacian_1d_matrix(X, alpha):
        """Apply 1D Laplacian smoothing along the first axis to each column (component-wise)."""
        if alpha <= 0: return X
        Y = X.copy()
        Y[1:-1, :] = X[1:-1, :] + alpha*(X[0:-2, :] - 2.0*X[1:-1, :] + X[2:, :])
        Y[0,  :]   = X[0,  :]   + alpha*(X[1, :] - X[0, :])
        Y[-1, :]   = X[-1, :]   + alpha*(X[-2, :] - X[-1, :])
        return Y

    @staticmethod
    def _vector_rate_limit(new_vecs, old_vecs, du_max):
        """
        Rate limit by vector *magnitude* per row:
          ||new - old|| <= du_max
        Keeps direction of the proposed change.
        """
        d = new_vecs - old_vecs                 # (Nb,2)
        norms = np.linalg.norm(d, axis=1, keepdims=True)  # (Nb,1)
        scale = np.minimum(1.0, du_max/np.maximum(norms, 1e-12))
        return old_vecs + d*scale

    def _clip_magnitude(self, X, mag_max):
        if not np.isfinite(mag_max): 
            return X
        mags = np.linalg.norm(X, axis=1, keepdims=True)
        scale = np.minimum(1.0, mag_max/np.maximum(mags, 1e-12))
        return X*scale

    def update(self, kappa_meas, nat_arr, dt):
        """
        Args:
            kappa_meas: (Nb,2) measured curvature (from BendEnergy.get_strain)
            nat_arr   : (Nb,2) natural curvature array to be updated *in-place*
            dt        : timestep
        Returns:
            nat_arr (same object), updated in-place.
        """
        kappa_meas = np.asarray(kappa_meas, dtype=np.float64)

        # PI(D) on curvature error
        e = self.kappa_ref - kappa_meas                # (Nb,2)
        self.I += e*dt

        if self.prev_meas is None:
            d_meas = np.zeros_like(e)
        else:
            d_meas = (kappa_meas - self.prev_meas)/max(dt, 1e-12)
        self.prev_meas = kappa_meas.copy()

        delta = self.Kp*e + self.Ki*self.I - self.Kd*d_meas   # (Nb,2)
        nat_prop = nat_arr + delta

        # spatial smoothing (component-wise)
        nat_prop = self._laplacian_1d_matrix(nat_prop, self.smooth_alpha)

        # optional absolute magnitude bound on natural curvature
        nat_prop = self._clip_magnitude(nat_prop, self.kappa_max)

        # vector-norm rate limit per spring
        nat_new = self._vector_rate_limit(nat_prop, nat_arr, self.du_rate)

        # anti-windup: if limited, relax the integral a bit
        if np.linalg.norm(nat_prop - nat_new) > 0:
            self.I *= 0.9

        # write back in-place and store
        nat_arr[...] = nat_new
        self.nat = nat_arr.copy()
        return nat_arr
