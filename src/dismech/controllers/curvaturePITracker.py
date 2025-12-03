import numpy as np

class CurvaturePITracker:
    """
    PI trajectory-tracking controller for natural curvature (bending).

    You can provide the reference in ONE of two ways:

    1) Functional reference:
        ref_func(t) -> array of shape (Nb, 2),
        where Nb is the number of bend springs.

    2) Scheduled reference:
        ref_times:  array of shape (T,), times
        ref_values: array of shape (T, Nb, 2), reference κ(t_k)

    The controller writes the new natural curvature into nat_arr
    (typically robot.bend_springs.nat_strain) in-place.
    """

    def __init__(self, nat0,
                 ref_func=None, ref_times=None, ref_values=None,
                 Kp=0.7, Ki=0.04, Kd=0.0, beta=0.6,
                 du_rate_per_step=0.10,     # per-step |Δu| limit (per row)
                 smooth_alpha=0.20,         # spatial Laplacian smoothing
                 kappa_mag_max=np.inf,      # absolute |nat| cap
                 antiwindup_bc=0.9):
        # --- Reference mode selection --------------------------------------
        # Must provide either a continuous function OR a discrete schedule
        assert (ref_func is not None) or (ref_times is not None and ref_values is not None), \
            "Provide either ref_func or (ref_times, ref_values)."

        # Mode 1: functional reference
        self.ref_func = ref_func

        # Mode 2: scheduled reference
        if ref_times is not None and ref_values is not None:
            self.ref_times  = np.asarray(ref_times, float)          # (T,)
            self.ref_values = np.asarray(ref_values, float)         # (T, Nb, 2)
            self.ref_t0 = float(self.ref_times[0])
            self.ref_N  = int(self.ref_values.shape[0])
            # assume (approximately) uniform grid; used for index lookup
            self.ref_dt = float(np.round(self.ref_times[1] - self.ref_times[0], 12))
        else:
            # In function mode we do not use these
            self.ref_times  = None
            self.ref_values = None
            self.ref_t0 = None
            self.ref_N  = None
            self.ref_dt = None

        # --- Controller gains & limits -------------------------------------
        self.nat = np.asarray(nat0, float).copy()    # current natural curvature (Nb,2)
        self.Kp, self.Ki, self.Kd = float(Kp), float(Ki), float(Kd)
        self.beta = float(beta)

        self.du_rate = float(du_rate_per_step)
        self.smooth_alpha = float(smooth_alpha)
        self.kappa_mag_max = float(kappa_mag_max)
        self.antiwindup_bc = float(antiwindup_bc)

        # --- Internal state -------------------------------------------------
        self.I = np.zeros_like(self.nat)   # integral term
        self.prev_meas = None             # previous measurement for D-term
        self.ref_filt = None              # last reference used (for logging/debug)

    # ----------------------------------------------------------------------
    # Utility methods
    # ----------------------------------------------------------------------
    @staticmethod
    def _laplacian_1d_matrix(X, a):
        """
        Simple 1D Laplacian smoother along the "bends" index (rows of X).
        X: (Nb, 2)
        a: smoothing coefficient; if <=0, returns X unchanged.
        """
        if a <= 0:
            return X
        Y = X.copy()
        # interior points: 1D Laplacian with Neumann-like ends
        Y[1:-1, :] = X[1:-1, :] + a * (X[0:-2, :] - 2.0 * X[1:-1, :] + X[2:, :])
        # simple one-sided at boundaries
        Y[0, :]    = X[0, :]    + a * (X[1, :] - X[0, :])
        Y[-1, :]   = X[-1, :]   + a * (X[-2, :] - X[-1, :])
        return Y

    @staticmethod
    def _vector_rate_limit(new_vecs, old_vecs, du_max):
        """
        Rate limiter: for each row i, limit ||new_vecs[i] - old_vecs[i]||
        to at most du_max. Works row-wise.

        new_vecs, old_vecs: (Nb,2)
        du_max: scalar
        """
        d = new_vecs - old_vecs                  # (Nb,2)
        n = np.linalg.norm(d, axis=1, keepdims=True)  # (Nb,1)
        scale = np.minimum(1.0, du_max / np.maximum(n, 1e-12))
        return old_vecs + d * scale

    def _clip_mag(self, X, m):
        """
        Clip the magnitude of each row of X to at most m.
        If m is infinite, do nothing.
        """
        if not np.isfinite(m):
            return X
        mag = np.linalg.norm(X, axis=1, keepdims=True)
        scale = np.minimum(1.0, m / np.maximum(mag, 1e-12))
        return X * scale

    # ----------------------------------------------------------------------
    # Reference handling
    # ----------------------------------------------------------------------
    def _get_ref(self, t):
        """
        Return reference curvature κ_ref at time t, shape (Nb,2).

        If ref_func was provided, we use it directly.
        Else, we use the tabulated schedule (ref_times, ref_values)
        with nearest-index lookup.
        """
        # Functional mode
        if self.ref_func is not None:
            ref = self.ref_func(float(t))        # user-supplied function
            ref = np.asarray(ref, float)
            return ref

        # Scheduled mode
        # Map time t to an index k on the reference grid
        k = int(np.round((float(t) - self.ref_t0) / self.ref_dt + 1e-12))
        if k < 0:
            k = 0
        elif k >= self.ref_N:
            k = self.ref_N - 1

        # ref_values[k] is already (Nb,2)
        return self.ref_values[k].astype(float, copy=False)

    # ----------------------------------------------------------------------
    # Main update
    # ----------------------------------------------------------------------
    def update(self, t, kappa_meas, nat_arr, dt):
        """
        Main PI(D) update.

        Inputs:
          t          : current time (float)
          kappa_meas : measured curvature array, shape (Nb,2)
          nat_arr    : natural curvature array to be updated in-place (Nb,2)
          dt         : time step (float)

        Returns:
          nat_arr (same object, updated)
        """
        kappa_meas = np.asarray(kappa_meas, float)

        # --- Reference -----------------------------------------------------
        self.ref_filt = self._get_ref(t)   # (Nb,2)

        # --- Error and integral term --------------------------------------
        # e = r - y
        e = self.ref_filt - kappa_meas
        self.I += e * dt

        # --- Derivative on measurement ------------------------------------
        if self.prev_meas is None:
            d_meas = np.zeros_like(e)
        else:
            d_meas = (kappa_meas - self.prev_meas) / max(dt, 1e-12)
        self.prev_meas = kappa_meas.copy()

        # --- PI(D) with setpoint weighting --------------------------------
        # feed-forward = reference (actuator is natural curvature)
        u_ff = self.ref_filt
        # proportional on β*r - y
        u_P  = self.Kp * (self.beta * self.ref_filt - kappa_meas)
        # integral on e
        u_I  = self.Ki * self.I
        # derivative on measurement
        u_D  = -self.Kd * d_meas

        nat_prop = u_ff + u_P + u_I + u_D

        # --- Spatial smoothing (optional) ---------------------------------
        nat_prop = self._laplacian_1d_matrix(nat_prop, self.smooth_alpha)

        # --- Magnitude clipping -------------------------------------------
        nat_prop = self._clip_mag(nat_prop, self.kappa_mag_max)

        # --- Rate limiting -----------------------------------------------
        nat_new = self._vector_rate_limit(nat_prop, nat_arr, self.du_rate)

        # --- Anti-windup: if we had to clip/rate-limit, shrink I ----------
        if np.linalg.norm(nat_prop - nat_new) > 0:
            self.I = self.antiwindup_bc * self.I

        # --- Write result back --------------------------------------------
        nat_arr[...] = nat_new       # in-place update of robot.bend_springs.nat_strain
        self.nat = nat_new.copy()    # store for inspection/logging
        return nat_arr
