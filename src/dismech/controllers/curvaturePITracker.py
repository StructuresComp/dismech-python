import numpy as np

class CurvaturePITracker:
    """
    PI trajectory-tracking controller for natural curvature (bending).

    Provide either:
      - ref_func(t) -> (Nb,2), or
      - (ref_times (T,), ref_values (T,Nb,2)) schedule.

    Writes to nat_arr (typically robot.bend_springs.inc_strain).
    """

    def __init__(self, nat0,
                ref_func=None, ref_times=None, ref_values=None,
                Kp=0.7, Ki=0.04, Kd=0.0, beta=0.6,
                du_rate_per_step=0.10,     # per-step |Δ| limit (per row)
                smooth_alpha=0.20,         # spatial Laplacian smoothing
                kappa_mag_max=np.inf,      # absolute |nat| cap
                antiwindup_bc=0.9
    ):
        assert (ref_func is not None) or (ref_times is not None and ref_values is not None)
        self.ref_func = ref_func
        self.ref_times = None if ref_times is None else np.asarray(ref_times, float)
        self.ref_values = None if ref_values is None else np.asarray(ref_values, float)

        self.nat = np.asarray(nat0, float).copy()
        self.Kp, self.Ki, self.Kd = float(Kp), float(Ki), float(Kd)
        self.beta = float(beta)

        self.du_rate = float(du_rate_per_step)
        self.smooth_alpha = float(smooth_alpha)
        self.kappa_mag_max = float(kappa_mag_max)
        self.antiwindup_bc = float(antiwindup_bc)

        self.I = np.zeros_like(self.nat)
        self.prev_meas = None
        self.ref_filt = None

        self.ref_t0 = float(self.ref_times[0])
        self.ref_N  = int(self.ref_values.shape[0])
        self.ref_dt = float(np.round(self.ref_times[1] - self.ref_times[0], 12))

    @staticmethod
    def _laplacian_1d_matrix(X, a):
        if a <= 0: return X
        Y = X.copy()
        Y[1:-1,:] = X[1:-1,:] + a*(X[0:-2,:] - 2.0*X[1:-1,:] + X[2:,:])
        Y[0,:]    = X[0,:]    + a*(X[1,:] - X[0,:])
        Y[-1,:]   = X[-1,:]   + a*(X[-2,:] - X[-1,:])
        return Y

    @staticmethod
    def _vector_rate_limit(new_vecs, old_vecs, du_max):
        d = new_vecs - old_vecs
        n = np.linalg.norm(d, axis=1, keepdims=True)
        scale = np.minimum(1.0, du_max/np.maximum(n, 1e-12))
        return old_vecs + d*scale

    def _clip_mag(self, X, m):
        if not np.isfinite(m): return X
        mag = np.linalg.norm(X, axis=1, keepdims=True)
        scale = np.minimum(1.0, m/np.maximum(mag, 1e-12))
        return X*scale

    def _get_ref(self, t):
        k = int(np.round((float(t) - self.ref_t0)/self.ref_dt + 1e-12))
        k = 0 if k < 0 else (self.ref_N - 1 if k >= self.ref_N else k)

        return self.ref_values[k].astype(float, copy=False)

    def update(self, t, kappa_meas, nat_arr, dt):
        kappa_meas = np.asarray(kappa_meas, float)

        # reference
        self.ref_filt = self._get_ref(t)

        # PI(D) with setpoint weighting (proportional on β*r - y, integral on r - y)
        e = self.ref_filt - kappa_meas
        self.I += e * dt

        if self.prev_meas is None:
            d_meas = np.zeros_like(e)
        else:
            d_meas = (kappa_meas - self.prev_meas)/max(dt, 1e-12)
        self.prev_meas = kappa_meas.copy()

        # feed-forward = reference (since the actuator is natural curvature)
        u_ff  = self.ref_filt
        u_P   = self.Kp*(self.beta*self.ref_filt - kappa_meas)
        u_I   = self.Ki*self.I
        u_D   = -self.Kd*d_meas

        nat_prop = u_ff + u_P + u_I + u_D
        nat_prop = self._laplacian_1d_matrix(nat_prop, self.smooth_alpha)
        nat_prop = self._clip_mag(nat_prop, self.kappa_mag_max)
        nat_new = self._vector_rate_limit(nat_prop, nat_arr, self.du_rate)

        # anti-windup if we clipped
        if np.linalg.norm(nat_prop - nat_new) > 0:
            self.I = self.antiwindup_bc*self.I

        nat_arr[...] = nat_new
        self.nat = nat_new.copy()
        return nat_arr

    