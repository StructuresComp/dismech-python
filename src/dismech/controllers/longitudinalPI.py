import numpy as np

class LongitudinalPI:
    def __init__(self, eps_ref, nat0,
                 Kp=0.5, Ki=0.03, Kd=0.0,
                 du_rate=0.02,          # max |Δ nat_strain| per step
                 smooth_alpha=0.10):    # Laplacian smoothing
        self.eps_ref = eps_ref.astype(np.float64)
        self.nat     = nat0.astype(np.float64).copy()    # (N_edges,)
        self.I       = np.zeros_like(eps_ref)
        self.prev_meas = None
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.du_rate = float(du_rate)
        self.smooth_alpha = float(smooth_alpha)

    @staticmethod
    def _laplacian_1d(x, a):
        if a <= 0: return x
        y = x.copy()
        y[1:-1] = x[1:-1] + a*(x[0:-2] - 2.0*x[1:-1] + x[2:])
        y[0]    = x[0]    + a*(x[1] - x[0])
        y[-1]   = x[-1]   + a*(x[-2] - x[-1])
        return y

    def update(self, eps_meas, nat_arr, dt):
        # PID on strain error
        e = self.eps_ref - eps_meas
        self.I += e*dt

        if self.prev_meas is None:
            d_meas = np.zeros_like(e)
        else:
            d_meas = (eps_meas - self.prev_meas)/max(dt,1e-12)
        self.prev_meas = eps_meas.copy()

        delta = self.Kp*e + self.Ki*self.I - self.Kd*d_meas
        nat_prop = nat_arr + delta

        # spatial smoothing + per-step rate limit
        nat_prop = self._laplacian_1d(nat_prop, self.smooth_alpha)
        dn = nat_prop - nat_arr
        clip = np.minimum(1.0, self.du_rate/np.maximum(np.abs(dn), 1e-12))
        nat_new = nat_arr + dn*clip

        # anti-windup if we clipped a lot
        if np.linalg.norm(nat_prop - nat_new) > 0:
            self.I *= 0.9

        nat_arr[...] = nat_new         # write back in-place
        self.nat = nat_arr.copy()
        return nat_arr
