import numpy as np
from typing import List

from .contact_pairs import ContactPair
from .imc_energy import IMCEnergy
from .imc_friction_helper import generate_velocity_jacobian_funcs


class IMCFrictionEnergy(IMCEnergy):
    """Coulomb-like friction for IMC contact, using precomputed contact forces."""

    def __init__(self,
                 pairs: List[ContactPair],
                 delta: float,
                 h: float,
                 dt: float,
                 vel_tol: float,
                 mu: float,
                 kc: float,
                 k_1: float = None,
                 scale: bool = True):
        # IMCEnergy(pairs, delta, h, kc, k_1=None, scale=True)
        super().__init__(pairs, delta, h, kc, k_1, scale)
        self.dt = float(dt)
        # vel_tol and mu can be scalar or per-pair
        self.vel_tol = float(vel_tol)
        # mu per pair from ContactPair
        self.mu = float(mu)
        # Jacobian generators (not used in grad_friction, but you already had them)
        self.dfr_dv_stick, self.dfr_df_stick, \
            self.dfr_dv_slide, self.dfr_df_slide = generate_velocity_jacobian_funcs()

    # ------------------------------------------------------------------
    # FRICTION GRADIENT (actually the friction force)
    # ------------------------------------------------------------------
    def grad_friction(self, state, robot, F_total, first_iter):
        """
        Compute IMC friction forces.

        - Recomputes normal contact forces via super().grad_hess_energy(...)
          (this also initializes/updates self.ind and self.pairs).
        - Uses scalar self.mu and self.vel_tol for all contact pairs.
        """

        # 1) Recompute normal contact forces + ensure self.ind is set
        contact_force, _ = super().grad_hess_energy(state, robot, F_total, first_iter)

        q = state.q
        u = state.u
        ndof = q.shape[0]

        # If no active contacts, nothing to do
        if self.ind.size == 0:
            return np.zeros_like(q)

        n_pairs = self.ind.shape[0]

        # ------------------------------------------------------------------
        # 2) Gather nodal velocities and normal contact forces
        # ------------------------------------------------------------------
        # self.ind: (n_pairs, 12) for 4 nodes × 3 dof each
        v_nodes = u[self.ind].reshape(n_pairs, 4, 3)
        v1s = v_nodes[:, 0, :]   # body 1, start node
        v1e = v_nodes[:, 1, :]   # body 1, end node
        v2s = v_nodes[:, 2, :]   # body 2, start node
        v2e = v_nodes[:, 3, :]   # body 2, end node

        f_nodes = contact_force[self.ind].reshape(n_pairs, 4, 3)
        f1s = f_nodes[:, 0, :]
        f1e = f_nodes[:, 1, :]
        f2s = f_nodes[:, 2, :]
        f2e = f_nodes[:, 3, :]

        # Norms of nodal normal forces
        f1s_n = np.linalg.norm(f1s, axis=1)   # (n_pairs,)
        f1e_n = np.linalg.norm(f1e, axis=1)
        f2s_n = np.linalg.norm(f2s, axis=1)
        f2e_n = np.linalg.norm(f2e, axis=1)

        # Combined normal force on body 1 (for fn and normal direction)
        f1_sum = f1s + f1e                     # (n_pairs, 3)
        fn = np.linalg.norm(f1_sum, axis=1)    # (n_pairs,)
        eps = 1e-14
        nonzero_fn = fn > eps

        # ------------------------------------------------------------------
        # 3) Contact interpolation weights t1, u1 (like MATLAB)
        #    t1 = ||f1s|| / ||f1s + f1e||
        #    u1 = ||f2s|| / ||f1s + f1e||
        # ------------------------------------------------------------------
        t1 = np.zeros_like(fn)
        u1 = np.zeros_like(fn)
        np.divide(f1s_n, fn, out=t1, where=nonzero_fn)
        np.divide(f2s_n, fn, out=u1, where=nonzero_fn)

        t1 = np.clip(t1, 0.0, 1.0)
        u1 = np.clip(u1, 0.0, 1.0)
        t2 = 1.0 - t1
        u2 = 1.0 - u1

        # ------------------------------------------------------------------
        # 4) Relative tangential velocity at the contact point
        # ------------------------------------------------------------------
        v1 = t1[:, None] * v1s + t2[:, None] * v1e   # body 1 vel at contact
        v2 = u1[:, None] * v2s + u2[:, None] * v2e   # body 2 vel at contact
        v_rel = v1 - v2                              # (n_pairs, 3)

        # Contact normal = (f1s + f1e) / ||f1s + f1e||
        contact_norm = np.zeros_like(v_rel)
        contact_norm[nonzero_fn] = f1_sum[nonzero_fn] / fn[nonzero_fn, None]

        # Tangential relative velocity: tv_rel = v_rel - (v_rel·n) n
        v_rel_dot_n = np.sum(v_rel * contact_norm, axis=1)     # (n_pairs,)
        tv_rel = v_rel - v_rel_dot_n[:, None] * contact_norm   # (n_pairs, 3)
        tv_rel_n = np.linalg.norm(tv_rel, axis=1)              # (n_pairs,)

        # ------------------------------------------------------------------
        # 5) Gamma logic with scalar mu, vel_tol (ZeroVel / Sliding / Sticking)
        # ------------------------------------------------------------------
        mu = float(self.mu)
        vel_tol = float(self.vel_tol)
        K2 = 15.0 / vel_tol

        gamma = np.zeros(n_pairs)
        moving = tv_rel_n > eps

        # Sliding: tv_rel_n > vel_tol → gamma = 1
        sliding = moving & (tv_rel_n > vel_tol)
        gamma[sliding] = 1.0

        # Sticking: 0 < tv_rel_n <= vel_tol → smooth transition
        sticking = moving & (tv_rel_n <= vel_tol)
        if np.any(sticking):
            x_stick = tv_rel_n[sticking]
            gamma[sticking] = 2.0 / (1.0 + np.exp(-K2 * x_stick)) - 1.0

        # Unit tangential direction
        tv_rel_u = np.zeros_like(tv_rel)
        nonzero_tv = tv_rel_n > eps
        tv_rel_u[nonzero_tv] = tv_rel[nonzero_tv] / tv_rel_n[nonzero_tv, None]

        # ------------------------------------------------------------------
        # 6) Friction force at contact point and distribution to nodes
        # ------------------------------------------------------------------
        # ffr_val = mu * gamma * tv_rel_u  (vector, per contact pair)
        ffr_scale = (mu * gamma)[:, None]     # (n_pairs, 1)
        ffr_val = ffr_scale * tv_rel_u        # (n_pairs, 3)

        # Nodal friction forces:
        #   body 1 nodes: +ffr_val * |f_i|
        #   body 2 nodes: -ffr_val * |f_i|
        ffr_nodes = np.zeros_like(f_nodes)    # (n_pairs, 4, 3)
        ffr_nodes[:, 0, :] = ffr_val * f1s_n[:, None]
        ffr_nodes[:, 1, :] = ffr_val * f1e_n[:, None]
        ffr_nodes[:, 2, :] = -ffr_val * f2s_n[:, None]
        ffr_nodes[:, 3, :] = -ffr_val * f2e_n[:, None]

        # Flatten and assemble into global DOF vector
        ffr_flat = ffr_nodes.reshape(n_pairs, 12)

        F_fric = np.zeros(ndof)
        np.add.at(F_fric, self.ind, ffr_flat)

        return F_fric