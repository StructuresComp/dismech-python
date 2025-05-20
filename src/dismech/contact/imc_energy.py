import numpy as np
from typing import List

from .contact_pairs import ContactPair
from .contact_energy import ContactEnergy
from .imc_helper import get_lambda_fns, delta_p_to_p, delta_p_to_e, delta_e_to_e


class IMCEnergy(ContactEnergy):

    def __init__(self, pairs: List[ContactPair], delta: float, h: float, kc: float, k_1: float = None, scale=True):
        if k_1 is None:
            k_1 = 15 / delta
        super().__init__(pairs, delta, h, k_1, kc, scale)
        self.__pp_fn, self.__grad_pp_fn, self.__hess_pp_fn = get_lambda_fns(
            delta_p_to_p)
        self.__pe_fn, self.__grad_pe_fn, self.__hess_pe_fn = get_lambda_fns(
            delta_p_to_e)
        self.__ee_fn, self.__grad_ee_fn, self.__hess_ee_fn = get_lambda_fns(
            delta_e_to_e)

    def get_Delta(self, q):
        return self._evaluate_symbolic(q, (self.__pp_fn, self.__pe_fn,  self.__ee_fn), ())

    def get_grad_hess_Delta(self, q):
        grad_Delta = self._evaluate_symbolic(
            q, (self.__grad_pp_fn, self.__grad_pe_fn, self.__grad_ee_fn), (12,))
        hess_Delta = self._evaluate_symbolic(
            q, (self.__hess_pp_fn, self.__hess_pe_fn, self.__hess_ee_fn), (12, 12,))
        return grad_Delta, hess_Delta

    def _evaluate_symbolic(self, q, fns, shape):
        t, u = self.get_lumelsky_coeff(q)
        mask = self._get_lumelsky_mask(t, u)
        ind = np.take_along_axis(self.ind, mask, axis=1)
        out = self._evalulate_piecewise(q, ind, t, u, *fns, shape)

        # Restore original order
        inv_mask = np.argsort(mask, axis=1)
        if out.ndim == 2:
            out = np.take_along_axis(out, inv_mask, axis=1)
        elif out.ndim == 3:
            out = np.take_along_axis(out, inv_mask[:, :, None], axis=1)
            out = np.take_along_axis(out, inv_mask[:, None, :], axis=2)
        return out

    def get_lumelsky_coeff(self, q):
        x, y, a, b = q[self.ind].reshape(-1, 4, 3).transpose(1, 0, 2)

        ei = y - x
        ej = b - a
        eij = a - x

        D1 = np.sum(ei * ei, axis=1)
        D2 = np.sum(ej * ej, axis=1)
        R = np.sum(ei * ej, axis=1)
        S1 = np.sum(ei * eij, axis=1)
        S2 = np.sum(ej * eij, axis=1)

        den = D1 * D2 - R ** 2
        epsilon = 1e-12  # Threshold for detecting parallelism
        is_parallel = np.abs(den) < epsilon

        # Regular case
        t = np.divide((S1 * D2 - S2 * R), den, where=~is_parallel)
        t = np.clip(t, 0, 1.0)
        u = (t * R - S2) / D2
        uf = np.clip(u, 0, 1.0)
        t = np.where(uf != u, (uf * R + S1) / D1, t)
        t = np.clip(t, 0, 1.0)

        # Parallel case: force midpoint projection
        #t_mid = 0.5 * np.ones_like(t)
        #u_mid = 0.5 * np.ones_like(uf)

        # Use midpoints where parallel
        #t = np.where(is_parallel, t_mid, t)
        #uf = np.where(is_parallel, u_mid, uf)

        return t, uf

    def _get_lumelsky_mask(self, t, u):
        # Define reordering patterns
        ORD0 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # (x, y, a, b)
        ORD1 = [3, 4, 5, 0, 1, 2, 6, 7, 8, 9, 10, 11]  # (y, x, a, b)
        ORD2 = [0, 1, 2, 3, 4, 5, 9, 10, 11, 6, 7, 8]  # (x, y, b, a)
        ORD3 = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]  # (y, x, b, a)
        ORD4 = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]  # (a, b, x, y)
        ORD5 = [9, 10, 11, 6, 7, 8, 0, 1, 2, 3, 4, 5]  # (b, a, x, y)

        mask_choices = np.stack([ORD0, ORD1, ORD2, ORD3, ORD4, ORD5])
        mask_idx = np.empty_like(t, dtype=np.int32)

        # bitwise for indexing
        is_0_or_1_t = np.isin(t, [0, 1])
        is_0_or_1_u = np.isin(u, [0, 1])

        # Point-to-point (p2p)
        mask_idx[(t == 1) & (u == 0)] = 1
        mask_idx[(t == 0) & (u == 1)] = 2
        mask_idx[(t == 1) & (u == 1)] = 3
        mask_idx[(t == 0) & (u == 0)] = 0

        # Point-to-edge (p2e)
        mask_idx[(t == 0) & (~is_0_or_1_u)] = 0
        mask_idx[(t == 1) & (~is_0_or_1_u)] = 1
        mask_idx[(~is_0_or_1_t) & (u == 0)] = 4
        mask_idx[(~is_0_or_1_t) & (u == 1)] = 5

        # Edge-to-edge (e2e)
        mask_idx[(~is_0_or_1_t) & (~is_0_or_1_u)] = 0

        return mask_choices[mask_idx]

    def _evalulate_piecewise(self, q, ind, t, u, fn_p2p, fn_p2e, fn_e2e, shape=(12,)):
        result = np.zeros((ind.shape[0],) + shape)

        # Classify types
        is_int_t = np.isin(t, [0, 1])
        is_int_u = np.isin(u, [0, 1])

        mask_p2p = is_int_t & is_int_u
        mask_p2e = is_int_t ^ is_int_u
        mask_e2e = ~is_int_t & ~is_int_u

        # Helper to extract input arrays from q
        def get_inputs(mask):
            idx = ind[mask]
            inputs = q[idx]
            return [inputs[:, i] for i in range(inputs.shape[1])]

        # Process p2p
        if np.any(mask_p2p):
            args = get_inputs(mask_p2p)
            result[mask_p2p] = fn_p2p(*args)

        # Process p2e
        if np.any(mask_p2e):
            args = get_inputs(mask_p2e)
            result[mask_p2e] = fn_p2e(*args)

        # Process e2e
        if np.any(mask_e2e):
            args = get_inputs(mask_e2e)
            result[mask_e2e] = fn_e2e(*args)

        return result
