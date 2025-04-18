import numpy as np

from .contact_energy import ContactEnergy
from .imc_helper import get_lambda_fns, delta_p_to_p, delta_p_to_e, delta_e_to_e


class IMCEnergy(ContactEnergy):

    def __init__(self, ind: np.ndarray, delta: float, h: float, k_1: float = None):
        if k_1 is None:
            k_1 = 15 / delta
        super().__init__(delta, h, k_1)
        self.ind = ind
        self.__pp_fn, self.__grad_pp_fn, self.__hess_pp_fn = get_lambda_fns(
            delta_p_to_p)
        self.__pe_fn, self.__grad_pe_fn, self.__hess_pe_fn = get_lambda_fns(
            delta_p_to_e)
        self.__ee_fn, self.__grad_ee_fn, self.__hess_ee_fn = get_lambda_fns(
            delta_e_to_e)

    def get_Delta(self, state):
        t, u = self.get_lumelsky_coeff(state)
        reordered_ind = self._get_lumelsky_mask(t, u)
        return self._evalulate_piecewise(state,
                                         reordered_ind,
                                         t, u,
                                         self.__pp_fn,
                                         self.__pe_fn,
                                         self.__ee_fn,
                                         ())

    def get_grad_hess_Delta(self, state):
        t, u = self.get_lumelsky_coeff(state)
        reordered_ind = self._get_lumelsky_mask(t, u)
        return self._evalulate_piecewise(state,
                                         reordered_ind,
                                         t, u,
                                         self.__grad_pp_fn,
                                         self.__grad_pe_fn,
                                         self.__grad_ee_fn), \
            self._evalulate_piecewise(state,
                                      reordered_ind,
                                      t, u,
                                      self.__hess_pp_fn,
                                      self.__hess_pe_fn,
                                      self.__hess_ee_fn,
                                      (12, 12,))

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
        t = np.divide((S1 * D2 - S2 * R), den, where=den != 0)
        t = np.clip(t, 0, 1.0)

        u = (t * R - S2) / D2
        uf = np.clip(u, 0, 1.0)

        t = np.where(uf != u, (uf * R + S1) / D1, t)

        return np.clip(t, 0, 1.0), uf

    def _get_lumelsky_mask(self, t, u):
        """
        Returns reordered node index masks for contact constraints based on
        Lumelsky coefficients t and u.

        Args:
            t (np.ndarray): (N,) array of Lumelsky coefficients for edge 1.
            u (np.ndarray): (N,) array of Lumelsky coefficients for edge 2.

        Returns:
            np.ndarray: (N, 12) reordered indices for each contact pair.
        """
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

        # Apply the selected permutation mask to each row
        reorder_masks = mask_choices[mask_idx]
        return np.take_along_axis(self.ind, reorder_masks, axis=1)

    def _evalulate_piecewise(self, q, reordered_ind, t, u, fn_p2p, fn_p2e, fn_e2e, shape=(12,)):
        """
        Dispatches batched piecewise computations based on contact type.

        Args:
            q: (num_nodes, 3) positions.
            reordered_ind: (N, 12) index into q for each contact.
            t, u: (N,) Lumelsky coefficients (can be int or float).
            fn_p2p, fn_p2e, fn_e2e: callable batched lambdified functions.
            shape: result shape without batch dimensions (i.e. (12) for (N,12)).

        Returns:
            (N, ...) result with same order as input
        """
        N = reordered_ind.shape[0]
        # or shape depending on function output
        result = np.zeros((N,) + shape)

        # Classify types
        is_int_t = np.isin(t, [0, 1])
        is_int_u = np.isin(u, [0, 1])

        mask_p2p = is_int_t & is_int_u
        mask_p2e = is_int_t ^ is_int_u
        mask_e2e = ~is_int_t & ~is_int_u

        # Helper to extract input arrays from q
        def get_inputs(mask):
            idx = reordered_ind[mask]
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
