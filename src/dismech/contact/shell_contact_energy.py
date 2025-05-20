# # for debug
# import debugpy


import numpy as np
from typing import List

from .contact_pairs import ContactPair
from .contact_energy import ContactEnergy
from .imc_helper import get_lambda_fns_shell, delta_p_to_p_shell, delta_p_to_e_shell, delta_e_to_e_shell, delta_p_to_t_shell 


class ShellContactEnergy(ContactEnergy):

    def __init__(self, pairs: List[ContactPair], delta: float, h: float, kc: float, k_1: float = None, scale=True):
        if k_1 is None:
            k_1 = 15 / delta
        super().__init__(pairs, delta, h, k_1, kc, scale)
        self.__pp_fn, self.__grad_pp_fn, self.__hess_pp_fn = get_lambda_fns_shell(
            delta_p_to_p_shell)
        self.__pe_fn, self.__grad_pe_fn, self.__hess_pe_fn = get_lambda_fns_shell(
            delta_p_to_e_shell)
        self.__ee_fn, self.__grad_ee_fn, self.__hess_ee_fn = get_lambda_fns_shell(
            delta_e_to_e_shell)
        self.__pt_fn, self.__grad_pt_fn, self.__hess_pt_fn = get_lambda_fns_shell(
            delta_p_to_t_shell)

    def get_Delta(self, q):
        Delta = self._evaluate_symbolic(q, (self.__pp_fn, self.__pe_fn,  self.__ee_fn, self.__pt_fn), ())
        # print(Delta)
        return Delta

    def get_grad_hess_Delta(self, q):
        grad_Delta = self._evaluate_symbolic(
            q, (self.__grad_pp_fn, self.__grad_pe_fn, self.__grad_ee_fn, self.__grad_pt_fn), (18,))
        # print(grad_Delta)
        hess_Delta = self._evaluate_symbolic(
            q, (self.__hess_pp_fn, self.__hess_pe_fn, self.__hess_ee_fn, self.__hess_pt_fn), (18, 18,))
        # print(hess_Delta)
        return grad_Delta, hess_Delta

    ## helper functions (may move to separate file later)
    # edge-edge distance (lumelsky)
    def edge_edge_dist_batched(self, p, a, q, b):
        """
        Compute closest points between batches of edges. (Lumelsky method)

        Parameters:
        p, a: start and direction vectors of first batch of edges (shape: [N, 3])
        q, b: start and direction vectors of second batch of edges (shape: [N, 3])

        Returns:
        cp: closest points on first edges (shape: [N, 3])
        cq: closest points on second edges (shape: [N, 3])
        """
        A = np.sum(a * a, axis=1)
        B = np.sum(a * b, axis=1)
        C = np.sum(b * b, axis=1)
        D = np.sum(a * (p - q), axis=1)
        E = np.sum(b * (p - q), axis=1)

        det = A * C - B * B
        s = np.zeros_like(det)
        t = np.zeros_like(det)

        # Non-parallel case
        mask = det > 1e-15
        s[mask] = (B[mask] * E[mask] - C[mask] * D[mask]) / det[mask]
        t[mask] = (A[mask] * E[mask] - B[mask] * D[mask]) / det[mask]

        # Clamp s and t
        s = np.clip(s, 0.0, 1.0)
        t = np.clip(t, 0.0, 1.0)

        # Parallel case
        mask_parallel = ~mask
        mask_A = A > 1e-15
        mask_C = C > 1e-15
        mask_A_parallel = mask_parallel & mask_A
        mask_C_parallel = mask_parallel & mask_C

        s[mask_A_parallel] = -D[mask_A_parallel] / A[mask_A_parallel]
        s[mask_A_parallel] = np.clip(s[mask_A_parallel], 0.0, 1.0)
        t[mask_C_parallel] = (s[mask_C_parallel] * B[mask_C_parallel] + E[mask_C_parallel]) / C[mask_C_parallel]
        t[mask_C_parallel] = np.clip(t[mask_C_parallel], 0.0, 1.0)
        s[mask_A_parallel] = (t[mask_A_parallel] * B[mask_A_parallel] - D[mask_A_parallel]) / A[mask_A_parallel]
        s[mask_A_parallel] = np.clip(s[mask_A_parallel], 0.0, 1.0)

        cp = p + s[:, None] * a
        cq = q + t[:, None] * b

        return cp, cq
    
    # barycentric ratios
    def compute_barycentric_batch(self, p, point):
        """
        Compute barycentric coordinates for a batch of triangle-point pairs.

        Parameters:
        p: (batch_size, 3, 3) array representing triangle vertices
        point: (batch_size, 3) array representing query points

        Returns:
        bary_coords: (batch_size, 3) array of barycentric coordinates (u, v, w)
        """
        v0 = p[:, 1] - p[:, 0]
        v1 = p[:, 2] - p[:, 0]
        v2 = point - p[:, 0]

        d00 = np.einsum('ij,ij->i', v0, v0)
        d01 = np.einsum('ij,ij->i', v0, v1)
        d11 = np.einsum('ij,ij->i', v1, v1)
        d20 = np.einsum('ij,ij->i', v2, v0)
        d21 = np.einsum('ij,ij->i', v2, v1)

        denom = d00 * d11 - d01 * d01

        # Initialize output
        bary_coords = np.zeros((p.shape[0], 3))

        nondegenerate = np.abs(denom) >= 1e-10

        v = np.zeros_like(denom)
        w = np.zeros_like(denom)

        v[nondegenerate] = (d11[nondegenerate] * d20[nondegenerate] - d01[nondegenerate] * d21[nondegenerate]) / denom[nondegenerate]
        w[nondegenerate] = (d00[nondegenerate] * d21[nondegenerate] - d01[nondegenerate] * d20[nondegenerate]) / denom[nondegenerate]
        u = 1.0 - v - w

        bary_coords[:, 0] = u
        bary_coords[:, 1] = v
        bary_coords[:, 2] = w

        # Degenerate cases default to [1, 0, 0]
        bary_coords[~nondegenerate] = np.array([1.0, 0.0, 0.0])

        return bary_coords
    
    # point to triangle distance
    def point_triangle_dist_batched(self, p, q, Sv, Tv, min_dist_squared, min_p, min_q, shown_disjoint):
        batch_size = p.shape[0]

        # Triangle normals
        Sn = np.cross(Sv[:, 0], Sv[:, 1])
        Snl = np.sum(Sn * Sn, axis=1)
        Tn = np.cross(Tv[:, 0], Tv[:, 1])
        Tnl = np.sum(Tn * Tn, axis=1)

        # Project q onto plane of p
        valid_p = Snl > 1e-15
        p0 = p[:, 0]
        Tp = np.einsum('ij,ikj->ik', Sn, p0[:, None, :] - q)
        mask_p = valid_p & (np.all(Tp > 0, axis=1) | np.all(Tp < 0, axis=1))
        index = np.argmin(np.abs(Tp), axis=1)
        q_index = q[np.arange(batch_size), index]
        z0 = np.cross(Sn, Sv[:, 0])
        z1 = np.cross(Sn, Sv[:, 1])
        z2 = np.cross(Sn, Sv[:, 2])
        v0 = q_index - p[:, 0]
        v1 = q_index - p[:, 1]
        v2 = q_index - p[:, 2]
        dot0 = np.einsum('ij,ij->i', v0, z0)
        dot1 = np.einsum('ij,ij->i', v1, z1)
        dot2 = np.einsum('ij,ij->i', v2, z2)
        inside = (dot0 > 0) & (dot1 > 0) & (dot2 > 0) & mask_p
        scale = Tp[np.arange(batch_size), index] / Snl
        cp = q_index + Sn * scale[:, None]
        cq = q_index
        dists = np.sum((cp - cq) ** 2, axis=1)
        update = inside & (dists < min_dist_squared)
        min_dist_squared[update] = dists[update]
        min_p[update] = cp[update]
        min_q[update] = cq[update]
        shown_disjoint[update] = True

        # Project p onto plane of q
        valid_q = Tnl > 1e-15
        q0 = q[:, 0]
        Sp = np.einsum('ij,ikj->ik', Tn, q0[:, None, :] - p)
        mask_q = valid_q & (np.all(Sp > 0, axis=1) | np.all(Sp < 0, axis=1))
        index = np.argmin(np.abs(Sp), axis=1)
        p_index = p[np.arange(batch_size), index]
        w0 = np.cross(Tn, Tv[:, 0])
        w1 = np.cross(Tn, Tv[:, 1])
        w2 = np.cross(Tn, Tv[:, 2])
        u0 = p_index - q[:, 0]
        u1 = p_index - q[:, 1]
        u2 = p_index - q[:, 2]
        dot0 = np.einsum('ij,ij->i', u0, w0)
        dot1 = np.einsum('ij,ij->i', u1, w1)
        dot2 = np.einsum('ij,ij->i', u2, w2)
        inside_q = (dot0 > 0) & (dot1 > 0) & (dot2 > 0) & mask_q
        scale = Sp[np.arange(batch_size), index] / Tnl
        cq = p_index + Tn * scale[:, None]
        cp = p_index
        dists = np.sum((cp - cq) ** 2, axis=1)
        update = inside_q & (dists < min_dist_squared)
        min_dist_squared[update] = dists[update]
        min_p[update] = cp[update]
        min_q[update] = cq[update]
        shown_disjoint[update] = True

        return min_dist_squared, min_p, min_q, shown_disjoint

    
    # triangle triangle distance (full vectorized implementation)
    def distance_triangle_triangle_squared_batch(self, p, q):
        """
        Optimized vectorized computation of minimum squared distance between triangle pairs.
        """
        batch_size = p.shape[0]

        Sv = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 1], p[:, 0] - p[:, 2]], axis=1)
        Tv = np.stack([q[:, 1] - q[:, 0], q[:, 2] - q[:, 1], q[:, 0] - q[:, 2]], axis=1)

        min_dist_squared = np.full(batch_size, np.inf)
        min_p = np.zeros((batch_size, 3))
        min_q = np.zeros((batch_size, 3))
        shown_disjoint = np.zeros(batch_size, dtype=bool)

        for i in range(3):
            for j in range(3):
                edge_p = p[:, i]
                edge_a = Sv[:, i]
                edge_q = q[:, j]
                edge_b = Tv[:, j]

                cp_temp, cq_temp = self.edge_edge_dist_batched(edge_p, edge_a, edge_q, edge_b)
                v = cq_temp - cp_temp
                dist_squared = np.sum(v * v, axis=1)

                update_mask = dist_squared <= min_dist_squared
                update_indices = np.where(update_mask)[0]

                if update_indices.size > 0:
                    min_dist_squared[update_mask] = dist_squared[update_mask]
                    min_p[update_mask] = cp_temp[update_mask]
                    min_q[update_mask] = cq_temp[update_mask]

                    id_p = (i + 2) % 3
                    z_p = p[update_mask, id_p] - cp_temp[update_mask]
                    a = np.sum(z_p * v[update_mask], axis=1)

                    id_q = (j + 2) % 3
                    z_q = q[update_mask, id_q] - cq_temp[update_mask]
                    b = np.sum(z_q * v[update_mask], axis=1)

                    early_exit_sub = (a <= 0.0) & (b >= 0.0)
                    early_exit = np.zeros(batch_size, dtype=bool)
                    early_exit[update_indices[early_exit_sub]] = True

                    early_exit_idxs = update_indices[early_exit_sub]
                    min_dist_squared[early_exit_idxs] = dist_squared[early_exit_idxs]
                    min_p[early_exit_idxs] = cp_temp[early_exit_idxs]
                    min_q[early_exit_idxs] = cq_temp[early_exit_idxs]
                    shown_disjoint[early_exit_idxs] = True

                    remaining_mask = update_mask & ~early_exit
                    if np.any(remaining_mask):
                        a_arr = a[~early_exit_sub]
                        b_arr = b[~early_exit_sub]
                        a_arr[a_arr <= 0.0] = 0.0
                        b_arr[b_arr > 0.0] = 0.0
                        disjoint_update = (min_dist_squared[remaining_mask] - a_arr + b_arr) > 0.0
                        shown_disjoint[remaining_mask] |= disjoint_update

        # Process point-to-triangle projection checks
        min_dist_squared, min_p, min_q, shown_disjoint = self.point_triangle_dist_batched(
        p, q, Sv, Tv, min_dist_squared, min_p, min_q, shown_disjoint
        )

        # Vectorized barycentric computation
        ratios_p = self.compute_barycentric_batch(p, min_p)
        ratios_q = self.compute_barycentric_batch(q, min_q)

        # Snap near-zero and near-one values
        eps = 1e-10
        ratios_p[ratios_p < eps] = 0.0
        ratios_p[ratios_p > 1.0 - eps] = 1.0

        ratios_q[ratios_q < eps] = 0.0
        ratios_q[ratios_q > 1.0 - eps] = 1.0

        # Handle intersecting cases
        intersecting = ~shown_disjoint
        min_dist_squared[intersecting] = 0.0
        min_p[intersecting] = p[intersecting, 0]
        min_q[intersecting] = q[intersecting, 0]
        ratios_p[intersecting] = np.array([1.0, 0.0, 0.0])
        ratios_q[intersecting] = np.array([1.0, 0.0, 0.0])

        return min_dist_squared, min_p, min_q, ratios_p, ratios_q


    # reorder vertices
    def reorder_triangle_pair_nodes_with_true_inverse_mask(self, ratios_p, ratios_q, node_indices):
        """
        Enhanced version with correct inverse permutation mask (position-based, not value-based).
        """
        batch_size = ratios_p.shape[0]
        reorder_mask = np.copy(node_indices)
        inverse_permutation_mask = np.tile(np.arange(6), (batch_size, 1))  # initialize as identity
        contact_types = np.array(["Unknown"] * batch_size, dtype=object)

        def classify_contact_type(rp, rq):
            is_vertex = (np.sum(rp == 1.0, axis=1) == 1) & (np.sum(rp == 0.0, axis=1) == 2)
            is_edge = (np.sum(rp > 0.0, axis=1) == 2) & (np.sum(rp == 0.0, axis=1) == 1)
            is_face = np.all(rp > 0.0, axis=1)

            types = np.array(["Unknown"] * batch_size, dtype=object)
            mask_pp = is_vertex & (np.sum(rq == 1.0, axis=1) == 1) & (np.sum(rq == 0.0, axis=1) == 2)
            mask_pe = is_vertex & ((np.sum(rq > 0.0, axis=1) == 2) & (np.sum(rq == 0.0, axis=1) == 1))
            mask_ep = ((np.sum(rp > 0.0, axis=1) == 2) & (np.sum(rp == 0.0, axis=1) == 1)) & (np.sum(rq == 1.0, axis=1) == 1) & (np.sum(rq == 0.0, axis=1) == 2)
            mask_ee = ((np.sum(rp > 0.0, axis=1) == 2) & (np.sum(rp == 0.0, axis=1) == 1)) & ((np.sum(rq > 0.0, axis=1) == 2) & (np.sum(rq == 0.0, axis=1) == 1))
            mask_pf = is_vertex & np.all(rq > 0.0, axis=1)
            mask_fp = np.all(rp > 0.0, axis=1) & (np.sum(rq == 1.0, axis=1) == 1) & (np.sum(rq == 0.0, axis=1) == 2)
            mask_ef = ((np.sum(rp > 0.0, axis=1) == 2) & (np.sum(rp == 0.0, axis=1) == 1)) & np.all(rq > 0.0, axis=1)
            mask_fe = np.all(rp > 0.0, axis=1) & ((np.sum(rq > 0.0, axis=1) == 2) & (np.sum(rq == 0.0, axis=1) == 1))

            types[mask_pp] = "PointToPoint"
            types[mask_pe | mask_ep] = "PointToEdge"
            types[mask_ee] = "EdgeToEdge"
            types[mask_pf | mask_fp] = "PointToFace"
            types[mask_ef | mask_fe] = "EdgeToFace"

            return types

        contact_types = classify_contact_type(ratios_p, ratios_q)

        for i in range(batch_size):
            p_nodes = node_indices[i, :3].tolist()
            q_nodes = node_indices[i, 3:].tolist()
            original = p_nodes + q_nodes
            rp, rq = ratios_p[i], ratios_q[i]
            ctype = contact_types[i]

            def rotate_to_start(lst, start_idx):
                return lst[start_idx:] + lst[:start_idx]

            def edge_reorder(lst, ei):
                a, b = ei
                lst_rot = rotate_to_start(lst, a)
                if lst_rot[1] != lst[b]:
                    lst_rot = [lst_rot[0], lst[b]] + [x for x in lst_rot if x not in [lst_rot[0], lst[b]]]
                return lst_rot

            if ctype == "PointToPoint":
                pi = np.argmax(rp)
                qi = np.argmax(rq)
                new_p = rotate_to_start(p_nodes, pi)
                new_q = rotate_to_start(q_nodes, qi)
            elif ctype == "PointToEdge":
                if (rp == 1.0).sum() == 1:
                    vi = np.argmax(rp)
                    ei = np.where(rq > 0.0)[0]
                    new_p = rotate_to_start(p_nodes, vi)
                    new_q = edge_reorder(q_nodes, ei)
                else:
                    vi = np.argmax(rq)
                    ei = np.where(rp > 0.0)[0]
                    new_q = rotate_to_start(q_nodes, vi)
                    new_p = edge_reorder(p_nodes, ei)
                    new_p, new_q = new_q, new_p  # swap
            elif ctype == "EdgeToEdge":
                ei_p = np.where(rp > 0.0)[0]
                ei_q = np.where(rq > 0.0)[0]
                new_p = edge_reorder(p_nodes, ei_p)
                new_q = edge_reorder(q_nodes, ei_q)
            elif ctype == "PointToFace":
                if (rp == 1.0).sum() == 1:
                    pi = np.argmax(rp)
                    new_p = rotate_to_start(p_nodes, pi)
                    new_q = q_nodes
                else:
                    qi = np.argmax(rq)
                    new_q = rotate_to_start(q_nodes, qi)
                    new_p = p_nodes
                    new_p, new_q = new_q, new_p  # swap
            elif ctype == "EdgeToFace":
                if ((rp > 0.0).sum() == 2):
                    ei = np.where(rp > 0.0)[0]
                    new_p = edge_reorder(p_nodes, ei)
                    new_q = q_nodes
                else:
                    ei = np.where(rq > 0.0)[0]
                    new_q = edge_reorder(q_nodes, ei)
                    new_p = p_nodes
                    new_p, new_q = new_q, new_p  # swap
            else:
                new_p, new_q = p_nodes, q_nodes

            new_order = new_p + new_q
            reorder_mask[i] = new_order

            # compute true inverse permutation: where did each of the original entries end up
            inverse = [new_order.index(val) for val in original]
            inverse_permutation_mask[i] = inverse

        return reorder_mask, contact_types.tolist(), inverse_permutation_mask

    def reorder_triangle_pair_dof_indices_with_inverse_vectorized(self,
        ratios_p, ratios_q, node_indices, ind, dof_per_node=3):
        """
        Fully corrected vectorized version of DOF reordering with accurate inverse permutation mask.
        """
        batch_size = node_indices.shape[0]

        # Step 1: Get node reordering and inverse mapping
        reorder_mask, contact_types, inverse_node_mask = self.reorder_triangle_pair_nodes_with_true_inverse_mask(
            ratios_p, ratios_q, node_indices
        )

        # Step 2: Build node index -> position map for each batch
        reorder_node_positions = np.zeros_like(reorder_mask)
        for i in range(batch_size):
            # Find the position of each node in the original node_indices[i]
            reorder_node_positions[i] = [np.where(node_indices[i] == node)[0][0] for node in reorder_mask[i]]

        # Step 3: Apply node reordering to DOF blocks
        node_dofs = ind.reshape(batch_size, 6, dof_per_node)
        reordered_node_dofs = np.take_along_axis(
            node_dofs,
            reorder_node_positions[:, :, None].repeat(dof_per_node, axis=2),
            axis=1
        )
        reordered_ind = reordered_node_dofs.reshape(batch_size, -1)

        # Step 4: Compute inverse DOF mask using inverse node permutation
        inverse_dof_mask = np.zeros((batch_size, 6 * dof_per_node), dtype=int)
        for i in range(batch_size):
            inv_mask = []
            for node_pos in inverse_node_mask[i]:
                base = node_pos * dof_per_node
                inv_mask.extend([base + j for j in range(dof_per_node)])
            inverse_dof_mask[i] = inv_mask

        return reordered_ind, inverse_dof_mask, contact_types

    def _evaluate_symbolic(self, q, fns, shape):
        # q[self.ind] is (B,18)
        q_tri = q.reshape(-1, 2, 3, 3)    # (B, 2, 3, 3)   
        dist_squared, cp, cq, ratios_p, ratios_q = self.distance_triangle_triangle_squared_batch(q_tri[:, 0], q_tri[:, 1])
        # print("dist_square: ", dist_squared)
        
        reordered_ind, inv_mask, contact_types = self.reorder_triangle_pair_dof_indices_with_inverse_vectorized(
            ratios_p, ratios_q, self.pairs, self.ind
        )
        # print("reordered_ind: ", reordered_ind)
        #print("contact_types: ", contact_types)

        #if "Unknown" in contact_types:
        #    print("ratios: ", ratios_p, ratios_q)
          
        out = self._evalulate_piecewise(q, reordered_ind,contact_types, *fns, shape)

        # Restore original order
        # print("inv mask: ", inv_mask)
        if out.ndim == 2:
            out = np.take_along_axis(out, inv_mask, axis=1)
        elif out.ndim == 3:
            out = np.take_along_axis(out, inv_mask[:, :, None], axis=1)
            out = np.take_along_axis(out, inv_mask[:, None, :], axis=2)

        return out

    def _evalulate_piecewise(self, q, ind, contact_type, fn_p2p, fn_p2e, fn_e2e, fn_p2t, shape=(18,)):
        result = np.zeros((ind.shape[0],) + shape)
        contact_type = np.array(contact_type)

        # if not debugpy.is_client_connected():
        #     try:
        #         debugpy.listen(5678)
        #     except RuntimeError:
        #         pass  # Already listening

        #     print("Waiting for debugger attach...")
        #     debugpy.wait_for_client()

        # debugpy.breakpoint()

        # Masks based on contact_type string array
        mask_p2p = contact_type == "PointToPoint"
        mask_p2e = contact_type == "PointToEdge"
        mask_e2e = contact_type == "EdgeToEdge"
        mask_p2t = contact_type == "PointToFace"

        # print("mask_p2p", mask_p2p)
        # print("mask_p2e", mask_p2e)
        # print("mask_e2e", mask_e2e)
        # print("mask_p2t", mask_p2t)

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

            # Process e2e
        if np.any(mask_p2t):
            args = get_inputs(mask_p2t)
            result[mask_e2e] = fn_p2t(*args)

        return result
