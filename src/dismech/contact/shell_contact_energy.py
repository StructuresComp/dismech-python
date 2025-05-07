import numpy as np
from typing import List

from .contact_pairs import ContactPair
from .contact_energy import ContactEnergy
from .imc_helper import get_lambda_fns_shell, delta_p_to_p_shell, delta_p_to_e_shell, delta_e_to_e_shell, delta_p_to_t_shell 


class ShellContactEnergy(ContactEnergy):

    def __init__(self, pairs: List[ContactPair], delta: float, h: float, k_1: float = None, scale=True):
        if k_1 is None:
            k_1 = 15 / delta
        super().__init__(pairs, delta, h, k_1, scale)
        self.__pp_fn, self.__grad_pp_fn, self.__hess_pp_fn = get_lambda_fns_shell(
            delta_p_to_p_shell)
        self.__pe_fn, self.__grad_pe_fn, self.__hess_pe_fn = get_lambda_fns_shell(
            delta_p_to_e_shell)
        self.__ee_fn, self.__grad_ee_fn, self.__hess_ee_fn = get_lambda_fns_shell(
            delta_e_to_e_shell)
        self.__pt_fn, self.__grad_pt_fn, self.__hess_pt_fn = get_lambda_fns_shell(
            delta_p_to_t_shell)

    def get_Delta(self, q):
        return self._evaluate_symbolic(q, (self.__pp_fn, self.__pe_fn,  self.__ee_fn, self.__pt_fn), ())

    def get_grad_hess_Delta(self, q):
        grad_Delta = self._evaluate_symbolic(
            q, (self.__grad_pp_fn, self.__grad_pe_fn, self.__grad_ee_fn, self.__grad_pt_fn), (18,))
        hess_Delta = self._evaluate_symbolic(
            q, (self.__hess_pp_fn, self.__hess_pe_fn, self.__hess_ee_fn, self.__hess_pt_fn), (18, 18,))
        return grad_Delta, hess_Delta
    
    # TO DO: make the below function better optimized
    def evaluate_contact(self, q):
        x, y, z, a, b, c = q[self.ind].reshape(-1, 6, 3).transpose(1, 0, 2)

        contact_type = np.full(x.shape[0], "", dtype=str)
        min_dist = np.zeros(x.shape[0])
        ratios = np.zeros((x.shape[0],2, 3))
        masks = np.zeros((x.shape[0],18))

        for k in range(x.shape[0]):
            V1 = np.column_stack((x[k], y[k], z[k]))
            V2 = np.column_stack((a[k], b[k], c[k]))
            
            contact_info = {}
            dist = np.inf

            # Point-to-point
            for i in range(3):
                for j in range(3):
                    d = np.linalg.norm(V1[:, i] - V2[:, j])
                    if d < dist:
                        dist = d
                        contact_type[k] = "PointToPoint"
                        contact_info = {"p1": V1[:, i], "p2": V2[:, j], "idx1_pp": i, "idx2_pp": j}
                        ratio = np.zeros((2, 3))
                        ratio[0, i] = 1
                        ratio[1, j] = 1
                        idx1_pp, idx2_pp = i, j

            # Point-to-edge (both directions)
            for i in range(3):
                for j in range(3):
                    d, cp, r = self._point_to_segment(V1[:, i], V2[:, j], V2[:, (j + 1) % 3])
                    if d < dist:
                        dist = d
                        contact_type[k] = "PointToEdge"
                        contact_info = {"point": V1[:, i], "edge": (V2[:, j], V2[:, (j + 1) % 3])}
                        ratio = np.zeros((2, 3))
                        ratio[0, i] = 1
                        ratio[1, j] = r[0]
                        ratio[1, (j + 1) % 3] = r[1]

                    d, cp, r = self._point_to_segment(V2[:, i], V1[:, j], V1[:, (j + 1) % 3])
                    if d < dist:
                        dist = d
                        contact_type[k] = "PointToEdge"
                        contact_info = {"point": V2[:, i], "edge": (V1[:, j], V1[:, (j + 1) % 3])}
                        ratio = np.zeros((2, 3))
                        ratio[1, i] = 1
                        ratio[0, j] = r[0]
                        ratio[0, (j + 1) % 3] = r[1]

            # Edge-to-edge
            for i in range(3):
                a1, b1 = V1[:, i], V1[:, (i + 1) % 3]
                for j in range(3):
                    a2, b2 = V2[:, j], V2[:, (j + 1) % 3]
                    d, p1, p2, r1, r2 = self._segment_to_segment(a1, b1, a2, b2)
                    if d < dist:
                        dist = d
                        contact_type[k] = "EdgeToEdge"
                        contact_info = {"edge1": (a1, b1), "edge2": (a2, b2)}
                        ratio = np.zeros((2, 3))
                        ratio[0, i] = r1[0]
                        ratio[0, (i + 1) % 3] = r1[1]
                        ratio[1, j] = r2[0]
                        ratio[1, (j + 1) % 3] = r2[1]

            # Point-to-face
            for i in range(3):
                d, inside, bary = self._point_to_triangle(V1[:, i], V2)
                if inside and d < dist:
                    dist = d
                    contact_type[k] = "PointToFace"
                    contact_info = {"point": V1[:, i], "triangle": V2}
                    ratio = np.zeros((2, 3))
                    ratio[0, i] = 1
                    ratio[1, :] = bary

                d, inside, bary = self._point_to_triangle(V2[:, i], V1)
                if inside and d < dist:
                    dist = d
                    contact_type[k] = "PointToFace"
                    contact_info = {"point": V2[:, i], "triangle": V1}
                    ratio = np.zeros((2, 3))
                    ratio[1, i] = 1
                    ratio[0, :] = bary

            # Edge-to-face (intersection)
            for i in range(3):
                a, b = V1[:, i], V1[:, (i + 1) % 3]
                intersects, pt, bary2, bary1 = self._segment_triangle_intersection(a, b, V2)
                if intersects:
                    contact_type[k] = "EdgeToFace"
                    contact_info = {"edge": (a, b), "triangle": V2, "intersection": pt}
                    dist = 0
                    ratio = np.zeros((2, 3))
                    ratio[0, i] = bary1[0]
                    ratio[0, (i + 1) % 3] = bary1[1]
                    ratio[1, :] = bary2
                    break

                a, b = V2[:, i], V2[:, (i + 1) % 3]
                intersects, pt, bary2, bary1 = self._segment_triangle_intersection(a, b, V1)
                if intersects:
                    contact_type[k] = "EdgeToFace"
                    contact_info = {"edge": (a, b), "triangle": V1, "intersection": pt}
                    dist = 0
                    ratio = np.zeros((2, 3))
                    ratio[1, i] = bary1[0]
                    ratio[1, (i + 1) % 3] = bary1[1]
                    ratio[0, :] = bary2
                    break

            tri_nodes_updated, ratio_updated, node_mask = self._rearrange_triangles_for_contact(contact_type[k], contact_info, V1, V2, self.pairs[k,0], self.pairs[k,1], ratio)
            ratios[k,:,:] = ratio_updated
            masks[k,:] = [3*i + j for i in node_mask for j in range(3)]
            min_dist[k] = dist

        return min_dist, contact_type, ratios, masks
    
    # TO DO: make the below function better optimized
    def _rearrange_triangles_for_contact(contact_type, contact_info, V1, V2, face_nodes1, face_nodes2, ratios):
        need_to_switch = False
        original_node_order = np.concatenate((face_nodes1, face_nodes2))

        def find_vertex_index(V, point):
            for i in range(3):
                if np.allclose(V[:, i], point):
                    return i
            raise ValueError("Point not found in triangle")

        def circshift(arr, k):
            return np.roll(arr, -k, axis=-1)

        updated_tri1, updated_tri2 = V1.copy(), V2.copy()
        updated_face_nodes1, updated_face_nodes2 = face_nodes1.copy(), face_nodes2.copy()

        if contact_type == "PointToPoint":
            idx1_pp = contact_info.get("idx1_pp")
            idx2_pp = contact_info.get("idx2_pp")

            updated_tri1 = circshift(V1, idx1_pp - 1)
            updated_face_nodes1 = circshift(face_nodes1, idx1_pp - 1)
            ratios[0] = circshift(ratios[0], idx1_pp - 1)

            updated_tri2 = circshift(V2, idx2_pp - 1)
            updated_face_nodes2 = circshift(face_nodes2, idx2_pp - 1)
            ratios[1] = circshift(ratios[1], idx2_pp - 1)

        elif contact_type == "PointToEdge":
            point = contact_info["point"]
            edge = contact_info["edge"]

            if any(np.allclose(point, V1[:, i]) for i in range(3)):
                idx_point = find_vertex_index(V1, point)
                updated_tri1 = circshift(V1, idx_point - 1)
                updated_face_nodes1 = circshift(face_nodes1, idx_point - 1)
                ratios[0] = circshift(ratios[0], idx_point - 1)

                idx_edge_start = find_vertex_index(V2, edge[0])
                updated_tri2 = circshift(V2, idx_edge_start - 1)
                updated_face_nodes2 = circshift(face_nodes2, idx_edge_start - 1)
                ratios[1] = circshift(ratios[1], idx_edge_start - 1)

            else:
                idx_point = find_vertex_index(V2, point)
                updated_tri2 = circshift(V2, idx_point - 1)
                updated_face_nodes2 = circshift(face_nodes2, idx_point - 1)
                ratios[1] = circshift(ratios[1], idx_point - 1)

                idx_edge_start = find_vertex_index(V1, edge[0])
                updated_tri1 = circshift(V1, idx_edge_start - 1)
                updated_face_nodes1 = circshift(face_nodes1, idx_edge_start - 1)
                ratios[0] = circshift(ratios[0], idx_edge_start - 1)

                need_to_switch = True

        elif contact_type == "EdgeToEdge":
            edge1 = contact_info["edge1"]
            edge2 = contact_info["edge2"]

            idx1 = find_vertex_index(V1, edge1[0])
            updated_tri1 = circshift(V1, idx1 - 1)
            updated_face_nodes1 = circshift(face_nodes1, idx1 - 1)
            ratios[0] = circshift(ratios[0], idx1 - 1)

            idx2 = find_vertex_index(V2, edge2[0])
            updated_tri2 = circshift(V2, idx2 - 1)
            updated_face_nodes2 = circshift(face_nodes2, idx2 - 1)
            ratios[1] = circshift(ratios[1], idx2 - 1)

        elif contact_type == "PointToFace":
            point = contact_info["point"]
            if any(np.allclose(point, V1[:, i]) for i in range(3)):
                idx_point = find_vertex_index(V1, point)
                updated_tri1 = circshift(V1, idx_point - 1)
                updated_face_nodes1 = circshift(face_nodes1, idx_point - 1)
                ratios[0] = circshift(ratios[0], idx_point - 1)
            else:
                idx_point = find_vertex_index(V2, point)
                updated_tri2 = circshift(V2, idx_point - 1)
                updated_face_nodes2 = circshift(face_nodes2, idx_point - 1)
                ratios[1] = circshift(ratios[1], idx_point - 1)
                need_to_switch = True

        elif contact_type == "EdgeToFace":
            edge = contact_info["edge"]
            if any(np.allclose(edge[0], V1[:, i]) for i in range(3)):
                idx_edge_start = find_vertex_index(V1, edge[0])
                updated_tri1 = circshift(V1, idx_edge_start - 1)
                updated_face_nodes1 = circshift(face_nodes1, idx_edge_start - 1)
                ratios[0] = circshift(ratios[0], idx_edge_start - 1)
            else:
                idx_edge_start = find_vertex_index(V2, edge[0])
                updated_tri2 = circshift(V2, idx_edge_start - 1)
                updated_face_nodes2 = circshift(face_nodes2, idx_edge_start - 1)
                ratios[1] = circshift(ratios[1], idx_edge_start - 1)
                need_to_switch = True

        # Flatten triangles
        tri1_flat = updated_tri1.T.reshape(-1)
        tri2_flat = updated_tri2.T.reshape(-1)

        if need_to_switch:
            tri_pair_updated = np.concatenate((tri2_flat, tri1_flat))
            tri_pair_nodes_updated = np.concatenate((updated_face_nodes2, updated_face_nodes1))
            ratios = np.flipud(ratios)
        else:
            tri_pair_updated = np.concatenate((tri1_flat, tri2_flat))
            tri_pair_nodes_updated = np.concatenate((updated_face_nodes1, updated_face_nodes2))

        # Compute the mask (original position → new position)
        mask = [np.where(tri_pair_nodes_updated == orig_node)[0][0] for orig_node in original_node_order]

        return tri_pair_nodes_updated, ratios, mask


    def _evaluate_symbolic(self, q, fns, shape):
        min_dist, contact_type, ratios, mask = self.evaluate_contact(q)
        ind = np.take_along_axis(self.ind, mask, axis=1)
        out = self._evalulate_piecewise(q, ind,contact_type, *fns, shape)

        # Restore original order
        inv_mask = np.argsort(mask, axis=1)
        if out.ndim == 2:
            out = np.take_along_axis(out, inv_mask, axis=1)
        elif out.ndim == 3:
            out = np.take_along_axis(out, inv_mask[:, :, None], axis=1)
            out = np.take_along_axis(out, inv_mask[:, None, :], axis=2)
        return out

    def _evalulate_piecewise(self, q, ind, contact_type, fn_p2p, fn_p2e, fn_e2e, fn_p2t, shape=(18,)):
        result = np.zeros((ind.shape[0],) + shape)

        # Masks based on contact_type string array
        mask_p2p = contact_type == "PointToPoint"
        mask_p2e = contact_type == "PointToEdge"
        mask_e2e = contact_type == "EdgeToEdge"
        mask_p2t = contact_type == "PointToTriangle"

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
