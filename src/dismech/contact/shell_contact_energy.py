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
    

    # helper for shell 
    def _segment_triangle_intersection(self, p0, p1, tri, eps=1e-8):
        """
        Determines if a segment (p0 to p1) intersects a triangle (given by 3 columns of tri).

        Parameters:
        - p0, p1: 3D numpy arrays (shape: (3,))
        - tri: 3x3 numpy array (columns are triangle vertices)
        - eps: numerical tolerance for degenerate configurations

        Returns:
        - intersects: bool
        - pt: intersection point (if any), else None
        - tri_bary: barycentric coordinates in triangle
        - seg_bary: barycentric coordinates on segment
        """
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        direction = p1 - p0
        edge1 = b - a
        edge2 = c - a

        h = np.cross(direction, edge2)
        det = np.dot(edge1, h)

        if abs(det) < eps:
            return False, None, None, None  # Segment and triangle are parallel

        inv_det = 1.0 / det
        s = p0 - a
        u = np.dot(s, h) * inv_det

        q = np.cross(s, edge1)
        v = np.dot(direction, q) * inv_det
        t = np.dot(edge2, q) * inv_det

        intersects = (u >= 0) and (v >= 0) and (u + v <= 1) and (0 <= t <= 1)
        if not intersects:
            return False, None, None, None

        pt = p0 + t * direction
        tri_bary = np.array([1 - u - v, u, v])
        seg_bary = np.array([1 - t, t])

        return True, pt, tri_bary, seg_bary
    
    def _point_to_segment(self, p, a, b):
        ab = b - a
        ab_dot = np.dot(ab, ab)
        
        if ab_dot == 0:
            # a and b are the same point
            closest_point = a
            dist = np.linalg.norm(p - a)
            ratios = [0.5, 0.5]  # arbitrary
            return dist, closest_point, ratios

        t = np.dot(p - a, ab) / ab_dot
        t = max(0.0, min(1.0, t))
        closest_point = a + t * ab
        dist = np.linalg.norm(p - closest_point)
        ratios = [np.linalg.norm(b - closest_point) / np.linalg.norm(ab),
                np.linalg.norm(a - closest_point) / np.linalg.norm(ab)]
        return dist, closest_point, ratios
    
    import numpy as np

    def _segment_to_segment(self, a1, b1, a2, b2):
        u = b1 - a1
        v = b2 - a2
        w0 = a1 - a2

        a = np.dot(u, u)
        b = np.dot(u, v)
        c = np.dot(v, v)
        d = np.dot(u, w0)
        e = np.dot(v, w0)

        D = a * c - b * b
        s = 0.0
        t = 0.0

        if D > 1e-8:
            s = (b * e - c * d) / D
            t = (a * e - b * d) / D
            s = max(0.0, min(1.0, s))
            t = max(0.0, min(1.0, t))

        p1 = a1 + s * u
        p2 = a2 + t * v
        dist = np.linalg.norm(p1 - p2)

        ratio1 = [1 - s, s]  # [a1, b1]
        ratio2 = [1 - t, t]  # [a2, b2]

        return dist, p1, p2, ratio1, ratio2
    
    import numpy as np

    def _point_to_triangle(self, p, tri):
        a = tri[:, 0]
        b = tri[:, 1]
        c = tri[:, 2]
        
        # Triangle normal
        n = np.cross(b - a, c - a)
        n = n / np.linalg.norm(n)
        
        # Project point onto triangle plane
        dist_signed = np.dot(p - a, n)
        proj = p - dist_signed * n
        dist = abs(dist_signed)
        
        # Compute barycentric coordinates
        v0 = b - a
        v1 = c - a
        v2 = proj - a

        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)

        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1 - v - w

        bary_coords = np.array([u, v, w])
        is_inside = np.all(bary_coords >= 0) and np.all(bary_coords <= 1)

        return dist, is_inside, bary_coords



    # TO DO: make the below function better optimized
    def evaluate_contact(self, q):
        x, y, z, a, b, c = q[self.ind].reshape(-1, 6, 3).transpose(1, 0, 2)

        contact_type = np.full(x.shape[0], "", dtype=object)
        min_dist = np.zeros(x.shape[0])
        # ratios = np.zeros((x.shape[0],2, 3))
        masks = np.zeros((x.shape[0], 18), dtype=np.int64)

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
                        # ratio = np.zeros((2, 3))
                        # ratio[0, i] = 1
                        # ratio[1, j] = 1

            # Point-to-edge (both directions)
            for i in range(3):
                for j in range(3):
                    d, cp, r = self._point_to_segment(V1[:, i], V2[:, j], V2[:, (j + 1) % 3])
                    # d = delta_p_to_e_shell(V1[0, i], V1[1, i], V1[2, i], V1[0, (i + 1) % 3], V1[1, (i + 1) % 3], V1[2, (i + 1) % 3], V1[0, (i + 2) % 3], V1[1, (i + 2) % 3], V1[2, (i + 2) % 3],
                    #                               V2[0, j], V2[1, j], V2[2, j], V2[0, (j + 1) % 3], V2[1, (j + 1) % 3], V2[2, (j + 1) % 3], V2[0, (j + 2) % 3], V2[1, (j + 2) % 3], V2[2, (j + 2) % 3])
                    # v1 = V1[:, [i, (i + 1) % 3, (i + 2) % 3]].flatten(order='F')
                    # v2 = V2[:, [j, (j + 1) % 3, (j + 2) % 3]].flatten(order='F')
                    # d = delta_p_to_e_shell(*v1, *v2)

                    if d < dist:
                        dist = d
                        contact_type[k] = "PointToEdge"
                        contact_info = {"point": V1[:, i], "edge": (V2[:, j], V2[:, (j + 1) % 3])}
                        # ratio = np.zeros((2, 3))
                        # ratio[0, i] = 1
                        # ratio[1, j] = r[0]
                        # ratio[1, (j + 1) % 3] = r[1]

                    d, cp, r = self._point_to_segment(V2[:, i], V1[:, j], V1[:, (j + 1) % 3])
                    # v1 = V1[:, [i, (i + 1) % 3, (i + 2) % 3]].flatten(order='F')
                    # v2 = V2[:, [j, (j + 1) % 3, (j + 2) % 3]].flatten(order='F')

                    # d = delta_p_to_e_shell(*v2, *v1)

                    if d < dist:
                        dist = d
                        contact_type[k] = "PointToEdge"
                        contact_info = {"point": V2[:, i], "edge": (V1[:, j], V1[:, (j + 1) % 3])}
                        # ratio = np.zeros((2, 3))
                        # ratio[1, i] = 1
                        # ratio[0, j] = r[0]
                        # ratio[0, (j + 1) % 3] = r[1]

            # Edge-to-edge
            for i in range(3):
                a1, b1 = V1[:, i], V1[:, (i + 1) % 3]
                # a1b1c1 = V1[:, [i, (i + 1) % 3, (i + 2) % 3]].flatten(order='F')
                for j in range(3):
                    a2, b2 = V2[:, j], V2[:, (j + 1) % 3]
                    # a2b2c2 = V2[:, [j, (j + 1) % 3, (j + 2) % 3]].flatten(order='F')
                    
                    d, p1, p2, r1, r2 = self._segment_to_segment(a1, b1, a2, b2)
                    # d = delta_e_to_e_shell(*a1b1c1, *a2b2c2)

                    if d < dist:
                        dist = d
                        contact_type[k] = "EdgeToEdge"
                        contact_info = {"edge1": (a1, b1), "edge2": (a2, b2)}
                        # ratio = np.zeros((2, 3))
                        # ratio[0, i] = r1[0]
                        # ratio[0, (i + 1) % 3] = r1[1]
                        # ratio[1, j] = r2[0]
                        # ratio[1, (j + 1) % 3] = r2[1]

            # Point-to-face
            for i in range(3):
                a1b1c1 = V1[:, [i, (i + 1) % 3, (i + 2) % 3]].flatten(order='F')
                d, inside, bary = self._point_to_triangle(V1[:, i], V2)
                # d = delta_p_to_t_shell(*a1b1c1, *(V2.flatten(order='F')))
                if inside and d < dist:
                # if d < dist:
                    dist = d
                    contact_type[k] = "PointToFace"
                    contact_info = {"point": V1[:, i], "triangle": V2}
                    # ratio = np.zeros((2, 3))
                    # ratio[0, i] = 1
                    # ratio[1, :] = bary

                a2b2c2 = V2[:, [i, (i + 1) % 3, (i + 2) % 3]].flatten(order='F')
                d, inside, bary = self._point_to_triangle(V2[:, i], V1)
                # d = delta_p_to_t_shell( *a2b2c2, *(V1.flatten(order='F')))
                if inside and d < dist:
                # if d < dist:
                    dist = d
                    contact_type[k] = "PointToFace"
                    contact_info = {"point": V2[:, i], "triangle": V1}
                    # ratio = np.zeros((2, 3))
                    # ratio[1, i] = 1
                    # ratio[0, :] = bary

            # Edge-to-face (intersection)
            for i in range(3):
                a, b = V1[:, i], V1[:, (i + 1) % 3]
                intersects, pt, bary2, bary1 = self._segment_triangle_intersection(a, b, V2)
                if intersects:
                    contact_type[k] = "EdgeToFace"
                    contact_info = {"edge": (a, b), "triangle": V2, "intersection": pt}
                    dist = 0
                    # ratio = np.zeros((2, 3))
                    # ratio[0, i] = bary1[0]
                    # ratio[0, (i + 1) % 3] = bary1[1]
                    # ratio[1, :] = bary2
                    break
                a, b = V2[:, i], V2[:, (i + 1) % 3]
                intersects, pt, bary2, bary1 = self._segment_triangle_intersection(a, b, V1)
                if intersects:
                    contact_type[k] = "EdgeToFace"
                    contact_info = {"edge": (a, b), "triangle": V1, "intersection": pt}
                    dist = 0
                    # ratio = np.zeros((2, 3))
                    # ratio[1, i] = bary1[0]
                    # ratio[1, (i + 1) % 3] = bary1[1]
                    # ratio[0, :] = bary2
                    break
            
            ratio = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] # for now

            print(contact_info)

            tri_nodes_updated, ratio_updated, node_mask = self._rearrange_triangles_for_contact(contact_type[k], contact_info, V1, V2, self.pairs[k,0:3], self.pairs[k,3:6], ratio)
            # ratios[k,:,:] = ratio_updated
            masks[k,:] = [3*i + j for i in node_mask for j in range(3)]
            min_dist[k] = dist
        
        print("min distance is: ", min_dist)

        return min_dist, contact_type, masks # , ratios
    
    # TO DO: make the below function better optimized
    def _rearrange_triangles_for_contact(self, contact_type, contact_info, V1, V2, face_nodes1, face_nodes2, ratios):
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

            updated_tri1 = circshift(V1, -idx1_pp)
            updated_face_nodes1 = circshift(face_nodes1, -idx1_pp)
            # print("updated_face_nodes1:", updated_face_nodes1)
            ratios[0] = circshift(ratios[0], -idx1_pp)

            updated_tri2 = circshift(V2, -idx2_pp)
            updated_face_nodes2 = circshift(face_nodes2, -idx2_pp)
            # print("updated_face_nodes2:", updated_face_nodes2)
            ratios[1] = circshift(ratios[1], -idx2_pp)

        elif contact_type == "PointToEdge":
            point = contact_info["point"]
            edge = contact_info["edge"]

            if any(np.allclose(point, V1[:, i]) for i in range(3)):
                idx_point = find_vertex_index(V1, point)
                updated_tri1 = circshift(V1, -idx_point)
                updated_face_nodes1 = circshift(face_nodes1, -idx_point)
                ratios[0] = circshift(ratios[0], -idx_point)

                idx_edge_start = find_vertex_index(V2, edge[0])
                updated_tri2 = circshift(V2, -idx_edge_start)
                updated_face_nodes2 = circshift(face_nodes2, -idx_edge_start)
                ratios[1] = circshift(ratios[1], -idx_edge_start)

            else:
                idx_point = find_vertex_index(V2, point)
                updated_tri2 = circshift(V2, -idx_point)
                updated_face_nodes2 = circshift(face_nodes2, -idx_point)
                ratios[1] = circshift(ratios[1], -idx_point)

                idx_edge_start = find_vertex_index(V1, edge[0])
                updated_tri1 = circshift(V1, -idx_edge_start)
                updated_face_nodes1 = circshift(face_nodes1, -idx_edge_start)
                ratios[0] = circshift(ratios[0], -idx_edge_start)

                need_to_switch = True

        elif contact_type == "EdgeToEdge":
            edge1 = contact_info["edge1"]
            edge2 = contact_info["edge2"]

            idx1 = find_vertex_index(V1, edge1[0])
            updated_tri1 = circshift(V1, -idx1)
            updated_face_nodes1 = circshift(face_nodes1, -idx1)
            ratios[0] = circshift(ratios[0], -idx1)

            idx2 = find_vertex_index(V2, edge2[0])
            updated_tri2 = circshift(V2, -idx2)
            updated_face_nodes2 = circshift(face_nodes2, -idx2)
            ratios[1] = circshift(ratios[1], -idx2)

        elif contact_type == "PointToFace":
            point = contact_info["point"]
            if any(np.allclose(point, V1[:, i]) for i in range(3)):
                idx_point = find_vertex_index(V1, point)
                updated_tri1 = circshift(V1, -idx_point)
                updated_face_nodes1 = circshift(face_nodes1, -idx_point)
                ratios[0] = circshift(ratios[0], -idx_point)
            else:
                idx_point = find_vertex_index(V2, point)
                updated_tri2 = circshift(V2, idx_point)
                updated_face_nodes2 = circshift(face_nodes2, idx_point)
                ratios[1] = circshift(ratios[1], idx_point)
                need_to_switch = True

        elif contact_type == "EdgeToFace":
            edge = contact_info["edge"]
            if any(np.allclose(edge[0], V1[:, i]) for i in range(3)):
                idx_edge_start = find_vertex_index(V1, edge[0])
                updated_tri1 = circshift(V1, -idx_edge_start)
                updated_face_nodes1 = circshift(face_nodes1, -idx_edge_start)
                ratios[0] = circshift(ratios[0], -idx_edge_start)
            else:
                idx_edge_start = find_vertex_index(V2, edge[0])
                updated_tri2 = circshift(V2, idx_edge_start)
                updated_face_nodes2 = circshift(face_nodes2, idx_edge_start)
                ratios[1] = circshift(ratios[1], idx_edge_start)
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

    # def _rearrange_triangles_for_contact(self, contact_type, contact_info, V1, V2, face_nodes1, face_nodes2, ratios):
    #     print(V1)
    #     print(V2)
    #     print("face_nodes1: ", face_nodes1)
    #     print("face_nodes2: ", face_nodes2)
    #     ratios = [np.asarray(r) for r in ratios]
    #     need_to_switch = False
    #     original_node_order = np.concatenate((face_nodes1, face_nodes2))

    #     def find_vertex_index(V, point):
    #         for i in range(3):
    #             if np.allclose(V[:, i], point):
    #                 return i
    #         raise ValueError("Point not found in triangle")

    #     def circshift(arr, k):
    #         return np.roll(arr, -k, axis=-1)

    #     def reorder_edge_first(V, face_nodes, edge):
    #         idx1 = find_vertex_index(V, edge[0])
    #         idx2 = find_vertex_index(V, edge[1])
    #         if (idx2 - idx1) % 3 == 1:  # orientation preserved
    #             perm = [idx1, idx2, 3 - idx1 - idx2]
    #         else:
    #             perm = [idx2, idx1, 3 - idx1 - idx2]
    #         return V[:, perm], face_nodes[perm], perm

    #     updated_tri1, updated_tri2 = V1.copy(), V2.copy()
    #     updated_face_nodes1, updated_face_nodes2 = face_nodes1.copy(), face_nodes2.copy()

    #     if contact_type == "PointToPoint":
    #         idx1_pp = contact_info["idx1_pp"]
    #         idx2_pp = contact_info["idx2_pp"]

    #         local_idx1 = np.where(face_nodes1 == idx1_pp)[0][0]
    #         updated_tri1 = circshift(V1, -local_idx1)
    #         updated_face_nodes1 = circshift(face_nodes1, -local_idx1)
    #         ratios[0] = circshift(ratios[0], -local_idx1)

    #         local_idx2 = np.where(face_nodes2 == idx2_pp)[0][0]
    #         updated_tri2 = circshift(V2, -local_idx2)
    #         updated_face_nodes2 = circshift(face_nodes2, -local_idx2)
    #         ratios[1] = circshift(ratios[1], -local_idx2)

    #     elif contact_type == "PointToEdge":
    #         point = contact_info["point"]
    #         edge = contact_info["edge"]

    #         if any(np.allclose(point, V1[:, i]) for i in range(3)):
    #             idx_point = find_vertex_index(V1, point)
    #             updated_tri1 = circshift(V1, -idx_point)
    #             updated_face_nodes1 = circshift(face_nodes1, -idx_point)
    #             ratios[0] = circshift(ratios[0], -idx_point)

    #             updated_tri2, updated_face_nodes2, perm = reorder_edge_first(V2, face_nodes2, edge)
    #             ratios[1] = ratios[1][perm]
    #         else:
    #             idx_point = find_vertex_index(V2, point)
    #             updated_tri2 = circshift(V2, -idx_point)
    #             updated_face_nodes2 = circshift(face_nodes2, -idx_point)
    #             ratios[1] = circshift(ratios[1], -idx_point)

    #             updated_tri1, updated_face_nodes1, perm = reorder_edge_first(V1, face_nodes1, edge)
    #             ratios[0] = ratios[0][perm]
    #             need_to_switch = True

    #     elif contact_type == "EdgeToEdge":
    #         edge1 = contact_info["edge1"]
    #         edge2 = contact_info["edge2"]

    #         updated_tri1, updated_face_nodes1, perm1 = reorder_edge_first(V1, face_nodes1, edge1)
    #         ratios[0] = ratios[0][perm1]

    #         updated_tri2, updated_face_nodes2, perm2 = reorder_edge_first(V2, face_nodes2, edge2)
    #         ratios[1] = ratios[1][perm2]

    #     elif contact_type == "PointToFace":
    #         point = contact_info["point"]
    #         if any(np.allclose(point, V1[:, i]) for i in range(3)):
    #             idx_point = find_vertex_index(V1, point)
    #             updated_tri1 = circshift(V1, -idx_point)
    #             updated_face_nodes1 = circshift(face_nodes1, -idx_point)
    #             ratios[0] = circshift(ratios[0], -idx_point)
    #         else:
    #             idx_point = find_vertex_index(V2, point)
    #             updated_tri2 = circshift(V2, -idx_point)
    #             updated_face_nodes2 = circshift(face_nodes2, -idx_point)
    #             ratios[1] = circshift(ratios[1], -idx_point)
    #             need_to_switch = True

    #     elif contact_type == "EdgeToFace":
    #         edge = contact_info["edge"]
    #         if any(np.allclose(edge[0], V1[:, i]) for i in range(3)):
    #             updated_tri1, updated_face_nodes1, perm = reorder_edge_first(V1, face_nodes1, edge)
    #             ratios[0] = ratios[0][perm]
    #         else:
    #             updated_tri2, updated_face_nodes2, perm = reorder_edge_first(V2, face_nodes2, edge)
    #             ratios[1] = ratios[1][perm]
    #             need_to_switch = True

    #     # Flatten triangles
    #     tri1_flat = updated_tri1.T.reshape(-1)
    #     tri2_flat = updated_tri2.T.reshape(-1)

    #     if need_to_switch:
    #         tri_pair_updated = np.concatenate((tri2_flat, tri1_flat))
    #         tri_pair_nodes_updated = np.concatenate((updated_face_nodes2, updated_face_nodes1))
    #         ratios = np.flipud(ratios)
    #     else:
    #         tri_pair_updated = np.concatenate((tri1_flat, tri2_flat))
    #         tri_pair_nodes_updated = np.concatenate((updated_face_nodes1, updated_face_nodes2))

    #     # Compute the mask (original position → new position)
    #     mask = [np.where(tri_pair_nodes_updated == orig_node)[0][0] for orig_node in original_node_order]

    #     print("updated nodes: ", tri_pair_nodes_updated)
    #     print("node mask: ", mask)

    #     return tri_pair_nodes_updated, ratios, mask



    def _evaluate_symbolic(self, q, fns, shape):
        min_dist, contact_type, mask = self.evaluate_contact(q)
        print("contact type is:", contact_type)
        print("mask:", mask)
        ind = np.take_along_axis(self.ind, mask, axis=1)
        out = self._evalulate_piecewise(q, ind,contact_type, *fns, shape)

        # Restore original order
        inv_mask = np.argsort(mask, axis=1)
        print("inv mask: ", inv_mask)
        if out.ndim == 2:
            out = np.take_along_axis(out, inv_mask, axis=1)
        elif out.ndim == 3:
            out = np.take_along_axis(out, inv_mask[:, :, None], axis=1)
            out = np.take_along_axis(out, inv_mask[:, None, :], axis=2)

        # print("out:")
        # print(out)
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
            print("args: ", args)
            # print(result)

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
