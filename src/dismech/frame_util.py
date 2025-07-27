import typing
import numpy as np


# FIXME: using triangle_energy as a template, vectorize this operations (not i,j,k)
def compute_tfc_midedge(p_s: np.ndarray, tau0_s: np.ndarray, s_s: np.ndarray) -> typing.Tuple[np.ndarray, ...]:
    # Sign adjust tau0
    tau = s_s[:, None] * tau0_s
    tau_i0 = s_s[:, 0][:, None] * tau0_s[:, 0, :]
    tau_j0 = s_s[:, 1][:, None] * tau0_s[:, 1, :]
    tau_k0 = s_s[:, 2][:, None] * tau0_s[:, 2, :]

    # Compute edge vectors
    vi = p_s[:, 2] - p_s[:, 1]
    vj = p_s[:, 0] - p_s[:, 2]
    vk = p_s[:, 1] - p_s[:, 0]

    # Compute edge lengths for each triangle in the batch
    li = np.linalg.norm(vi, axis=1)
    lj = np.linalg.norm(vj, axis=1)
    lk = np.linalg.norm(vk, axis=1)

    # Compute the face normal (using the cross product of vk and vi)
    normal = np.cross(vk, vi)
    norm_normal = np.linalg.norm(normal, axis=1, keepdims=True)
    A = norm_normal / 2.0  # area of the triangle face for each batch
    unit_norm = normal / norm_normal  # normalized face normal for each batch

    # Compute tangent vectors (perpendicular to the edges and in the plane of the triangle)
    t_i = np.cross(vi, unit_norm)
    t_j = np.cross(vj, unit_norm)
    t_k = np.cross(vk, unit_norm)

    # Normalize the tangent vectors before computing dot products
    t_i_norm = np.linalg.norm(t_i, axis=1, keepdims=True)
    t_j_norm = np.linalg.norm(t_j, axis=1, keepdims=True)
    t_k_norm = np.linalg.norm(t_k, axis=1, keepdims=True)

    t_i_normalized = t_i / t_i_norm
    t_j_normalized = t_j / t_j_norm
    t_k_normalized = t_k / t_k_norm

    # Compute the dot products needed for the c_i's (one dot per triangle in the batch)
    dot_i = np.sum(t_i_normalized * tau_i0, axis=1)
    dot_j = np.sum(t_j_normalized * tau_j0, axis=1)
    dot_k = np.sum(t_k_normalized * tau_k0, axis=1)

    # Compute scalar coefficients c_i, c_j, c_k
    c_i = 1.0 / (A.flatten() * li * dot_i)
    c_j = 1.0 / (A.flatten() * lj * dot_j)
    c_k = 1.0 / (A.flatten() * lk * dot_k)

    # Compute force components f_i, f_j, f_k as the dot products of the face normal with the scaled tau0 vectors
    f_i = np.sum(unit_norm * tau_i0, axis=1)
    f_j = np.sum(unit_norm * tau_j0, axis=1)
    f_k = np.sum(unit_norm * tau_k0, axis=1)

    fs = np.stack([f_i, f_j, f_k], axis=1)
    ts = np.stack([t_i, t_j, t_k], axis=1)
    cs = np.stack([c_i, c_j, c_k], axis=1)

    return ts.transpose(0, 2, 1), fs, cs


def compute_reference_twist(edges: np.ndarray,
                            sgn: np.ndarray,
                            a1: np.ndarray, tangent: np.ndarray,
                            ref_twist: np.ndarray) -> np.ndarray:
    e0 = edges[:, 0]
    e1 = edges[:, 1]
    t0 = tangent[e0] * sgn[:, 0][:, None]
    t1 = tangent[e1] * sgn[:, 1][:, None]
    u0 = a1[e0]
    u1 = a1[e1]

    ut = parallel_transport(u0, t0, t1)
    ut = rotate_axis_angle(ut, t1, ref_twist)

    angles = signed_angle(ut, u1, t1)
    return ref_twist + angles


def parallel_transport(u: np.ndarray, t_start: np.ndarray, t_end: np.ndarray) -> np.ndarray:
    # Determine if inputs are batched and ensure 2D arrays
    batched = t_start.ndim > 1
    if not batched:
        u = np.expand_dims(u, axis=0)
        t_start = np.expand_dims(t_start, axis=0)
        t_end = np.expand_dims(t_end, axis=0)

    # Compute cross product and its norm
    b = np.cross(t_start, t_end)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    mask = b_norm.squeeze(axis=1) < 1e-10

    # Safe normalization of cross product
    safe_b_norm = np.where(b_norm < 1e-10, 1.0, b_norm)
    b_normalized = b / safe_b_norm

    # Orthogonalize against t_start
    dot_prod = np.einsum('ij,ij->i', b_normalized, t_start)
    b_ortho = b_normalized - dot_prod[:, np.newaxis] * t_start

    # Safe renormalization
    b_ortho_norm = np.linalg.norm(b_ortho, axis=1, keepdims=True)
    safe_b_ortho_norm = np.where(b_ortho_norm < 1e-10, 1.0, b_ortho_norm)
    b_ortho_normalized = b_ortho / safe_b_ortho_norm

    # Compute transport bases
    n1 = np.cross(t_start, b_ortho_normalized)
    n2 = np.cross(t_end, b_ortho_normalized)

    # Project u onto components and transport
    components = (
        np.einsum('ij,ij->i', u, t_start)[:, np.newaxis] * t_end +
        np.einsum('ij,ij->i', u, n1)[:, np.newaxis] * n2 +
        np.einsum('ij,ij->i', u,
                  b_ortho_normalized)[:, np.newaxis] * b_ortho_normalized
    )

    # Preserve original vectors where transport is unnecessary
    result = np.where(mask[:, np.newaxis], u, components)

    # Remove batch dimension if input was non-batched
    if not batched:
        result = result.squeeze(axis=0)

    return result


def rotate_axis_angle(v: np.ndarray, axis: np.ndarray, theta: np.ndarray) -> np.ndarray:
    # Ensure inputs are at least 2D for batched processing
    batched = v.ndim > 1
    if not batched:
        v = np.expand_dims(v, axis=0)
        axis = np.expand_dims(axis, axis=0)
        theta = np.expand_dims(theta, axis=0)

    # Compute rotation components
    cos_theta = np.cos(theta)[:, None]
    sin_theta = np.sin(theta)[:, None]
    dot_product = np.einsum('ij,ij->i', axis, v)[:, None]

    # Apply rotation formula
    rotated_v = (
        cos_theta * v +
        sin_theta * np.cross(axis, v) +
        (1 - cos_theta) * dot_product * axis
    )

    # Remove batch dimension if input was non-batched
    if not batched:
        rotated_v = rotated_v.squeeze(axis=0)

    return rotated_v


def signed_angle(u: np.ndarray, v: np.ndarray, n: np.ndarray) -> np.ndarray:
    # Ensure inputs are at least 2D for batched processing
    batched = u.ndim > 1
    if not batched:
        u = np.expand_dims(u, axis=0)
        v = np.expand_dims(v, axis=0)
        n = np.expand_dims(n, axis=0)

    # Compute cross product and its norm
    w = np.cross(u, v)
    norm_w = np.linalg.norm(w, axis=-1, keepdims=False)

    # Compute dot product and handle near-zero denominators
    dot_uv = np.einsum('...i,...i', u, v)
    safe_denominator = np.where(np.abs(dot_uv) < 1e-10, 1.0, dot_uv)
    angle = np.arctan2(norm_w, safe_denominator)

    # Compute sign using the normal vector
    sign = np.sign(np.einsum('...i,...i', n, w))

    # Remove batch dimension if input was non-batched
    if not batched:
        angle = angle.squeeze(axis=0)
        sign = sign.squeeze(axis=0)

    return angle * sign


def construct_edge_combinations(edges: np.ndarray) -> np.ndarray:
    n = edges.shape[0]
    if n == 0:
        return np.array([])

    i, j = np.triu_indices(n, 1)
    mask = ~np.any((edges[i, None] == edges[j][:, None, :]) | (
        edges[i, None] == edges[j][:, None, ::-1]), axis=(1, 2))
    valid = np.column_stack((i[mask], j[mask]))
    return np.hstack((edges[valid[:, 0]], edges[valid[:, 1]]))

def construct_triangle_combinations(triangles: np.ndarray) -> np.ndarray:
    n = triangles.shape[0]
    if n == 0:
        return np.empty((0, 6), dtype=triangles.dtype)  # Return correct shape if empty

    i, j = np.triu_indices(n, 1) # Generate all unique pairs of triangle indices (i < j)

    # Check for shared nodes
    shared_node_mask = np.array([
        len(set(triangles[a]) & set(triangles[b])) > 0
        for a, b in zip(i, j)
    ], dtype=bool)

    valid = np.column_stack((i[~shared_node_mask], j[~shared_node_mask])) # Invert mask to get pairs with no shared nodes

    return np.hstack((triangles[valid[:, 0]], triangles[valid[:, 1]])) # Return stacked triangles (each row: triangle1 + triangle2 = 6 node indices)
