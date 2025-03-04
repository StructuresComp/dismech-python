import numpy as np


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
