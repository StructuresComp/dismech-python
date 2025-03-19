import numpy as np
import numba


@numba.njit(numba.float64[:, :, :, :, :, :, :](numba.float64[:, :, :, :]))
def compute_delfi_sq_jit(delfi):
    N = delfi.shape[0]

    delfi_sq = np.empty((N, 3, 3, 3, 3, 3, 3))
    for n in range(N):
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for a in range(3):
                        for b in range(3):
                            for c in range(3):
                                delfi_sq[n, i, j, a, b, k, c] = delfi[n,
                                                                      i, j, k] * delfi[n, a, b, c]
    return delfi_sq

@numba.njit(numba.float64[:, :, :](   # Return type (N, 9, 9)
    numba.float64[:, :, :],  # dp_coeff1
    numba.float64[:, :, :],  # dp_coeff2
    numba.float64[:, :, :],  # dp_coeff3
    numba.float64[:, :, :],  # factor1
    numba.float64[:, :, :],  # factor2
    numba.float64[:, :, :, :, :, :],  # ddelfi
    numba.float64[:, :, :, :, :, :, :],  # delfi_sq
    numba.float64[:, :, :]   # ci_cj
))
def compute_dpdp_jit(dp_coeff1, dp_coeff2, dp_coeff3, factor1, factor2, ddelfi, delfi_sq, ci_cj):
    N = dp_coeff1.shape[0]
    result = np.empty((N, 9, 9))
    result.fill(0.0)

    for n in range(N):
        for t in range(3):
            for a in range(3):
                for k in range(3):
                    for b in range(3):
                        p = t * 3 + a
                        q = k * 3 + b
                        for i in range(3):
                            for j in range(3):
                                # Compute M1: -ci_cj * s_xis_j * factor1 * ddelfi[k2, k1]
                                result[n, p, q] += dp_coeff1[n, i, j] * factor1[n, i, j] * \
                                    ddelfi[n, t, k, j, a, b]

                                # Compute M2: -ci_cj * factor1 * (delfi_j_k1.T @ delfi_i_k2) and its transpose
                                ret = ci_cj[n, i, j] * factor1[n, i, j] * \
                                    delfi_sq[n, k, j, t, i, a, b]
                                result[n, p, q] += ret
                                result[n, q, p] += ret

                                # Compute M3: -ci_cj * s_xis_i * factor1 * ddelfi[k2, k1]
                                result[n, p, q] += dp_coeff2[n, i, j] * factor1[n, i, j] * \
                                    ddelfi[n, t, k, i, a, b]

                                # Compute M5: 2 * ci_init_cj * factor2 * s_initxis_j * ddelfi[k2, k1]
                                result[n, p, q] += dp_coeff3[n, i, j] * factor2[n, i, j] * \
                                    ddelfi[n, t, k, i, a, b]

    return result

@numba.njit
def compute_dpdp_jit_mirror(dp_coeff1, dp_coeff2, dp_coeff3, factor1, factor2, ddelfi, delfi_sq, ci_cj):
    N = dp_coeff1.shape[0]
    result = np.empty((N, 9, 9))
    result.fill(0.0)

    for n in range(N):
        for t in range(3):
            for a in range(3):
                for k in range(3):
                    for b in range(3):
                        p = t * 3 + a
                        q = k * 3 + b
                        # Only update if we're in the upper triangle
                        if p <= q:
                            for i in range(3):
                                for j in range(3):
                                    # M1: dp_coeff1 contribution
                                    result[n, p, q] += dp_coeff1[n, i, j] * factor1[n, i, j] * \
                                        ddelfi[n, t, k, j, a, b]

                                    # M2: symmetric contribution from delfi_sq (2 * for M4)
                                    if p == q:
                                        result[n, p, q] += 2 * ci_cj[n, i, j] * factor1[n, i, j] * \
                                            delfi_sq[n, k, j, t, i, a, b]
                                    else:
                                        result[n, p, q] += ci_cj[n, i, j] * factor1[n, i, j] * delfi_sq[n, k, j, t, i, a, b] + \
                                            ci_cj[n, i, j] * factor1[n, i, j] * delfi_sq[n, t, j, k, i, b, a]

                                    # M3: dp_coeff2 contribution
                                    result[n, p, q] += dp_coeff2[n, i, j] * factor1[n, i, j] * \
                                        ddelfi[n, t, k, i, a, b]

                                    # M5: dp_coeff3 contribution
                                    result[n, p, q] += dp_coeff3[n, i, j] * factor2[n, i, j] * \
                                        ddelfi[n, t, k, i, a, b]

    # Mirror
    for n in range(N):
        for i in range(9):
            for j in range(i + 1, 9):
                result[n, j, i] = result[n, i, j]

    return result