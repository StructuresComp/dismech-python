import numpy as np
import numba


@numba.njit(numba.float64[:, :](
    numba.float64[:, :, :],  # dp_coeff1
    numba.float64[:, :, :],  # dp_coeff2
    numba.float64[:, :, :],  # dp_coeff3
    numba.float64[:, :, :],  # factor1
    numba.float64[:, :, :],  # factor2
    numba.float64[:, :, :, :],  # delfi
))
def compute_dp_jit(dp_coeff1, dp_coeff2, dp_coeff3, factor1, factor2, delfi):
    N = delfi.shape[0]
    dp = np.empty((N, 3, 3))

    for n in range(N):
        for l in range(3):
            for a in range(3):
                val = 0.0
                for i in range(3):
                    for j in range(3):
                        # M13 term
                        val += delfi[n, l, j, a] * (
                            dp_coeff1[n, i, j] * factor1[n, i, j] +
                            dp_coeff3[n, i, j] * factor2[n, i, j]
                        )
                # add M2 term
                for i in range(3):
                    for j in range(3):
                        val += delfi[n, l, i, a] * (
                            dp_coeff2[n, i, j] * factor1[n, i, j]
                        )
                dp[n, l, a] = val
    return dp.reshape(N, 9)


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

    for n in range(N):
        # First handle the symmetric part (M2 + M4 + M5)
        for p in range(9):
            for q in range(p, 9):  # Only upper triangle
                t, a = divmod(p, 3)
                k, b = divmod(q, 3)

                sym_value = 0.0
                for i in range(3):
                    for j in range(3):
                        ret = ci_cj[n, i, j] * factor1[n, i, j] * delfi_sq[n, k, j, t, i, a, b] + \
                            ci_cj[n, j, i] * factor1[n, j, i] * \
                            delfi_sq[n, t, i, k, j, a, b]

                        ret += dp_coeff3[n, i, j] * factor2[n,
                                                            i, j] * ddelfi[n, t, k, i, a, b]

                        sym_value += ret

                result[n, p, q] = sym_value
                if p != q:
                    result[n, q, p] = sym_value  # Mirror to lower triangle

        # Now handle the asymmetric parts (M1 + M3)
        for p in range(9):
            for q in range(9):
                t, a = divmod(p, 3)
                k, b = divmod(q, 3)

                asym_value = 0.0
                for i in range(3):
                    for j in range(3):
                        asym_value += dp_coeff1[n, i, j] * \
                            factor1[n, i, j] * ddelfi[n, t, k, j, a, b]

                        asym_value += dp_coeff2[n, i, j] * \
                            factor1[n, i, j] * ddelfi[n, t, k, i, a, b]

                result[n, p, q] += asym_value

    return result
