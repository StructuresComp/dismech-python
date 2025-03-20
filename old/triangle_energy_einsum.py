def compute_dp(factor1, factor2):
            M13 = np.einsum('nij,nlja->nla', dp_coeff_1 *
                            factor1 + dp_coeff_3 * factor2, delfi)
            M2 = np.einsum('nij,nlia->nla', dp_coeff_2 * factor1, delfi)
            return (M2 + M13).reshape(-1, 9)

def compute_dpdp(factor1, factor2):
            # -ci_cj * s_xis_j * factor1 * ddelfi[k2, k1]
            M1 = np.einsum('nij,ntkjab->ntakb', dp_coeff_1 * factor1, ddelfi)
            # -ci_cj * factor1 * (delfi_j_k1.T @ delfi_i_k2)
            M2 = np.einsum('nij,nkjtiab->ntakb', ci_cj * factor1, delfi_sq)
            # -ci_cj * s_xis_i * factor1 * ddelfi[k2, k1]
            M3 = np.einsum('nij,ntkiab->ntakb', dp_coeff_2 * factor1, ddelfi)
            # -ci_cj * factor1 * (delfi_i_k1.T @ delfi_j_k2) (Transpose of M2)
            # 2 * ci_init_cj * factor2 * s_initxis_j * ddelfi[k2, k1]
            M5 = np.einsum('nij,ntkiab->ntakb', dp_coeff_3 * factor2, ddelfi)

            return (M1 + M2 + M3 + M2.transpose(0, 3, 4, 1, 2) + M5).reshape(N, 9, 9)