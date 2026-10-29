# Published bath-correlation audit: checked directly against upstream GitHub files

**Date:** 2026-10-29  
**Upstream Git revision:** `7fb6ffb6115db42b299cc5c5092b27bfa9050628`  
**Status:** Direct source-data check, **not** a HEOM or PINN-DQME reproduction.

| Source file | Correlation dimensions | Rows | Slowest decay rate | Slowest time constant |
|---|---|---:|---:|---:|
| `examples/case1_T=0.3/res_corr.data` | (2 signs, 2 spins, 1 orbital, 2 leads, 8 terms) | 64 | 0.14318758989 ps^-1 | 6.98384546288 ps |
| `examples/pinn_case1_T=3.0/res_corr.data` | (2 signs, 2 spins, 1 orbital, 2 leads, 4 terms) | 32 | 1.43187608793 ps^-1 | 0.69838445409 ps |

For each file, the slowest rate is the same across its eight (sign, spin, orbital, lead) groups. The rate ratio high/low is approximately 10.0.

The checked source uses complex coefficients `eta_k` and exponents `gamma_k` with negative real parts. A fitted channel correlation is represented as `C(t)=sum_k eta_k exp(gamma_k t)`, subject to the original source's sign and normalization conventions. The executable audit re-fetches the original data at its pinned revision and saves a per-channel table.

**Interpretation:** Low temperature has a tenfold slower fitted environmental mode in this source. This motivates a harder non-Markovian calculation, but does not prove anything about PINN error propagation, solve a quantum master equation, or establish a learning advantage.

**Next:** run the [HEOM workflow](../.github/workflows/continuum-heom.yml), verify QuTiP's bath normalization against the coefficients, then perform depth/Padé convergence tests.
