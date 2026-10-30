# Correcting the published source-state and lead-phase comparison

**Date:** October 30, 2026. **Status:** Upstream source inspected; GitHub validation workflow written, remote execution not confirmed.

## Finding 1 — The stored impurity is not empty

The first four zero-tier diagonal elements of the original `table_cut3.data` files are:

| T/Gamma | p_empty | p_up | p_down | p_double | n_total |
|---:|---:|---:|---:|---:|---:|
| 0.3 | 0.795495093033 | 0.091388796876 | 0.091388796875 | 0.021727313213 | 0.226232220176 |
| 3.0 | 0.457253555565 | 0.245690450178 | 0.245690450178 | 0.051365544079 | 0.594111988515 |

These are **stored reduced states**, not automatically the `t=0` state of a particular PINN trajectory. The upstream example also contains nonzero higher-tier dissipaton information. The original PINN tutorial references a pretrained checkpoint to initialize the `t=4.5` subdomain, which is absent from the path checked.

Our preliminary QuTiP HEOM script starts with an empty dot `|00><00|`. Consequently, **its occupation trace cannot be compared directly to the authors' stored state or published reference trajectory**, even if its bath expansion is correct. Giving QuTiP only the reduced diagonal while zeroing higher hierarchy ADOs would also discard system-bath correlations.

Reproducible checks: `scripts/upstream_initial_state_audit.py`, `tests/test_upstream_initial_state.py`, original pinned data in `examples/case1_T=0.3/table_cut3.data` and `examples/pinn_case1_T=3.0/table_cut3.data`.

## Finding 2 — Original chemical-potential phase is in the Liouvillian, not in res_corr.data

From upstream `src/nqsdqme/core/liouville.py`:

```python
mu[a1,a2,a3,a4,a5] = 1.j*(-1)**(a1+1)*chem_potentials[a4,a2]
self.gamma = self.gamma + mu.flatten()/hbar
```

The source coefficients themselves are shared between leads at a fixed temperature; chemical potential introduces a sign-dependent imaginary shift in the exponent. For 0-based sign index `a1=0`, shift is `-i mu/hbar`, while for `a1=1` it is `+i mu/hbar`.

In QuTiP's notation, `C_sigma(t) = sum ck exp(-vk t)` with `vk = rate - sigma i mu/hbar`. Hence the original sign-index order maps onto the two QuTiP exponent sectors in reverse (source first sign corresponds to negative phase, second sign to positive phase). The exact labels used to identify physical C+ and C- should be checked against their operator definitions, not inferred from array order alone.

The new `scripts/verify_quutip_correlations.py` evaluates both sign sectors in the built-in QuTiP `LorentzianPadeBath` and the independently reconstructed `FermionicBath`. It compares them for `T/Gamma=0.3,3.0` and chemical potentials 0 and 0.05 meV, enforcing max error below 1e-8.

## Next scientific gate

A source-matched **reservoir correlation** is necessary but insufficient. Before claiming a reproduction of Cao et al., we still require:
1. Confirm QuTiP correlation sign conventions (automated test).
2. Obtain a physically equivalent **full initial ADO hierarchy** (not just four diagonal populations).
3. Independently converge Padé order, hierarchy depth, and solver tolerance.
4. Reproduce the specific neural low-temperature observable failure.
5. Only then benchmark an observable-specific neural error-backflow objective.

No novel deep-learning solver result is claimed.
