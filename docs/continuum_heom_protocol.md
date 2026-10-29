# Phase 1B: original bath audit and spinful fermionic HEOM

**Status: Implemented, tests for coefficient parsing written, remote HEOM execution not yet confirmed. Nothing here should be called a reproduction of the published PINN-DQME failure.**

## Primary sources

- Cao et al., *Simulating non-Markovian open quantum dynamics by exploiting physics-informed neural network*, J. Chem. Phys. 164, 194101 (2026), arXiv:2603.08081.
- Source code (pinned): https://github.com/caolong-cn/NQS-DQME_fermions/tree/7fb6ffb6115db42b299cc5c5092b27bfa9050628
- Original data: `examples/case1_T=0.3/res_corr.data` and `examples/pinn_case1_T=3.0/res_corr.data`.
- Original Hamiltonian inputs: `input_h` = [0, -0.5, -0.5, 1.0], which means two orbital energies -0.5 meV and U=1 meV in the source.
- Original chemical potentials: `chem_potentials.data` includes 0.00 and 0.05 meV.
- QuTiP fermionic HEOM documentation: https://qutip.readthedocs.io/en/stable/guide/heom/fermionic.html

## What we VERIFIED from original files

Both data files have dimensions (sign=2, spin=2, orbital=1, leads=2, terms=N).

- T/Gamma=3.0: N=4 exponentials per (sign,spin,orbital,lead) group; total 32 rows.
- T/Gamma=0.3: N=8 exponentials per group; total 64 rows.
- Slowest real decay rate at 3.0: 1.4318760879309518 ps^-1, time constant 0.6983844540940628 ps.
- Slowest real decay rate at 0.3: 0.14318758989079544 ps^-1, time constant 6.9838454628831155 ps.
- A tenfold slowest-mode time-constant ratio is supported by these files. It does not itself predict PINN failure.
- Upstream code uses hbar = 0.658211928 meV ps and converts local Hamiltonian energies by dividing by hbar.

Reproduce: `python scripts/upstream_bath_audit.py`. The workflow pulls data from the fixed upstream Git commit during the job, and saves machine-readable results.

## Intermediate QuTiP reference (different bath parameterization)

`python scripts/fermionic_heom_reference.py` implements the interacting local Hamiltonian (Jordan-Wigner spin-up and spin-down fermions) coupled to four continuum Lorentzian baths, one per spin and lead. Padé expansion and finite-depth hierarchy are handled by QuTiP.

The following are **provisional**:
- spectral coupling gamma = 0.05 meV per lead;
- Lorentzian width W = 5 meV;
- chemical potentials (0,0.05) meV;
- Gamma=0.1 meV scale used to translate T/Gamma labels into temperature.

**Even though these numbers are motivated by source conventions, the fitted bath coefficients have not been shown to match the authors' res_corr data.** QuTiP's fermionic correlation normalization and exponent/chemical-potential conventions must be mapped explicitly before a like-for-like comparison.

### Required checks before claiming a converged reference

1. Compare the QuTiP reconstructed C^+ and C^- with the original per-channel exponential coefficients, checking phase, sign, temperature and units.
2. Increase Padé order Nk while separately increasing hierarchy depth; don't confuse one kind of convergence with the other.
3. Decrease ODE integration tolerances and increase integration time resolution.
4. Cross-check initial density matrix, n_up/n_down, current sign and continuity equation, and positivity/trace where physically expected.
5. Evaluate the EXACT published PINN-DQME model/temperature/observable rather than another spinful Anderson variant.
6. Only then benchmark the published network against HEOM and investigate error-backflow-aware losses.

### Computational envelope

The default short time interval and shallow depth (1 and 2) are deliberately a *smoke test*, not a converged physical calculation. There are four fermionic baths, and raising Nk and depth substantially increases auxiliary density operator count. A moderate CPU job can test the setup, but a more difficult convergence study may require longer CPU runtime or substantial memory.

### Reproduction blocker

The upstream PINN tutorial loads a checkpoint absent at the GitHub path inspected; the upstream T/Gamma=0.3 example instead uses a variational RBM/TDVP driver. See [issue #2](https://github.com/HugoWEI/nonmarkovian-neural-dynamics/issues/2) for why a full PINN reproduction cannot yet be claimed.

No original deep-learning improvement has been demonstrated.
