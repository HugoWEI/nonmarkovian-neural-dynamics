# First-order omitted-tier memory feedback: interacting HEOM predictive experiment

**Date:** October 30, 2026  
**Verified run:** https://github.com/HugoWEI/nonmarkovian-neural-dynamics/actions/runs/40983944363

Model: spinful interacting single-impurity Anderson Hamiltonian with U=1meV, epsilon_up/down=-0.5meV, one Lorentzian fermionic reservoir per spin, Gamma=0.1meV, bandwidth 5meV, 6ps quench, low-temperature source-Padé Nk7 / high-temperature Nk3. The initial hierarchy is the **low-depth correlated steady state**, embedded into high depth with the omitted tiers initially zero. This is a mathematically controlled common initial condition, **not** the high-depth fully correlated equilibrium state of Cao et al.

## What is predicted

Partition the high-depth HEOM generator in retained and omitted ADO blocks:

\[
L_\mathrm{high}=\begin{pmatrix}A&B\\C&D\end{pmatrix},\quad
x'=Ax+Bu,\quad u'=Cx+Du.
\]

Our first-order method solves the **low-depth** state x0'=Ax0 and a driven omitted-tier equation u1'=Du1+Cx0 (with u1(0)=0). It estimates the impact on final occupation via the **low-depth adjoint**:

\[
\delta O_\mathrm{first}(T)=
\operatorname{Re}\int_0^T z_A(t)^\dagger B u_1(t)\,dt,
\qquad z_A(t)=e^{A^\dagger(T-t)}c.
\]

The full high-depth occupation is computed **only to validate** this approximation. In contrast to the prior full-high-depth dual reconstruction, this predictor uses no high-depth adjoint and no high-depth correlated steady-state preparation. It still constructs high-depth sparse B,C,D blocks; their construction cost must be counted.

## Verified results

| T/Gamma | Nk | Low-depth occupation | High-depth withheld reference | First-order correction | Uncorrected error | Corrected error | Error reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3.0 | 3 | 0.510815674 | 0.510872163 | +0.000060986 | 0.000056490 | 0.000004496 | **92.0%** |
| 0.3 | 3 | 0.500514948 | 0.501837211 | +0.001655464 | 0.001322264 | 0.000333201 | **74.8%** |
| 0.3 | 7 | 0.502192623 | 0.503461947 | +0.001610202 | 0.001269324 | 0.000340879 | **73.1%** |

### Cost and limitations

The initial low Nk7 test measured lower correlated steady preparation ~0.82 CPU seconds, first-order predictor evolution ~0.71 seconds, and independent high-depth forward validation ~1.08 seconds. However these components were **not** complete end-to-end timings because the high-depth generator construction was outside the predictor timer. A revised workflow and a prespecified voltage-strength stress test now measure complete costs separately.

This is not a new memory-kernel theory. Closely related preexisting work includes:

- Vadimov, **On truncations of hierarchical equations of motion for finite-dimensional systems**, SciPost Physics Core 9, 062 (2026), arXiv:2604.22568 — Schur-complement-type HEOM terminators and spectral convergence.
  https://doi.org/10.21468/SciPostPhysCore.9.3.062
- Krug & Stockburger, **On stability issues of the HEOM method** (2024), arXiv:2408.05520.
- Adriazola & Roszak, **Learning Volterra memory kernels for non-Markovian qubit dynamics**, Phys. Rev. A accepted September 23, 2026, DOI:10.1103/bc2w-c6dj.
- **On the exact truncation tier of fermionic hierarchical equations of motion** (2018), https://pubmed.ncbi.nlm.nih.gov/29935503/.

Our result therefore validates a **classical first-order boundary-memory correction in a specific interacting fermionic benchmark**. It does not demonstrate a novel deep-learning algorithm or a total computational speedup. The remaining scientific question is whether a learned fast approximation to high-tier feedback can outperform classical memory-kernel/HEOM terminators at equal end-to-end cost and under independently verified physical initial conditions.
