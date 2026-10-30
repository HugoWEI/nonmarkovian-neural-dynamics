# Global-terminal quantum observable error control — mathematical and experimental audit

**Status:** Exact identity and candidate learning objective implemented. This is *not* a new theorem, and no neural advantage is claimed without verified comparative results.

## Dynamics and exact error identity

On the reduced finite-bath quantum benchmark, take a Hermitian single-particle matrix \(H\) and define

\[
\dot w(t)=A w(t),\qquad A=+\frac{i}{\hbar}H,\qquad U(t,s)=e^{A(t-s)}.
\]

We use the backward single-particle **column** evolved by \(+iH/\hbar\); because \(H\) is real symmetric and the dot begins empty, its squared components give the same occupation as the dot row of the usual forward propagator. That equivalence must not be presumed for arbitrary complex tunnelling phases without the appropriate transpose/conjugation.

For a piecewise differentiable neural solution \(\widehat w\), set \(r(t)=\dot{\widehat w}(t)-A\widehat w(t)\), \(e(t)=\widehat w(t)-w(t)\), and let \(J_j=\widehat w(t_j^+)-\widehat w(t_j^-)\) be any interface jumps. Then

\[
e(T)=U(T,0)e(0)+\sum_j U(T,t_j)J_j+\int_0^T U(T,t)r(t)\,dt.
\]

For the physical occupation \(O(w)=w^\dagger F w\), with \(F\) diagonal containing the fermionic Fermi occupations of the bath modes, we have **exactly**

\[
O(\widehat w(T))-O(w(T))
=2\operatorname{Re}\,[\widehat w(T)^\dagger F e(T)]
-e(T)^\dagger F e(T).
\]

The quadratic remainder is necessary and can dominate when the neural error is not small. If \(e(T)\) is obtained from the exact propagator, this identity is tautologically as accurate as solving the small model; that makes it a **verification control**, not a computational improvement.

The test module \`src/nonmarkovian_neural_dynamics/goal_error.py\` and \`tests/test_goal_error.py\` check this identity with a prescribed smooth residual and with a piecewise solution containing a controlled interface jump. The sign and quadratic remainder are included.

## Previously implemented local-window adjoint

Our first neural ablation used the end of **each window** as its goal, but judged occupation at the end of **all four windows**. These objectives are different. This is one reason local adjoint projection was not guaranteed to improve the final observable.

## New global-terminal *lookahead* objective

For the network currently training over \([t_j,t_{j+1}]\), form a forecast of the final state by propagating its predicted endpoint **without yet knowing future neural errors**:

\[
\widetilde w_T=U(T,t_{j+1})\widehat w(t_{j+1}).
\]

Form \(g_{j+1}=U(T,t_{j+1})^\dagger F\widetilde w_T\), then the projected residual sensitivity within this window is

\[
z_j(t)=U(t_{j+1},t)^\dagger g_{j+1}.
\]

The implemented loss compares the usual residual norm with either \(\int|{\rm Re}\,z^\dagger r|^2dt\) or \(|\int {\rm Re}\,z^\dagger r\,dt|^2\), using an architecture and optimizer identical to the existing piecewise baseline. This is a **global-terminal lookahead**, not the full exact propagated defect of all trained windows.

It uses a known quadratic propagator (matrix diagonalization). This provides exact state propagation for the small reduced model, which makes any claimed *computational speed advantage* particularly implausible. The experiment is to test the optimization principle and identify failure modes before approaching the expensive interacting system.

## Specific tests / falsification criteria

1. All three old objectives and both global-terminal variants, matched across **five seeds** and two temperatures.
2. Compare full-trajectory occupation RMSE, **terminal occupation error**, physical norm drift and **total measured runtime**.
3. Check exact residual/observable identities, including interface terms, before trusting any goal loss.
4. Report failures transparently; if global-terminal lookahead fails, do not infer a new research result from reduced trajectory RMSE.
5. Avoid novelty claims until compared with existing goal-oriented PINNs, dual-weighted residual methods and direct many-body solvers.

Source code: \`scripts/reduced_fermionic_pinn.py\`, \`scripts/global_goal_ablation.py\`. CI: \`.github/workflows/global-goal-ablation.yml\`.

**Scientific limitation:** This quadratic finite-bath system is an *independent validation problem*, not Cao et al.'s interacting, finite-temperature, correlated dissipaton model. A successful numerical experiment here would be motivation for the real benchmark, not proof of publication-level novelty.
