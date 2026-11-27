# HEOMTierElimination.jl

Exact deepest-tier elimination for steady states of [HierarchicalEOM.jl](https://github.com/qutip/HierarchicalEOM.jl) models: a drop-in alternative to `steadystate(M)`.

## Install

In the Julia REPL:

```julia
using Pkg
Pkg.add(url = "https://github.com/HugoWEI/nonmarkovian-neural-dynamics", subdir = "julia/HEOMTierElimination")
```

## Use

```julia
using HierarchicalEOM, HEOMTierElimination

M = M_Fermion(H, 4, baths)               # or M_Boson, M_Boson_Fermion
ados = steadystate_eliminated(M)         # GMRES on the reduced system (default)
ados = steadystate_eliminated(M; method = :direct)   # exact LU, small/medium sizes
ρ = getRho(ados)
```

Options: `rtol = 1e-12`, `atol = 1e-14` (same defaults as `steadystate`), `restart = 0` (set e.g. `100` to bound GMRES memory on very large hierarchies), `verbose = false`.

## Why it is faster

ADOs at the maximal hierarchy level couple only to the level below, never to each other, so at stationarity they can be removed exactly. They are most of the hierarchy (72–90% in our fermionic benchmarks), so GMRES works on vectors 4–10× shorter. On a spinful Anderson impurity this made the default steady-state solve 2.2–16.8× faster at equal accuracy, and 4.4× faster at 3.9 million unknowns. See the [paper](../../paper/main.pdf) for details and limitations.

Lazy (`assemble = :none`/`:combine`) operators are not supported: the matrix must be assembled (the default).

## Test

```julia
using Pkg; Pkg.test("HEOMTierElimination")
```
