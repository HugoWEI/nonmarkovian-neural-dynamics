# HEOM steady-state benchmark

Compares ways of computing the correlated stationary state of the HEOM hierarchy
(spinful Anderson impurity, U = 1 meV, ε = −U/2, one Lorentzian lead per spin,
Γ = 0.1 meV, W = 5 meV, T/Γ = 0.3, μ = 0.05 meV, Padé expansion with Nk poles).

| Script | What it does |
|---|---|
| `bench.py` | QuTiP direct solve vs exact deepest-tier elimination (direct) |
| `bench.jl` | HierarchicalEOM.jl default `steadystate` vs elimination on the Julia generator; first checks the Julia bath equals QuTiP's |
| `bench_iter.jl` | Same GMRES (Krylov.jl, rtol 1e-12, atol 1e-14) on the full hierarchy vs on the eliminated system (explicit and matrix-free), plus exact direct solves |
| `plot_benchmark.py` | Builds `results/speed_comparison.{svg,png,pdf}` from the results |
| `reference.json` | QuTiP bath exponents and reference ⟨n↑⟩ values |

`results/laptop_2026-11-26.json` holds the numbers from the runs on Hugo's laptop
(Windows 11, 31.7 GB RAM), transcribed from the console output.

## Run (PowerShell)

```
julia --project=. -e 'using Pkg; Pkg.add(string.([:HierarchicalEOM, :JSON, :Krylov]))'
julia --project=. -t auto bench.jl --big
julia --project=. -t auto bench_iter.jl --big
julia --project=. -t auto bench_iter.jl 5,5
julia --project=. -t auto bench_iter.jl 7,5 --restart=100
pip install qutip
python bench.py
```
