# Quickstart: run from julia/HEOMTierElimination with
#   julia --project=. examples/quickstart.jl
using HierarchicalEOM, HEOMTierElimination

HBAR = 0.658211928                                   # meV ps
d_up = tensor(destroy(2), qeye(2))
d_dn = tensor(sigmaz(), destroy(2))                  # Jordan-Wigner
n_up, n_dn = d_up' * d_up, d_dn' * d_dn
H = (-0.5 / HBAR) * (n_up + n_dn) + (1.0 / HBAR) * n_up * n_dn
bath(op) = Fermion_Lorentz_Pade(op, 0.1 / HBAR, 0.05 / HBAR, 5.0 / HBAR, 0.03 / HBAR, 5)
M = M_Fermion(H, 4, [bath(d_up), bath(d_dn)]; verbose = false)   # 207,216 unknowns

steadystate(M; verbose = false); steadystate_eliminated(M)        # compile once

t1 = @elapsed ref = steadystate(M; verbose = false)
t2 = @elapsed ados = steadystate_eliminated(M; verbose = true)
println("HierarchicalEOM steadystate : <n_up> = ", real(expect(n_up, getRho(ref))), "  (", round(t1, digits = 2), " s)")
println("steadystate_eliminated      : <n_up> = ", real(expect(n_up, getRho(ados))), "  (", round(t2, digits = 2), " s)")
