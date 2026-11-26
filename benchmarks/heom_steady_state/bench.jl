# HEOM stationary-state benchmark in Julia: HierarchicalEOM.jl's own steadystate
# vs exact deepest-tier elimination applied to the same Julia generator.
#
# Same model as bench.py: spinful Anderson impurity, U = 1 meV, eps = -0.5 meV,
# one Lorentzian lead per spin (Gamma = 0.1 meV, W = 5 meV), Pade with Nk poles,
# T/Gamma = 0.3, mu = 0.05 meV. Energies in meV/hbar (1/ps), as in the repository.
#
# Run from this folder:  julia --project=. bench.jl          (default cases)
#                        julia --project=. bench.jl --big    (adds Nk=7, depth 4)
# It first checks that the Julia bath has the same exponents as QuTiP's
# (reference.json) and reproduces the reference <n_up>, then times the solvers.

using HierarchicalEOM, SparseArrays, LinearAlgebra, JSON, Printf

const HBAR = 0.658211928
const GAMMA, W, U, EPS, MU, RATIO = 0.1, 5.0, 1.0, -0.5, 0.05, 0.3
const REF = JSON.parsefile(joinpath(@__DIR__, "reference.json"))
const CASES = ("--big" in ARGS) ? [(3,3), (3,4), (7,3), (5,4), (7,4)] :
              ("--quick" in ARGS) ? [(3,3)] : [(3,3), (3,4), (7,3), (5,4)]

d_up = tensor(destroy(2), qeye(2))
d_dn = tensor(sigmaz(), destroy(2))          # Jordan-Wigner, as in QuTiP
n_up = d_up' * d_up
n_dn = d_dn' * d_dn
Hsys = (EPS / HBAR) * (n_up + n_dn) + (U / HBAR) * n_up * n_dn

# ---- 1. Find the coupling convention that gives QuTiP's exact bath ----------
function exps(b)
    out = ComplexF64[]
    for term in b.bath, e in term.η
        push!(out, ComplexF64(e))
    end
    out
end
ref_sum(key) = sum(abs(complex(e[2][1], e[2][2])) for e in REF[key]["exponents"])
mkbath(op, λ, nk) = Fermion_Lorentz_Pade(op, λ, MU / HBAR, W / HBAR, RATIO * GAMMA / HBAR, nk)

g = GAMMA / HBAR
candidates = ["sqrt(Γ)" => sqrt(g), "Γ" => g, "sqrt(Γ/2)" => sqrt(g / 2),
              "sqrt(2Γ)" => sqrt(2g), "sqrt(Γ/2π)" => sqrt(g / 2π)]
target = ref_sum("3,3")
λname, λ = "", NaN
for (name, val) in candidates
    s = sum(abs, exps(mkbath(d_up, val, 3)))
    @printf("coupling %-10s  sum|η| = %.12g   (QuTiP %.12g)\n", name, s, target)
    if abs(s - target) < 1e-8 * target
        global λname, λ = name, val
    end
end
isnan(λ) && error("No coupling convention reproduces QuTiP's bath exponents; send me the lines above.")
println("Using λ = $λname: Julia bath exponents match QuTiP's.\n")

# ---- 2. Helpers -------------------------------------------------------------
function generator(M)
    d = M.data
    for _ in 1:4
        d isa SparseMatrixCSC && return d
        d isa AbstractMatrix && return sparse(d)
        if hasproperty(d, :data); d = getproperty(d, :data)
        elseif hasproperty(d, :A); d = getproperty(d, :A)
        else break end
    end
    error("Could not extract the HEOM generator matrix (type $(typeof(M.data))).")
end

tier(n) = hasproperty(n, :level) ? n.level : sum(n.data)

function with_trace(S, dim)
    r, c, v = findnz(S)
    keep = r .!= 1
    r, c, v = r[keep], c[keep], v[keep]
    tr = [k * (dim + 1) + 1 for k in 0:dim-1]
    append!(r, ones(Int, dim)); append!(c, tr); append!(v, ones(ComplexF64, dim))
    A = sparse(r, c, v, size(S)...)
    b = zeros(ComplexF64, size(S, 1)); b[1] = 1
    A, b
end

function eliminated(L, M, dim)
    blk = dim^2
    tiers = [tier(n) for n in M.hierarchy.idx2nvec]
    top = maximum(tiers)
    expand(ix) = vec([(i - 1) * blk + j for j in 1:blk, i in ix])
    keep = expand(findall(<(top), tiers)); last = expand(findall(==(top), tiers))
    A = L[keep, keep]; B = L[keep, last]; C = L[last, keep]; Dm = L[last, last]
    nnz(Dm - Diagonal(diag(Dm))) == 0 || error("deepest-tier block is not diagonal")
    Dinv = Diagonal(1 ./ diag(Dm))
    S = A - B * Dinv * C
    As, b = with_trace(S, dim)
    x = As \ b
    y = zeros(ComplexF64, size(L, 1))
    y[keep] = x; y[last] = -(Dinv * (C * x))
    y
end

nup_of(y, dim) = real(tr(Matrix(n_up.data) * reshape(y[1:dim^2], dim, dim)))
mem_gb() = round(Sys.maxrss() / 2^30, digits = 2)

# ---- 3. Benchmark -----------------------------------------------------------
results = Dict{String,Any}("machine" => Sys.MACHINE, "cpu" => Sys.cpu_info()[1].model,
    "threads" => Threads.nthreads(), "ram_gb" => round(Sys.total_memory() / 2^30, digits = 1),
    "julia" => string(VERSION), "coupling" => λname, "cases" => Any[])

warm = M_Fermion(Hsys, 2, [mkbath(d_up, λ, 3), mkbath(d_dn, λ, 3)]; verbose = false)
steadystate(warm; verbose = false); eliminated(generator(warm), warm, 4)   # compile once

for (nk, depth) in CASES
    key = "$nk,$depth"
    t = @elapsed M = M_Fermion(Hsys, depth, [mkbath(d_up, λ, nk), mkbath(d_dn, λ, nk)]; verbose = false)
    L = generator(M)
    rec = Dict{String,Any}("Nk" => nk, "depth" => depth, "ados" => M.N, "unknowns" => size(L, 1),
                           "build_s" => round(t, digits = 3))
    t = @elapsed y = eliminated(L, M, 4)
    rec["eliminated_s"] = round(t, digits = 3)
    rec["residual"] = norm(L * y)
    rec["n_up_eliminated"] = nup_of(y, 4)
    try
        t = @elapsed ados = steadystate(M; verbose = false)
        rec["julia_steadystate_s"] = round(t, digits = 3)
        rec["n_up_julia"] = real(expect(n_up, getRho(ados)))
    catch e
        rec["julia_steadystate_s"] = e isa OutOfMemoryError ? "out of memory" : "error: $(typeof(e))"
    end
    haskey(REF, key) && (rec["n_up_reference_qutip"] = REF[key]["n_up"])
    rec["peak_memory_gb"] = mem_gb()
    push!(results["cases"], rec)
    println(JSON.json(rec)); flush(stdout)
    open(joinpath(@__DIR__, "results_julia.json"), "w") do io
        JSON.print(io, results, 1)
    end
end
println("done; results in results_julia.json")
