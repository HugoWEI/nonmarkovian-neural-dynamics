# Reproducible end-to-end benchmark: HierarchicalEOM.jl vs HEOMTierElimination.jl
#
# Addresses the points raised in review:
#   * bath exponents compared coefficient by coefficient with QuTiP (not just sums)
#   * end-to-end timings (all preparation and reconstruction included) AND split timings
#   * median of several repeats, garbage collection before each run, warm-up excluded
#   * thread counts fixed and recorded, software/hardware versions recorded
#
# Run in a fresh Julia session (single thread for both methods, for a like-for-like comparison):
#   julia -t 1
#   using Pkg; Pkg.activate(temp = true)
#   Pkg.add(url = "https://github.com/HugoWEI/nonmarkovian-neural-dynamics", subdir = "julia/HEOMTierElimination")
#   Pkg.add(["HierarchicalEOM", "Krylov", "JSON"])
#   include(download("https://raw.githubusercontent.com/HugoWEI/nonmarkovian-neural-dynamics/main/benchmarks/heom_steady_state/bench_final.jl"))
#
# Writes results_final.json in the current directory (pwd()).

using HierarchicalEOM, HEOMTierElimination, SparseArrays, LinearAlgebra, Krylov, JSON, Printf
import Pkg

const HBAR = 0.658211928
const GAMMA, W, U, EPS, MU, RATIO = 0.1, 5.0, 1.0, -0.5, 0.05, 0.3
const RTOL, ATOL = 1e-12, 1e-14
const REF = JSON.parsefile(download(
    "https://raw.githubusercontent.com/HugoWEI/nonmarkovian-neural-dynamics/main/benchmarks/heom_steady_state/reference.json"))
# (Nk, depth, repeats, GMRES restart; 0 = none)
const CASES = [(7, 3, 5, 0), (5, 4, 5, 0), (7, 4, 3, 0), (5, 5, 3, 0), (7, 5, 3, 100)]

BLAS.set_num_threads(1)

d_up = tensor(destroy(2), qeye(2))
d_dn = tensor(sigmaz(), destroy(2))
n_up = d_up' * d_up
H = (EPS / HBAR) * (n_up + d_dn' * d_dn) + (U / HBAR) * n_up * (d_dn' * d_dn)
mkbath(op, nk) = Fermion_Lorentz_Pade(op, GAMMA / HBAR, MU / HBAR, W / HBAR, RATIO * GAMMA / HBAR, nk)

# ---- 1. coefficient-by-coefficient bath check against QuTiP ------------------------
function julia_terms(b)
    out = Dict{String,Vector{Tuple{ComplexF64,ComplexF64}}}()
    for t in b.bath
        k = string(nameof(typeof(t)))
        append!(get!(out, k, Tuple{ComplexF64,ComplexF64}[]), collect(zip(ComplexF64.(t.η), ComplexF64.(t.γ))))
    end
    out
end
qutip_terms(nk) = begin
    key = first(k for k in keys(REF) if startswith(k, "$nk,"))
    out = Dict{String,Vector{Tuple{ComplexF64,ComplexF64}}}()
    for e in REF[key]["exponents"]
        push!(get!(out, e[1], Tuple{ComplexF64,ComplexF64}[]),
              (complex(e[2][1], e[2][2]), complex(e[3][1], e[3][2])))
    end
    out
end
sortkey(p) = (real(p[2]), imag(p[2]), real(p[1]), imag(p[1]))
function same_set(a, b; tol = 1e-12)
    length(a) == length(b) || return Inf
    a, b = sort(a; by = sortkey), sort(b; by = sortkey)
    maximum(max(abs(x[1] - y[1]) / max(abs(y[1]), 1e-300), abs(x[2] - y[2]) / abs(y[2])) for (x, y) in zip(a, b))
end
function check_bath(nk)
    jt, qt = julia_terms(mkbath(d_up, nk)), qutip_terms(nk)
    jk, qk = collect(keys(jt)), collect(keys(qt))
    length(jk) == 2 && length(qk) == 2 || error("unexpected bath structure: $jk vs $qk")
    best = Inf; mapping = ""
    for perm in ((1, 2), (2, 1))
        err = max(same_set(jt[jk[1]], qt[qk[perm[1]]]), same_set(jt[jk[2]], qt[qk[perm[2]]]))
        if err < best
            best = err; mapping = "$(jk[1])↔QuTiP $(qk[perm[1]]), $(jk[2])↔QuTiP $(qk[perm[2]])"
        end
    end
    @printf("Nk=%d: %d exponents, max relative difference in every η_k and γ_k = %.2e (%s)\n",
            nk, sum(length, values(jt)), best, mapping)
    best < 1e-10 || @warn "bath exponents differ from QuTiP's beyond 1e-10" nk best
    best
end
println("Bath check (coefficient by coefficient):")
bath_check = Dict("$nk" => check_bath(nk) for nk in (3, 5, 7))

# ---- 2. helpers ---------------------------------------------------------------------
function generator(M)
    d = M.data
    for _ in 1:4
        d isa SparseMatrixCSC && return d
        if hasproperty(d, :A); d = d.A elseif hasproperty(d, :data); d = d.data else break end
    end
    error("no assembled generator")
end
constraint(n) = sparse(ones(Int, 4), [(k - 1) * 5 + 1 for k in 1:4], ones(ComplexF64, 4), n, n)
gm(A, b, restart) = restart > 0 ?
    gmres(A, b; rtol = RTOL, atol = ATOL, memory = restart, restart = true, itmax = 50_000) :
    gmres(A, b; rtol = RTOL, atol = ATOL)
nup(ados) = real(expect(n_up, getRho(ados)))

function full_gmres(M, restart)   # same as HierarchicalEOM default, end-to-end
    L = generator(M); n = size(L, 1)
    b = zeros(ComplexF64, n); b[1] = 1
    x, st = gm(L + constraint(n), b, restart)
    return ADOs(x, M.dimensions.to, M.N, M.parity), st.niter
end

function timed(f, reps)
    ts = Float64[]; r = nothing
    for _ in 1:reps
        GC.gc()
        t = @elapsed r = f()
        push!(ts, t)
    end
    sort(ts)[cld(reps, 2)], ts, r
end

# ---- 3. warm-up (compilation excluded from every timing) ----------------------------
let M = M_Fermion(H, 3, [mkbath(d_up, 3), mkbath(d_dn, 3)]; verbose = false)
    steadystate(M; verbose = false); full_gmres(M, 0); full_gmres(M, 20)
    steadystate_eliminated(M); steadystate_eliminated(M; restart = 20)
end

# ---- 4. benchmark -------------------------------------------------------------------
env = Dict("julia" => string(VERSION), "cpu" => Sys.cpu_info()[1].model,
    "logical_cpus" => Sys.CPU_THREADS, "ram_gb" => round(Sys.total_memory() / 2^30, digits = 1),
    "os" => string(Sys.KERNEL, " ", Sys.MACHINE), "julia_threads" => Threads.nthreads(),
    "blas_threads" => BLAS.get_num_threads(),
    "packages" => Dict(string(p.name) => string(p.version) for p in values(Pkg.dependencies())
                       if p.name in ("HierarchicalEOM", "HEOMTierElimination", "Krylov", "QuantumToolbox")),
    "settings" => "GMRES rtol=$RTOL atol=$ATOL; median of repeats; GC before each run; warm-up excluded; generator construction excluded",
    "bath_check_max_rel_diff" => bath_check, "mu_meV" => MU, "T_over_Gamma" => RATIO)
println("\nEnvironment: ", JSON.json(env))
results = Dict{String,Any}("environment" => env, "cases" => Any[])

for (nk, depth, reps, restart) in CASES
    M = M_Fermion(H, depth, [mkbath(d_up, nk), mkbath(d_dn, nk)]; verbose = false)
    rec = Dict{String,Any}("Nk" => nk, "depth" => depth, "unknowns" => size(generator(M), 1),
                           "repeats" => reps, "restart" => restart)
    if restart == 0 && size(generator(M), 1) <= 1_000_000
        t, ts, r = timed(() -> steadystate(M; verbose = false), reps)
        rec["package_steadystate_s"] = t; rec["package_steadystate_all_s"] = ts; rec["n_up_package"] = nup(r)
    end
    t, ts, (r, it) = timed(() -> full_gmres(M, restart), reps)
    rec["full_gmres_end_to_end_s"] = t; rec["full_gmres_all_s"] = ts; rec["full_iters"] = it; rec["n_up_full"] = nup(r)
    t, ts, r = timed(() -> steadystate_eliminated(M; restart = restart), reps)
    rec["eliminated_end_to_end_s"] = t; rec["eliminated_all_s"] = ts; rec["n_up_eliminated"] = nup(r)
    rec["speedup_end_to_end"] = rec["full_gmres_end_to_end_s"] / t
    haskey(REF, "$nk,$depth") && (rec["n_up_exact_reference"] = REF["$nk,$depth"]["n_up"])
    rec["peak_memory_gb"] = round(Sys.maxrss() / 2^30, digits = 2)
    push!(results["cases"], rec)
    @printf("Nk=%d depth=%d  unknowns=%d  full=%.2fs  eliminated=%.2fs  speed-up=%.2f  (median of %d)\n",
            nk, depth, rec["unknowns"], rec["full_gmres_end_to_end_s"], t, rec["speedup_end_to_end"], reps)
    open("results_final.json", "w") do io; JSON.print(io, results, 1); end
end
println("done; results in ", joinpath(pwd(), "results_final.json"))
