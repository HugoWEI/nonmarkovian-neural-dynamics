# Iterative-solver benchmark: does exact deepest-tier elimination help GMRES too?
#
# HierarchicalEOM.jl's steadystate uses GMRES (rtol 1e-12, atol 1e-14) on the full
# hierarchy. Here every variant uses Krylov.gmres with the SAME settings:
#   full      : GMRES on the full hierarchy + trace constraint (what Julia does)
#   schur     : GMRES on the explicit Schur complement S = A - B D^-1 C (+ constraint)
#   schur_mf  : GMRES on S applied matrix-free (A x - B (D^-1 (C x))), no fill-in
# plus the two exact direct solves for reference (UMFPACK on full, and on S).
#
# Run:  julia --project=. -t auto bench_iter.jl         (default cases)
#       julia --project=. -t auto bench_iter.jl --big   (adds Nk=7, depth 4)

using HierarchicalEOM, SparseArrays, LinearAlgebra, JSON, Printf, Krylov

const HBAR = 0.658211928
const GAMMA, W, U, EPS, MU, RATIO = 0.1, 5.0, 1.0, -0.5, 0.05, 0.3
const REF = JSON.parsefile(joinpath(@__DIR__, "reference.json"))
# cases from the command line, e.g.  bench_iter.jl 5,5 7,5   (default: the small set)
parsecase(s) = Tuple(parse.(Int, split(s, ",")))
const USER = [parsecase(a) for a in ARGS if occursin(r"^\d+,\d+$", a)]
const CASES = !isempty(USER) ? USER :
              ("--big" in ARGS) ? [(3,3), (7,3), (5,4), (7,4)] : [(3,3), (7,3), (5,4)]
const RTOL, ATOL = 1e-12, 1e-14
# GMRES memory: unlimited by default; --restart=K caps it (same K for every variant)
const RESTART = something(tryparse(Int, replace(first(vcat(filter(a -> startswith(a, "--restart="), ARGS), ["--restart=0"])), "--restart=" => "")), 0)
gm(A, b) = RESTART > 0 ? gmres(A, b; rtol = RTOL, atol = ATOL, memory = RESTART, restart = true, itmax = 20_000) :
                         gmres(A, b; rtol = RTOL, atol = ATOL)

d_up = tensor(destroy(2), qeye(2))
d_dn = tensor(sigmaz(), destroy(2))
n_up = d_up' * d_up
n_dn = d_dn' * d_dn
Hsys = (EPS / HBAR) * (n_up + n_dn) + (U / HBAR) * n_up * n_dn
mkbath(op, nk) = Fermion_Lorentz_Pade(op, GAMMA / HBAR, MU / HBAR, W / HBAR, RATIO * GAMMA / HBAR, nk)

function generator(M)
    d = M.data
    for _ in 1:4
        d isa SparseMatrixCSC && return d
        d isa AbstractMatrix && return sparse(d)
        if hasproperty(d, :A); d = getproperty(d, :A)
        elseif hasproperty(d, :data); d = getproperty(d, :data)
        else break end
    end
    error("Could not extract the HEOM generator matrix")
end
tier(n) = hasproperty(n, :level) ? n.level : sum(n.data)

# trace constraint exactly as HierarchicalEOM adds it (row 1 += trace functional)
constraint(dim, n) = sparse(ones(Int, dim), [(k - 1) * (dim + 1) + 1 for k in 1:dim],
                            ones(ComplexF64, dim), n, n)

struct SchurOp
    A::SparseMatrixCSC{ComplexF64,Int}; B::SparseMatrixCSC{ComplexF64,Int}
    C::SparseMatrixCSC{ComplexF64,Int}; dinv::Vector{ComplexF64}
    E::SparseMatrixCSC{ComplexF64,Int}; tmp::Vector{ComplexF64}
end
Base.size(S::SchurOp) = size(S.A)
Base.size(S::SchurOp, i::Int) = size(S.A, i)
Base.eltype(::SchurOp) = ComplexF64
function LinearAlgebra.mul!(y::AbstractVector, S::SchurOp, x::AbstractVector)
    mul!(y, S.A, x)
    mul!(S.tmp, S.C, x)
    S.tmp .*= S.dinv
    mul!(y, S.B, S.tmp, -1, 1)
    mul!(y, S.E, x, 1, 1)
    return y
end

nup_of(y) = real(tr(Matrix(n_up.data) * reshape(y[1:16], 4, 4)))

function run_case(nk, depth)
    M = M_Fermion(Hsys, depth, [mkbath(d_up, nk), mkbath(d_dn, nk)]; verbose = false)
    L = generator(M); n = size(L, 1); blk = 16
    tiers = [tier(v) for v in M.hierarchy.idx2nvec]; top = maximum(tiers)
    expand(ix) = vec([(i - 1) * blk + j for j in 1:blk, i in ix])
    keep = expand(findall(<(top), tiers)); last = expand(findall(==(top), tiers))
    ref = haskey(REF, "$nk,$depth") ? REF["$nk,$depth"]["n_up"] : NaN   # NaN: no reference yet
    rec = Dict{String,Any}("Nk" => nk, "depth" => depth, "unknowns_full" => n,
                           "unknowns_schur" => length(keep), "nnz_full" => nnz(L))

    full = L + constraint(4, n); b = zeros(ComplexF64, n); b[1] = 1
    t = @elapsed (y, st) = gm(full, b)
    rec["gmres_full_s"] = round(t, digits = 3); rec["gmres_full_iters"] = st.niter
    rec["n_up_gmres_full"] = nup_of(y); rec["gmres_full_err"] = abs(nup_of(y) - ref)
    rec["gmres_full_solved"] = st.solved

    A = L[keep, keep]; B = L[keep, last]; C = L[last, keep]
    dinv = 1 ./ diag(L[last, last])
    E = constraint(4, length(keep)); bs = zeros(ComplexF64, length(keep)); bs[1] = 1
    expand_y(x) = (yy = zeros(ComplexF64, n); yy[keep] = x; yy[last] = -(dinv .* (C * x)); yy)
    big = n > 2_000_000   # very large: skip building S explicitly (saves memory)
    if !big
        t = @elapsed S = A - B * Diagonal(dinv) * C
        rec["schur_build_s"] = round(t, digits = 3); rec["nnz_schur"] = nnz(S)

        Sc = S + E
        t = @elapsed (x, st) = gm(Sc, bs)
        rec["gmres_schur_s"] = round(t + rec["schur_build_s"], digits = 3); rec["gmres_schur_iters"] = st.niter
        rec["gmres_schur_err"] = abs(nup_of(expand_y(x)) - ref)
    else
        rec["gmres_schur_s"] = "skipped (matrix-free only at this size)"
    end

    op = SchurOp(A, B, C, dinv, E, zeros(ComplexF64, length(last)))
    t = @elapsed (x, st) = gm(op, bs)
    rec["gmres_schur_mf_s"] = round(t, digits = 3); rec["gmres_schur_mf_iters"] = st.niter
    rec["n_up_gmres_schur_mf"] = nup_of(expand_y(x)); rec["gmres_schur_mf_err"] = abs(nup_of(expand_y(x)) - ref)
    rec["gmres_schur_mf_solved"] = st.solved
    rec["peak_memory_gb"] = round(Sys.maxrss() / 2^30, digits = 2)

    if n <= 300_000   # the direct solve of the full hierarchy gets very slow beyond this
        t = @elapsed y = full \ b
        rec["direct_full_s"] = round(t, digits = 3); rec["direct_full_err"] = abs(nup_of(y) - ref)
    else
        rec["direct_full_s"] = "skipped (too large)"
    end
    if !big && length(keep) <= 150_000
        t = @elapsed x = Sc \ bs
        rec["direct_schur_s"] = round(t + rec["schur_build_s"], digits = 3)
        rec["n_up_direct_schur"] = nup_of(expand_y(x))
        rec["direct_schur_err"] = abs(rec["n_up_direct_schur"] - ref)
    else
        rec["direct_schur_s"] = "skipped (too large)"
    end
    rec
end

run_case(3, 3)   # warm-up: compiles everything once
results = Any[]
for (nk, depth) in CASES
    r = run_case(nk, depth)
    for (k, v) in r   # no reference for new sizes: JSON can't hold NaN
        v isa AbstractFloat && isnan(v) && (r[k] = "n/a (no reference)")
    end
    haskey(r, "n_up_gmres_full") && (r["full_vs_elimination_diff"] = abs(r["n_up_gmres_full"] - r["n_up_gmres_schur_mf"]))
    push!(results, r)
    println(JSON.json(r)); flush(stdout)
    open(joinpath(@__DIR__, "results_iter.json"), "w") do io
        JSON.print(io, results, 1)
    end
end
println("done; results in results_iter.json")
