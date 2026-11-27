"""
    HEOMTierElimination

Exact deepest-tier elimination for stationary states of HierarchicalEOM.jl models.

ADOs whose total hierarchy level is maximal couple only to the level below,
never to each other. At stationarity they can therefore be eliminated exactly
(a Schur complement), leaving a smaller system with the same solution.
"""
module HEOMTierElimination

using HierarchicalEOM
using LinearAlgebra
using SparseArrays
using Krylov

export steadystate_eliminated

# ---------------------------------------------------------------------------
# Access to the HEOM Liouvillian and its hierarchy
# ---------------------------------------------------------------------------

function _generator(M)
    d = M.data
    for _ in 1:4
        d isa SparseMatrixCSC && return d
        d isa AbstractMatrix && return sparse(d)
        if hasproperty(d, :A)
            d = getproperty(d, :A)
        elseif hasproperty(d, :data)
            d = getproperty(d, :data)
        else
            break
        end
    end
    error("steadystate_eliminated needs an assembled sparse HEOMLS matrix " *
          "(lazy operators are not supported); got $(typeof(M.data)).")
end

_level(n) = hasproperty(n, :level) ? n.level : sum(n.data)
_level(n::Tuple) = sum(_level, n)          # mixed boson-fermion hierarchies

# indices of ADOs below the maximal total level and at the maximal level
function _split_ados(M)
    levels = [_level(n) for n in M.hierarchy.idx2nvec]
    top = maximum(levels)
    top == 0 && error("the hierarchy has no auxiliary tier to eliminate")
    return findall(<(top), levels), findall(==(top), levels)
end

_expand(ados, blk) = vec([(i - 1) * blk + j for j in 1:blk, i in ados])

# inverse of a block-diagonal matrix with square blocks of size blk
function _block_inverse(D::SparseMatrixCSC, blk::Int)
    n = size(D, 1)
    if isdiag(D)
        dg = diag(D)
        any(iszero, dg) && throw(SingularException(findfirst(iszero, dg)))
        return spdiagm(0 => 1 ./ dg)
    end
    I, J, V = Int[], Int[], ComplexF64[]
    for b in 0:(n ÷ blk - 1)
        r = (b * blk + 1):((b + 1) * blk)
        Binv = inv(Matrix(D[r, r]))
        for j in 1:blk, i in 1:blk
            push!(I, r[i]); push!(J, r[j]); push!(V, Binv[i, j])
        end
    end
    return sparse(I, J, V, n, n)
end

# trace condition Tr(rho) = 1, added to the first row (as HierarchicalEOM does)
_constraint(d, n) = sparse(ones(Int, d), [(k - 1) * (d + 1) + 1 for k in 1:d],
                           ones(ComplexF64, d), n, n)

# reduced operator S x = A x - B (D^-1 (C x)) + E x, applied matrix-free
struct ReducedOperator
    A::SparseMatrixCSC{ComplexF64, Int}
    B::SparseMatrixCSC{ComplexF64, Int}
    C::SparseMatrixCSC{ComplexF64, Int}
    Dinv::SparseMatrixCSC{ComplexF64, Int}
    E::SparseMatrixCSC{ComplexF64, Int}
    t1::Vector{ComplexF64}
    t2::Vector{ComplexF64}
end
Base.size(S::ReducedOperator) = size(S.A)
Base.size(S::ReducedOperator, i::Int) = size(S.A, i)
Base.eltype(::ReducedOperator) = ComplexF64
function LinearAlgebra.mul!(y::AbstractVector, S::ReducedOperator, x::AbstractVector)
    mul!(y, S.A, x)
    mul!(S.t1, S.C, x)
    mul!(S.t2, S.Dinv, S.t1)
    mul!(y, S.B, S.t2, -1, 1)
    mul!(y, S.E, x, 1, 1)
    return y
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    steadystate_eliminated(M; method = :gmres, rtol = 1e-12, atol = 1e-14,
                           restart = 0, verbose = false) -> ADOs

Stationary state of the HEOM Liouvillian `M` (from `M_Boson`, `M_Fermion` or
`M_Boson_Fermion`), computed after exact elimination of the deepest tier.
Drop-in alternative to `steadystate(M)`; returns an `ADOs` object.

- `method = :gmres` (default): GMRES on the reduced operator, applied matrix-free.
  Same default tolerances as `HierarchicalEOM.steadystate`.
- `method = :direct`: sparse LU of the reduced matrix; exact to machine precision,
  best for small and medium hierarchies.
- `restart`: GMRES memory; `0` means no restart. Use e.g. `100` for very large
  hierarchies to bound memory.
"""
function steadystate_eliminated(M::HierarchicalEOM.AbstractHEOMLSMatrix; method::Symbol = :gmres,
                                rtol::Real = 1e-12, atol::Real = 1e-14,
                                restart::Int = 0, verbose::Bool = false)
    typeof(M.parity) == typeof(EVEN) || error("The parity of M should be EVEN.")
    L = _generator(M)
    n = size(L, 1)
    blk = n ÷ M.N
    d = isqrt(blk)
    d * d == blk || error("unexpected ADO block size $blk")

    low, top = _split_ados(M)
    keep, last = _expand(low, blk), _expand(top, blk)
    A = L[keep, keep]; B = L[keep, last]; C = L[last, keep]; Dm = L[last, last]
    Dinv = _block_inverse(Dm, blk)
    E = _constraint(d, length(keep))
    b = zeros(ComplexF64, length(keep)); b[1] = 1
    verbose && println("eliminated $(length(top)) of $(M.N) ADOs; " *
                       "solving $(length(keep)) instead of $n unknowns")

    if method === :direct
        x = (A - B * Dinv * C + E) \ b
    elseif method === :gmres
        S = ReducedOperator(A, B, C, Dinv, E, zeros(ComplexF64, length(last)),
                            zeros(ComplexF64, length(last)))
        x, st = restart > 0 ?
            gmres(S, b; rtol = rtol, atol = atol, memory = restart, restart = true, itmax = 50_000) :
            gmres(S, b; rtol = rtol, atol = atol)
        st.solved || @warn "GMRES did not reach the requested tolerance" st
        verbose && println("GMRES iterations: $(st.niter)")
    else
        throw(ArgumentError("method must be :gmres or :direct"))
    end

    y = zeros(ComplexF64, n)
    y[keep] = x
    y[last] = -(Dinv * (C * x))
    # keep the system's tensor structure (e.g. two spins = [2, 2], not [4])
    return hasproperty(M, :dimensions) ? ADOs(y, M.dimensions.to, M.N, M.parity) :
                                         ADOs(y, M.N, M.parity)
end

end # module
