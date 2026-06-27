using LinearAlgebra
using Random

"""
    tsvd(A; cutoff=1e-15, maxdim=typemax(Int)) -> (U, S, V, discarded)

Truncated thin SVD with `A ≈ U * Diagonal(S) * V'`. Falls back to the QR-iteration
algorithm if the divide-and-conquer driver fails (cf. npsvd.py gesvd fallback).

The 4th return value `discarded` is the sum of all singular values beyond the kept
set (i.e. those dropped by the `cutoff` or `maxdim` truncation). Existing callers
using `U, S, V = tsvd(...)` are unaffected — Julia tuple-destructuring ignores
extra elements.
"""
function tsvd(A::AbstractMatrix; cutoff::Real=1e-15, maxdim::Int=typemax(Int))
    F = try
        svd(A)
    catch e
        e isa LinearAlgebra.LAPACKException || rethrow()
        svd(A; alg=LinearAlgebra.QRIteration())
    end
    S = F.S
    nfull = length(S)
    nkeep = count(>(cutoff), S)
    nkeep = nkeep == 0 ? 1 : min(nkeep, maxdim, nfull)
    discarded = nkeep < nfull ? sum(@view S[nkeep+1:nfull]) : zero(eltype(S))
    return F.U[:, 1:nkeep], S[1:nkeep], F.V[:, 1:nkeep], discarded
end

# Thin Q materialization: returns Q of shape (m, n) where n = number of columns in R.
# Device-agnostic replacement for Matrix(F.Q) when only the thin Q is needed.
function _thin_q(Q, ref::AbstractArray, ::Type{T}, n::Int) where {T}
    E = similar(ref, T, (n, n))
    fill!(E, zero(T))
    @views E[diagind(E)] .= one(T)
    return Q * E
end

"""
    rsvd(A, k, oversample=10, power=10; rng) -> (U, S, V)

Randomized SVD (port of npsvd.py:rsvd) with `A ≈ U * Diagonal(S) * V'`.
"""
function rsvd(A::AbstractMatrix{T}, k::Int, oversample::Int=10, power::Int=10;
             rng::AbstractRNG=Random.default_rng()) where {T}
    n = size(A, 2)
    p = min(n, oversample * k)
    Y = A * randn!(rng, similar(A, T, (n, p)))
    for _ in 1:power
        M = A * (A' * Y)
        Y = _thin_q(qr(M).Q, A, T, size(M, 2))
    end
    Q = _thin_q(qr(Y).Q, A, T, size(Y, 2))
    B = Q' * A
    F = svd(B)
    kk = min(k, size(F.U, 2))
    return (Q * F.U)[:, 1:kk], F.S[1:kk], F.V[:, 1:kk]
end
