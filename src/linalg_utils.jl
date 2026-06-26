using LinearAlgebra
using Random

"""
    tsvd(A; cutoff=1e-15, maxdim=typemax(Int)) -> (U, S, V)

Truncated thin SVD with `A ≈ U * Diagonal(S) * V'`. Falls back to the QR-iteration
algorithm if the divide-and-conquer driver fails (cf. npsvd.py gesvd fallback).
"""
function tsvd(A::AbstractMatrix; cutoff::Real=1e-15, maxdim::Int=typemax(Int))
    F = try
        svd(A)
    catch e
        e isa LinearAlgebra.LAPACKException || rethrow()
        svd(A; alg=LinearAlgebra.QRIteration())
    end
    S = F.S
    nkeep = count(>(cutoff), S)
    nkeep = nkeep == 0 ? 1 : min(nkeep, maxdim, length(S))
    return F.U[:, 1:nkeep], S[1:nkeep], F.V[:, 1:nkeep]
end

"""
    rsvd(A, k, oversample=10, power=10; rng) -> (U, S, V)

Randomized SVD (port of npsvd.py:rsvd) with `A ≈ U * Diagonal(S) * V'`.
"""
function rsvd(A::AbstractMatrix{T}, k::Int, oversample::Int=10, power::Int=10;
             rng::AbstractRNG=Random.default_rng()) where {T}
    n = size(A, 2)
    p = min(n, oversample * k)
    Y = A * randn(rng, T, n, p)
    for _ in 1:power
        Y = Matrix(qr(A * (A' * Y)).Q)
    end
    Q = Matrix(qr(Y).Q)
    B = Q' * A
    F = svd(B)
    kk = min(k, size(F.U, 2))
    return (Q * F.U)[:, 1:kk], F.S[1:kk], F.V[:, 1:kk]
end
