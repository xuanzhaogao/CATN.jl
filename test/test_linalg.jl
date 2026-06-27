using CATN: tsvd, rsvd
using LinearAlgebra, Random, Test

@testset "linalg_utils" begin
    A = randn(8, 5)
    U, S, V = tsvd(A)
    @test U * Diagonal(S) * V' ≈ A
    @test issorted(S, rev=true)

    # maxdim truncation keeps the leading subspace
    U2, S2, V2 = tsvd(A; maxdim=2)
    @test length(S2) == 2
    @test S2 ≈ S[1:2]

    # cutoff drops tiny singular values
    B = U[:, 1:3] * Diagonal([1.0, 1e-3, 1e-20]) * V[:, 1:3]'
    _, S3, _ = tsvd(B; cutoff=1e-10)
    @test length(S3) == 2

    # rsvd approximates the leading singular triple of a low-rank matrix
    Random.seed!(1)
    L = randn(50, 4) * randn(4, 40)      # rank 4
    Ur, Sr, Vr = rsvd(L, 4, 10, 10; rng=MersenneTwister(0))
    @test Ur * Diagonal(Sr) * Vr' ≈ L atol=1e-8

    @testset "rsvd uses device-following random sketch" begin
        Random.seed!(2)
        L = randn(40, 3) * randn(3, 30)          # rank 3
        U, S, V = rsvd(L, 3, 10, 10; rng=MersenneTwister(7))
        @test U * Diagonal(S) * V' ≈ L atol=1e-8
        # CPU determinism preserved across calls with the same seed
        U2, S2, V2 = rsvd(L, 3, 10, 10; rng=MersenneTwister(7))
        @test S ≈ S2
    end
end
