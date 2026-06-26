using CATN: MPSNode, mps2raw, raw2mps, order, shape
using LinearAlgebra, Test

@testset "raw2mps/mps2raw round-trip" begin
    for dims in [(4,), (3, 5), (2, 3, 4), (2, 3, 2, 3)]
        T = randn(dims...)
        nb = collect(1:length(dims))
        node = MPSNode(T, nb; chi=1000, cutoff=1e-15)
        @test order(node) == length(dims)
        @test shape(node) == [dims...] || (length(dims) == 1 && shape(node) == [dims[1]])
        @test mps2raw(node) ≈ T
    end

    # chi truncation on a genuinely low-rank tensor is (near) lossless
    U = randn(6, 2); V = randn(2, 6)
    M = U * V                      # rank 2 matrix, viewed as order-2 tensor
    node = MPSNode(M, [1, 2]; chi=2, cutoff=1e-15)
    @test mps2raw(node) ≈ M atol=1e-10
end
