using CATN: MPSNode, mps2raw, raw2mps, order, shape, cano_to!, left_canonical!
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

@testset "canonicalization preserves tensor" begin
    T = randn(2, 3, 4, 2)
    node = MPSNode(T, [1,2,3,4]; chi=1000)
    ref = mps2raw(node)
    for idx in [1, 2, 3, 4, 0]
        cano_to!(node, idx)
        @test mps2raw(node) ≈ ref
    end
    left_canonical!(node)
    @test node.cano == 4
    @test mps2raw(node) ≈ ref
    # left-isometry of all but the center after left_canonical!
    for i in 1:3
        A = node.mps[i]; dl, d, dr = size(A)
        M = reshape(A, dl*d, dr)
        @test M' * M ≈ I atol=1e-8
    end
end

@testset "cano_to! preserves rank for tiny-norm tensor" begin
    T = randn(2,3,4,2) .* 1e-16
    node = MPSNode(T, [1,2,3,4]; chi=1000, cutoff=0.0)  # build at full rank
    node.cutoff = 1e-15                                   # buggy cano_to! would now collapse
    ref = mps2raw(node)
    cano_to!(node, 1)                                     # full left sweep
    @test mps2raw(node) ≈ ref rtol=1e-8
    cano_to!(node, 4)                                     # full right sweep
    @test mps2raw(node) ≈ ref rtol=1e-8
end
