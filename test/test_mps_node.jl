using CATN: MPSNode, mps2raw, raw2mps, order, shape, cano_to!, left_canonical!, merge!, compress!, compress_opt!
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

using CATN: swap!, reverse!, move2tail!, move2head!, move!, find_neighbor,
            add_neighbor!, delete_neighbor!, logdim, lognorm, clear!

@testset "swap/reverse preserve tensor (up to leg permutation)" begin
    T = randn(2, 3, 4, 5)
    node = MPSNode(T, [10,20,30,40]; chi=1000)
    swap!(node, 2, 3)                       # swap legs 2 and 3
    @test node.neighbor == [10,30,20,40]
    @test mps2raw(node) ≈ permutedims(T, (1,3,2,4))

    node2 = MPSNode(T, [10,20,30,40]; chi=1000)
    reverse!(node2)
    @test node2.neighbor == [40,30,20,10]
    @test mps2raw(node2) ≈ permutedims(T, (4,3,2,1))

    node3 = MPSNode(T, [10,20,30,40]; chi=1000)
    move2tail!(node3, 1)                    # move first leg to the end
    @test node3.neighbor[end] == 10
    @test mps2raw(node3) ≈ permutedims(T, (2,3,4,1))
end

@testset "neighbor helpers" begin
    node = MPSNode(randn(2,2,2), [5,6,7])
    @test find_neighbor(node, 6) == 2
    @test find_neighbor(node, 99) == 0
    delete_neighbor!(node, 6)
    @test node.neighbor == [5,7]
end

@testset "swap! reports truncation error" begin
    T = randn(4,4,4)                 # swapping middle bonds can need bond dim up to 16
    node = MPSNode(T, [1,2,3]; chi=2, cutoff=1e-15)  # chi=2 forces truncation
    err = swap!(node, 1, 2)
    @test err > 0
end

@testset "swap! going left (i>j)" begin
    T = randn(2,3,4,5)
    node = MPSNode(T, [10,20,30,40]; chi=1000)
    swap!(node, 3, 2)                # swap sites 3 and 2, going left
    @test node.neighbor == [10,30,20,40]
    @test mps2raw(node) ≈ permutedims(T, (1,3,2,4))
end

@testset "compress preserves tensor" begin
    T = randn(2,3,4,3,2)
    for f! in (compress!, compress_opt!)
        node = MPSNode(T, collect(1:5); chi=1000)
        ref = mps2raw(node)
        err = f!(node)
        @test node.cano == 1
        @test mps2raw(node) ≈ ref
        @test err isa Float64
        @test err >= 0.0
    end
    # compress removes inflated bonds on a low-rank chain
    A = randn(3,2); M = kron(A, A')         # contrived low-rank-ish
    node = MPSNode(reshape(M, 3,2,2,3), collect(1:4); chi=1000)
    compress!(node)
    @test mps2raw(node) ≈ reshape(M, 3,2,2,3)
end

@testset "merge! fuses duplicate-neighbor legs" begin
    # order-3 tensor, legs 1 and 3 both point to neighbor 7 (a duplicate)
    T = randn(2, 5, 3)
    node = MPSNode(T, [7, 8, 7]; chi=1000)
    merge!(node, 7)                 # fuse legs to neighbor 7
    @test count(==(7), node.neighbor) == 1
    # remaining represented tensor: legs (7-fused, 8).
    # cross=false: move leg at idx2=3 to idx1+1=2 (i.e., swap legs 2 and 3),
    # fuse legs 1 and 2 of the reordered tensor (original legs 1 and 3).
    # mps2raw on the fused node has shape (2*3, 5); combined index runs
    # original leg1 fast (since it stays at idx1=1), original leg3 slow.
    raw = mps2raw(node)
    # After swap (move leg3 to pos2): tensor becomes permutedims(T,(1,3,2)) of shape (2,3,5)
    # Fusing dims 1 and 2 in column-major order: reshape to (2*3, 5)
    expected = reshape(permutedims(T, (1, 3, 2)), 2*3, 5)
    @test reshape(raw, size(raw, 1), size(raw, 2)) ≈ expected
end
