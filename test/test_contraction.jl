using CATN: TensorNetwork, MPSNode, mps2raw, order
using CATN: dim_after_merge, select_edge_init!, select_edge_min_dim, select_edge_sequentially
using CATN: cut_bondim!, cut_bondim_opt!
using CATN: contraction!
using Test
# exact_contract is available from exact.jl (included earlier in runtests.jl)

@testset "TensorNetwork construction" begin
    A = randn(2,3); B = randn(3,4); C = randn(4,2)
    ixs = [[:a,:b],[:b,:c],[:c,:a]]
    tn = TensorNetwork([A,B,C], ixs; chi=1000)
    @test length(tn.tensors) == 3
    @test mps2raw(tn.tensors[1]) ≈ A
    @test mps2raw(tn.tensors[2]) ≈ B
    # node 1 neighbors: leg :a -> node 3, leg :b -> node 2
    @test tn.tensors[1].neighbor == [3, 2]
    @test tn.num_isolated == 0
end

@testset "edge selection" begin
    A = randn(2,2); B = randn(2,2); C = randn(2,8); D = randn(8,2)
    # chain 1-2-3-4 with a fat bond between 3 and 4
    ixs = [[:a,:x],[:x,:y],[:y,:z],[:z,:a]]   # a loop 1-2-3-4-1
    tn = TensorNetwork([A,B,C,D], ixs; chi=1000)
    select_edge_init!(tn)
    i, j = select_edge_min_dim(tn)
    @test Set([i,j]) != Set([3,4])             # the fat bond is the costliest, not chosen first
end

@testset "select_edge_sequentially" begin
    # Triangle network: nodes 1-2, 2-3, 3-1.  All bonds have equal bond dim (2).
    # The edge with smallest i+j is (1,2) with sum=3 < (1,3)=4 < (2,3)=5.
    # The rewritten selector scans nodes directly, so select_edge_init! is NOT required.
    A = randn(2,2); B = randn(2,2); C = randn(2,2)
    ixs = [[:a,:b], [:b,:c], [:c,:a]]
    tn = TensorNetwork([A,B,C], ixs; chi=1000)
    # No select_edge_init! call — verify the selector works without edge_count
    result = select_edge_sequentially(tn)
    @test result == (1, 2)
end

@testset "cut_bondim is exact when bond is low-rank" begin
    # bond between two order-2 nodes is rank 2 but stored as dim 4.
    # ixs = [[:a,:m],[:m,:b]]: leg 2 of node 1 (label :m) is the shared bond (dim 4).
    # P = randn(3,4) rank-2, Qm = randn(4,3) rank-2; shared bond :m has dim 4, rank 2.
    P = randn(3,2)*randn(2,4); Qm = randn(4,2)*randn(2,3)
    ixs = [[:a,:m],[:m,:b]]
    for cutter! in (cut_bondim!, cut_bondim_opt!)
        tn = TensorNetwork([P,Qm], ixs; chi=1000, Dmax=2)
        # idx_j_in_i=2: leg 2 of node 1 is :m, pointing to node 2
        cutter!(tn, 1, 2)
        @test size(tn.tensors[1].mps[2], 2) == 2   # physical bond truncated to 2
        # network contraction value preserved
        got = ein"am,mb->ab"(mps2raw(tn.tensors[1]), mps2raw(tn.tensors[2]))
        @test got ≈ P*Qm atol=1e-8
    end
end

# ---------------------------------------------------------------------------
# contraction! tests
# ---------------------------------------------------------------------------

function catn_value(tensors, ixs; kwargs...)
    tn = TensorNetwork(tensors, ixs; kwargs...)
    lnZ, err, psi = contraction!(tn)
    return exp(lnZ) * psi
end

@testset "contraction! exact mode matches oracle" begin
    networks = [
        # chain: 3-node loop
        ([randn(3,4), randn(4,5), randn(5,3)], [[:a,:b],[:b,:c],[:c,:a]]),
        # star/tree: 4-node star with 3 leaves
        ([randn(2,3,4), randn(2), randn(3), randn(4)],
         [[:a,:b,:c],[:a],[:b],[:c]]),
        # single loop of 4 nodes
        ([randn(2,3),randn(3,2),randn(2,3),randn(3,2)],
         [[:a,:b],[:b,:c],[:c,:d],[:d,:a]]),
    ]
    for (ts, ixs) in networks
        ref = exact_contract(ts, ixs)[]
        for sel in 0:2, rev in (false,true), comp in (false,true)
            got = catn_value(ts, ixs; Dmax=-1, chi=10_000,
                             select=sel, reverse=rev, compress=comp, norm_method=1)
            @test got ≈ ref rtol=1e-8
        end
    end
end

@testset "contraction! finite Dmax is close on a contractible loop" begin
    ts, ixs = ([randn(4,4),randn(4,4),randn(4,4),randn(4,4)],
               [[:a,:b],[:b,:c],[:c,:d],[:d,:a]])
    ref = exact_contract(ts, ixs)[]
    got = catn_value(ts, ixs; Dmax=4, chi=64, select=1, compress=true)
    @test got ≈ ref rtol=1e-6
end
