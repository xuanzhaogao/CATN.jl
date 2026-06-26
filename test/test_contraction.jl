using CATN: TensorNetwork, MPSNode, mps2raw, order
using CATN: dim_after_merge, select_edge_init!, select_edge_min_dim
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

using CATN: dim_after_merge, select_edge_init!, select_edge_min_dim

@testset "edge selection" begin
    A = randn(2,2); B = randn(2,2); C = randn(2,8); D = randn(8,2)
    # chain 1-2-3-4 with a fat bond between 3 and 4
    ixs = [[:a,:x],[:x,:y],[:y,:z],[:z,:a]]   # a loop 1-2-3-4-1
    tn = TensorNetwork([A,B,C,D], ixs; chi=1000)
    select_edge_init!(tn)
    i, j = select_edge_min_dim(tn)
    @test Set([i,j]) != Set([3,4])             # the fat bond is the costliest, not chosen first
end
