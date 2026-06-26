using CATN: TensorNetwork, MPSNode, mps2raw, order
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
