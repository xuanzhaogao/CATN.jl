using CATN
using CATN: MPSNode, TensorNetwork, mps2raw, contraction!, ising_network
using Adapt
using Test

@testset "adapt to Array round-trips on CPU" begin
    # generic network
    A = randn(2,3); B = randn(3,4); C = randn(4,2)
    tn = TensorNetwork([A,B,C], [[:a,:b],[:b,:c],[:c,:a]]; chi=1000)
    tn2 = adapt(Array, tn)
    @test tn2 isa TensorNetwork{Float64, Array{Float64,3}}
    @test mps2raw(tn2.tensors[1]) ≈ mps2raw(tn.tensors[1])
    @test tn2.n == tn.n && tn2.beta == tn.beta && tn2.Dmax == tn.Dmax
    @test tn2.tensors !== tn.tensors                  # adapt_structure rebuilt the Dict
    @test tn2.rng !== tn.rng                           # rng was deepcopied, not shared

    # adapting a node preserves the represented tensor
    node = MPSNode(randn(2,3,4), [1,2,3]; chi=1000)
    nb = adapt(Array, node)
    @test mps2raw(nb) ≈ mps2raw(node)

    # adapt(Array, ...) does not change the contraction result
    g1 = ising_network(3, [(1,2),(2,3),(3,1)], ones(3), zeros(3), 0.4; Dmax=-1, chi=10_000)
    g2 = ising_network(3, [(1,2),(2,3),(3,1)], ones(3), zeros(3), 0.4; Dmax=-1, chi=10_000)
    lnZ1, = contraction!(g1)
    lnZ2, = contraction!(adapt(Array, g2))
    @test lnZ1 ≈ lnZ2
end
