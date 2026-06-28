using CATN
using CATN: MPSNode, TensorNetwork, contraction!, eat!, find_neighbor, mps2raw
using LinearAlgebra, Test
# exact_contract is available from exact.jl (included earlier in runtests.jl)

@testset "complex" begin
    @testset "eat! both-leaves uses non-conjugating product" begin
        u = randn(ComplexF64, 4); v = randn(ComplexF64, 4)
        ni = MPSNode(u, [2]; norm_method=0)
        nj = MPSNode(v, [1]; norm_method=0)
        lognorm, err, phase = eat!(ni, nj, 1, 1)
        @test isempty(ni.mps)
        @test exp(lognorm) * phase ≈ sum(u .* v)        # NON-conjugating; dot(u,v) would conjugate u
        @test !(exp(lognorm) * phase ≈ dot(u, v))        # guard: must differ from the conjugating dot
    end

    @testset "complex generic contraction matches oracle (exact mode)" begin
        networks = [
            ([randn(ComplexF64,3,4), randn(ComplexF64,4,5), randn(ComplexF64,5,3)],
             [[:a,:b],[:b,:c],[:c,:a]]),                                   # chain/loop
            ([randn(ComplexF64,2,3,4), randn(ComplexF64,2), randn(ComplexF64,3), randn(ComplexF64,4)],
             [[:a,:b,:c],[:a],[:b],[:c]]),                                 # star/tree
            ([randn(ComplexF64,2,3),randn(ComplexF64,3,2),randn(ComplexF64,2,3),randn(ComplexF64,3,2)],
             [[:a,:b],[:b,:c],[:c,:d],[:d,:a]]),                           # loop of 4
        ]
        for (ts, ixs) in networks
            ref = exact_contract(ts, ixs)[]
            for sel in 0:2
                tn = TensorNetwork(ts, ixs; Dmax=-1, chi=10_000, select=sel,
                                   reverse=true, compress=true, norm_method=1)
                lnZ, err, psi = contraction!(tn)
                @test exp(lnZ) * psi ≈ ref rtol=1e-10
            end
        end
    end
end
