using CATN
using CATN: MPSNode, TensorNetwork, contraction!, eat!, find_neighbor, mps2raw, cut_bondim!, cut_bondim_opt!
using LinearAlgebra, Test
using OMEinsum
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

    @testset "cut_bondim preserves complex low-rank bond" begin
        # two order-2 complex nodes whose shared bond is rank 2, stored as dim 4
        P = randn(ComplexF64,3,2) * randn(ComplexF64,2,4)   # 3x4, rank 2 on the :m leg
        Qm = randn(ComplexF64,4,2) * randn(ComplexF64,2,3)  # 4x3, rank 2
        ixs = [[:a,:m],[:m,:b]]
        for cutter! in (cut_bondim!, cut_bondim_opt!)
            tn = TensorNetwork([P,Qm], ixs; chi=1000, Dmax=2)
            cutter!(tn, 1, 2)                                  # :m is leg 2 of node 1
            @test size(tn.tensors[1].mps[2], 2) == 2          # truncated to Dmax=2
            got = ein"am,mb->ab"(mps2raw(tn.tensors[1]), mps2raw(tn.tensors[2]))
            @test got ≈ P * Qm atol=1e-8                      # value preserved (lossless: bond was rank 2)
        end
    end

    @testset "complex finite-Dmax contraction matches oracle" begin
        # loop of 4 with genuinely low-rank complex bonds (rank 2), Dmax=2 lossless
        mats = [randn(ComplexF64,4,2)*randn(ComplexF64,2,4) for _ in 1:4]   # each 4x4, rank 2
        ts = [reshape(mats[1], 4,4), reshape(mats[2],4,4), reshape(mats[3],4,4), reshape(mats[4],4,4)]
        ixs = [[:a,:b],[:b,:c],[:c,:d],[:d,:a]]
        ref = exact_contract(ts, ixs)[]
        for opt in (true, false)
            tn = TensorNetwork(ts, ixs; Dmax=2, chi=64, select=1, compress=true, svdopt=opt)
            lnZ, err, psi = contraction!(tn)
            @test exp(lnZ) * psi ≈ ref rtol=1e-6
        end
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
