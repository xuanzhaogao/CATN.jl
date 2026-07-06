using CATN
using CATN: TensorNetwork, contraction!, result_tensor
using OMEinsum, LinearAlgebra, Test
# exact_contract available from exact.jl (included earlier in runtests.jl)

@testset "open-leg (partial) contraction" begin
    # helper: full result of an open contraction
    function open_value(ts, ixs; kwargs...)
        tn = TensorNetwork(ts, ixs; kwargs...)
        lnZ, err, psi = contraction!(tn)
        return result_tensor(tn) .* (exp(lnZ) * psi)
    end

    @testset "chain with open physical legs (exact mode) vs oracle" begin
        # T1(p1,a) T2(a,p2,b) T3(b,p3): virtual bonds a,b contracted; p1,p2,p3 open
        for T in (Float64, ComplexF64)
            T1 = randn(T, 2, 4); T2 = randn(T, 4, 2, 4); T3 = randn(T, 4, 2)
            ts = [T1, T2, T3]; ixs = [[:p1,:a],[:a,:p2,:b],[:b,:p3]]
            ref = exact_contract(ts, ixs)                    # tensor over p1,p2,p3
            got = open_value(ts, ixs; Dmax=-1, chi=10_000)
            # compare up to leg permutation: same size-multiset and sorted values
            @test sort(collect(size(got))) == sort(collect(size(ref)))
            @test vec(sort(vec(abs.(got)))) ≈ vec(sort(vec(abs.(ref)))) rtol=1e-10
        end
    end

    @testset "closed network unchanged (result_tensor is scalar one)" begin
        A = randn(2,3); B = randn(3,4); C = randn(4,2)
        ixs = [[:a,:b],[:b,:c],[:c,:a]]
        ref = exact_contract([A,B,C], ixs)[]
        tn = TensorNetwork([A,B,C], ixs; Dmax=-1, chi=10_000)
        lnZ, err, psi = contraction!(tn)
        @test exp(lnZ)*psi ≈ ref rtol=1e-10             # closed API unchanged
        @test ndims(result_tensor(tn)) == 0             # 0-d one() for closed
        @test result_tensor(tn)[] * exp(lnZ) * psi ≈ ref rtol=1e-10
    end
end
