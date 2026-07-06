using CATN
using CATN: TensorNetwork, mps2raw, contraction!, ising_network, tsvd, rsvd
using Adapt, LinearAlgebra, Random, Test
using CUDA

if !CUDA.functional()
    @info "CUDA not functional — skipping GPU tests"
else
    using cuTENSOR     # enables OMEinsum's GPU einsum fast path
    # exact_contract is available from exact.jl (included earlier in runtests.jl)

    @testset "GPU" begin
        @testset "adapt to CuArray round-trips" begin
            A = randn(2,3); B = randn(3,4); C = randn(4,2)
            tn = TensorNetwork([A,B,C], [[:a,:b],[:b,:c],[:c,:a]]; chi=1000)
            gtn = adapt(CuArray, tn)
            @test gtn.tensors[1].mps[1] isa CuArray
            back = adapt(Array, gtn)
            @test mps2raw(back.tensors[1]) ≈ mps2raw(tn.tensors[1])
        end

        @testset "tsvd/rsvd on CuArray" begin
            Acpu = randn(8,5); A = CuArray(Acpu)
            U, S, V = tsvd(A)
            @test Array(U * Diagonal(S) * V') ≈ Acpu atol=1e-8
            Lcpu = randn(40,3) * randn(3,30); L = CuArray(Lcpu)
            Ur, Sr, Vr = rsvd(L, 3, 10, 10; rng=Random.default_rng())
            @test Array(Ur * Diagonal(Sr) * Vr') ≈ Lcpu atol=1e-5
        end

        @testset "GPU generic contraction matches oracle" begin
            ts = [randn(3,4), randn(4,5), randn(5,3)]
            ixs = [[:a,:b],[:b,:c],[:c,:a]]
            ref = exact_contract(ts, ixs)[]
            gtn = TensorNetwork([CuArray(t) for t in ts], ixs; Dmax=-1, chi=10_000)
            lnZ, err, psi = contraction!(gtn)
            @test exp(lnZ) * psi ≈ ref rtol=1e-8
        end

        @testset "GPU Ising matches CPU + oracle" begin
            edges = [(1,2),(2,3),(3,1)]
            w = [0.5,-0.8,1.1]; h = [0.2,-0.1,0.4]; β = 0.3
            cpu = ising_network(3, edges, w, h, β; Dmax=-1, chi=10_000)
            gpu = ising_network(3, edges, w, h, β; Dmax=-1, chi=10_000)
            lnZ_cpu, = contraction!(cpu)
            lnZ_gpu, = contraction!(adapt(CuArray, gpu))
            @test lnZ_gpu ≈ lnZ_cpu rtol=1e-8
        end

        @testset "GPU 2D Ising grid (finite Dmax) ≈ CPU" begin
            L = 6; β = 0.4
            edges = Tuple{Int,Int}[]
            idx(i,j) = (i-1)*L + j
            for i in 1:L, j in 1:L
                j < L && push!(edges, (idx(i,j), idx(i,j+1)))
                i < L && push!(edges, (idx(i,j), idx(i+1,j)))
            end
            n = L*L; w = ones(length(edges)); hh = zeros(n)
            cpu = ising_network(n, edges, w, hh, β; Dmax=16, chi=32, compress=true)
            gpu = ising_network(n, edges, w, hh, β; Dmax=16, chi=32, compress=true)
            lnZ_cpu, = contraction!(cpu)
            lnZ_gpu, = contraction!(adapt(CuArray, gpu))
            @test lnZ_gpu ≈ lnZ_cpu rtol=1e-5
        end

        @testset "Float32 GPU path" begin
            ts = [randn(Float32,3,4), randn(Float32,4,5), randn(Float32,5,3)]
            ixs = [[:a,:b],[:b,:c],[:c,:a]]
            ref = exact_contract([Float64.(t) for t in ts], ixs)[]
            gtn = TensorNetwork([CuArray(t) for t in ts], ixs; Dmax=-1, chi=10_000)
            lnZ, err, psi = contraction!(gtn)
            @test exp(lnZ) * psi ≈ ref rtol=1e-3
        end

        @testset "cu(tn) yields a Float32 GPU network and matches CPU" begin
            # closed Ising network (Float64) -> cu downcasts to Float32
            edges = [(1,2),(2,3),(3,1)]
            β = 0.3
            cpu = ising_network(3, edges, [0.5,-0.8,1.1], [0.2,-0.1,0.4], β; Dmax=-1, chi=10_000)
            g   = ising_network(3, edges, [0.5,-0.8,1.1], [0.2,-0.1,0.4], β; Dmax=-1, chi=10_000)
            lnZ_cpu, = contraction!(cpu)
            gtn = cu(g)
            @test gtn isa TensorNetwork{Float32, <:CuArray{Float32,3}}
            lnZ_g, = contraction!(gtn)
            @test Float64(real(lnZ_g)) ≈ real(lnZ_cpu) rtol=1e-4       # Float32 precision
            # adapt(CuArray, ·) preserves Float64
            g2 = ising_network(3, edges, [0.5,-0.8,1.1], [0.2,-0.1,0.4], β; Dmax=-1, chi=10_000)
            a2 = adapt(CuArray, g2)
            @test a2 isa TensorNetwork{Float64, <:CuArray{Float64,3}}
            lnZ_a, = contraction!(a2)
            @test real(lnZ_a) ≈ real(lnZ_cpu) rtol=1e-10
        end

        @testset "complex GPU contraction matches CPU + oracle" begin
            ts = [randn(ComplexF64,3,4), randn(ComplexF64,4,5), randn(ComplexF64,5,3)]
            ixs = [[:a,:b],[:b,:c],[:c,:a]]
            ref = exact_contract(ts, ixs)[]
            gtn = TensorNetwork([CuArray(t) for t in ts], ixs; Dmax=-1, chi=10_000)
            lnZ, err, psi = contraction!(gtn)
            @test exp(lnZ) * psi ≈ ref rtol=1e-7
        end
    end
end
