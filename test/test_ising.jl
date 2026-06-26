using CATN: ising_network, free_energy, contraction!, magnetization, correlation
using Test
using LinearAlgebra

# analytic lnZ of an open Ising chain of L spins, coupling J, no field:
# Z = 2^L * (cosh(βJ))^(L-1) * ... ; use transfer matrix for the ground truth
function chain_lnZ(L, J, β)
    T = [exp(β*J) exp(-β*J); exp(-β*J) exp(β*J)]
    v = ones(2)
    M = v' * T^(L-1) * v
    return log(M)
end

@testset "Ising chain lnZ" begin
    L, J, β = 6, 0.7, 0.4
    edges = [(i, i+1) for i in 1:L-1]
    tn = ising_network(L, edges, fill(J, L-1), zeros(L), β; Dmax=-1, chi=10_000)
    lnZ, err, psi = contraction!(tn)
    @test lnZ ≈ chain_lnZ(L, J, β) rtol=1e-9
end

@testset "Ising small loop matches brute force" begin
    # triangle with random couplings and fields, brute-force Z
    β = 0.3
    edges = [(1,2),(2,3),(3,1)]
    w = [0.5, -0.8, 1.1]; h = [0.2, -0.1, 0.4]
    function brute(edges, w, h, β, n)
        Z = 0.0
        for bits in 0:(2^n-1)
            s = [(bits >> (k-1)) & 1 == 1 ? 1 : -1 for k in 1:n]
            E = sum(w[e]*s[edges[e][1]]*s[edges[e][2]] for e in eachindex(edges)) +
                sum(h[k]*s[k] for k in 1:n)
            Z += exp(β*E)
        end
        return log(Z)
    end
    tn = ising_network(3, edges, w, h, β; Dmax=-1, chi=10_000)
    lnZ, _, _ = contraction!(tn)
    @test lnZ ≈ brute(edges, w, h, β, 3) rtol=1e-9
end

@testset "magnetization & correlation vs brute force" begin
    β = 0.35
    edges = [(1,2),(2,3),(3,1)]
    w = [0.4, 0.6, -0.3]; h = [0.1, -0.2, 0.05]; n = 3
    function brute_obs(edges, w, h, β, n)
        Z = 0.0; m = zeros(n); c = zeros(length(edges))
        for bits in 0:(2^n-1)
            s = [(bits >> (k-1)) & 1 == 1 ? 1 : -1 for k in 1:n]
            E = sum(w[e]*s[edges[e][1]]*s[edges[e][2]] for e in eachindex(edges)) +
                sum(h[k]*s[k] for k in 1:n)
            p = exp(β*E); Z += p
            for k in 1:n; m[k] += p*s[k]; end
            for e in eachindex(edges); c[e] += p*s[edges[e][1]]*s[edges[e][2]]; end
        end
        return m./Z, c./Z
    end
    m_ref, c_ref = brute_obs(edges, w, h, β, n)
    m = magnetization(n, edges, w, h, β; Dmax=-1, chi=10_000)
    c = correlation(n, edges, w, h, β; Dmax=-1, chi=10_000)
    @test m ≈ m_ref rtol=1e-7
    @test c ≈ c_ref rtol=1e-7
end

@testset "free_energy consistency" begin
    # Build a triangle Ising system and verify the free_energy return values
    # are self-consistent with the total lnZ from an independent contraction.
    β = 0.45
    n = 3
    edges = [(1,2),(2,3),(3,1)]
    w = [0.6, -0.5, 0.9]; h = [0.15, -0.3, 0.2]
    # Reference total lnZ from a freshly-built identical network
    tn_ref = ising_network(n, edges, w, h, β; Dmax=-1, chi=10_000)
    lnZ_total, _, _ = contraction!(tn_ref)
    # free_energy builds and contracts the same network internally
    tn_fe = ising_network(n, edges, w, h, β; Dmax=-1, chi=10_000)
    lnZ_per_site, F = free_energy(tn_fe)
    # Check per-site relationship
    @test lnZ_per_site ≈ lnZ_total / n rtol=1e-9
    # Check free energy formula: F = -lnZ_per_site / β
    @test F ≈ -lnZ_total / (n * β) rtol=1e-9
end

@testset "isolated spin lnZ" begin
    # A single spin with no edges and an external field h.
    # The exact partition function is Z = exp(β*h) + exp(-β*h) = 2*cosh(β*h),
    # so lnZ = log(2*cosh(β*h)).  This exercises the degree-0 / num_isolated path.
    β = 0.6
    h = 0.5
    tn = ising_network(1, Tuple{Int,Int}[], Float64[], [h], β; Dmax=-1, chi=10_000)
    lnZ, err, psi = contraction!(tn)
    @test lnZ ≈ log(2 * cosh(β * h)) rtol=1e-9
    # Zero-field case: Z = 2 → lnZ = log(2)
    tn0 = ising_network(1, Tuple{Int,Int}[], Float64[], [0.0], β; Dmax=-1, chi=10_000)
    lnZ0, _, _ = contraction!(tn0)
    @test lnZ0 ≈ log(2.0) rtol=1e-9
    # Two isolated spins with different fields: lnZ is additive
    tn2 = ising_network(2, Tuple{Int,Int}[], Float64[], [h, -h], β; Dmax=-1, chi=10_000)
    lnZ2, _, _ = contraction!(tn2)
    @test lnZ2 ≈ log(2 * cosh(β * h)) + log(2 * cosh(β * h)) rtol=1e-9
end
