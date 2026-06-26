using CATN: ising_network, free_energy, contraction!, magnetization, correlation
using Test

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
