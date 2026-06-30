using CATN
using CATN: BraKetNode, braket_node, mps2raw, cano_to!, left_canonical!
using CATN: BraKetNetwork, braket_network
using OMEinsum, LinearAlgebra, Test

# Exact ⟨ψ|ψ⟩ via a direct double-layer contraction (independent oracle).
function exact_norm(tensors, ixs)
    n = length(tensors)
    # relabel: ket keeps labels; bra gets distinct labels for virtual bonds, shares physical.
    counts = Dict{Any,Int}(); for ix in ixs, l in ix; counts[l]=get(counts,l,0)+1; end
    bra_ix = [map(l -> counts[l]==2 ? (l, :bra) : l, ix) for ix in ixs]  # virtual→distinct, physical shared
    all_t   = vcat(collect(tensors), [conj(t) for t in tensors])
    all_ix  = vcat([collect(ix) for ix in ixs], [collect(b) for b in bra_ix])
    return exact_contract(all_t, all_ix)[]
end

@testset "braket" begin
    @testset "braket_node builds the double tensor E_i (complex, degree-2)" begin
        # degree-2 site: axes (v1, p, v2); physical index is axis 2
        D, d = 3, 2
        Ti = randn(ComplexF64, D, d, D)
        node = braket_node(Ti, [10, 20], 2; chi=10_000)   # neighbors 10 (v1), 20 (v2)
        # exact E_i[v1,v2,v1',v2'] = Σ_p Ti[v1,p,v2] conj(Ti)[v1',p,v2']
        # ein"apb,cpd->abcd" gives indices (a=v1, b=v2, c=v1', d=v2')
        E = ein"apb,cpd->abcd"(Ti, conj(Ti))              # (v1, v2, v1', v2')
        # node site order: [ket v1, ket v2, bra v1', bra v2']
        raw = mps2raw(node)                                # dims (v1, v2, v1', v2')
        @test size(raw) == (D, D, D, D)
        @test raw ≈ E
        @test node.layer == [true, true, false, false]    # ket block then bra block
        @test node.neighbor == [10, 20, 10, 20]
        # physical legs (dim of site 2) are ≤ D; internal junction bond ≤ max(D,d)
        @test all(size(s,2) ≤ D for s in node.mps)
        @test maximum(size(s,3) for s in node.mps[1:end-1]) ≤ max(D, d)
    end

    @testset "braket_node builds the double tensor E_i (real, degree-2)" begin
        D, d = 4, 3
        Ti = randn(Float64, D, d, D)
        node = braket_node(Ti, [1, 2], 2; chi=10_000)
        E = ein"apb,cpd->abcd"(Ti, conj(Ti))
        raw = mps2raw(node)
        @test size(raw) == (D, D, D, D)
        @test raw ≈ E
        @test node.layer == [true, true, false, false]
        @test node.neighbor == [1, 2, 1, 2]
    end

    @testset "braket_node builds the double tensor E_i (degree-1 leaf, complex)" begin
        # degree-1 site: axes (v1, p); physical index is axis 2
        D, d = 3, 2
        Ti = randn(ComplexF64, D, d)
        node = braket_node(Ti, [5], 2; chi=10_000)
        # E[v1, v1'] = Σ_p Ti[v1,p] * conj(Ti)[v1',p]
        E = ein"ap,cp->ac"(Ti, conj(Ti))                  # (v1, v1')
        raw = mps2raw(node)
        @test size(raw) == (D, D)
        @test raw ≈ E
        @test node.layer == [true, false]
        @test node.neighbor == [5, 5]
    end

    @testset "braket_node builds the double tensor E_i (degree-3, real)" begin
        D, d = 2, 2
        Ti = randn(Float64, D, d, D, D)
        # physical index is axis 2, virtual axes 1,3,4
        node = braket_node(Ti, [10, 20, 30], 2; chi=10_000)
        # E[v1,v2,v3,v1',v2',v3'] = Σ_p Ti[v1,p,v2,v3] * conj(Ti)[v1',p,v2',v3']
        # ein indices: a=v1,p=phys,b=v2,c=v3 for ket; d=v1',e=v2',f=v3' for bra
        E = ein"apbc,dpef->abcdef"(Ti, conj(Ti))
        raw = mps2raw(node)
        @test size(raw) == (D, D, D, D, D, D)
        @test raw ≈ E
        @test node.layer == [true, true, true, false, false, false]
        @test node.neighbor == [10, 20, 30, 10, 20, 30]
    end

    @testset "braket_node is left-canonical" begin
        D, d = 4, 2
        Ti = randn(ComplexF64, D, d, D)
        node = braket_node(Ti, [1, 2], 2; chi=10_000)
        @test node.cano == length(node.mps)
        # all sites before the center are left-isometric: M'*M ≈ I
        for k in 1:node.cano-1
            A = node.mps[k]; dl, dp, dr = size(A)
            M = reshape(A, dl*dp, dr)
            @test M' * M ≈ I atol=1e-8
        end
        # canonicalization preserves the represented tensor
        E = ein"apb,cpd->abcd"(Ti, conj(Ti))   # (v1, v2, v1', v2')
        @test mps2raw(node) ≈ E
    end
end

@testset "braket_network construction" begin
    # 3-site chain (acyclic): T1(p1,a) T2(a,p2,b) T3(b,p3)
    T1 = randn(ComplexF64, 2, 3)
    T2 = randn(ComplexF64, 3, 2, 3)
    T3 = randn(ComplexF64, 3, 2)
    tensors = [T1, T2, T3]
    ixs = [[:p1, :a], [:a, :p2, :b], [:b, :p3]]
    bk = braket_network(tensors, ixs; chi=10_000)
    @test bk isa BraKetNetwork
    @test length(bk.tensors) == 3
    # each node reconstructs its E_i (spot-check node 2 has 4 physical legs: 2 ket + 2 bra)
    @test count(bk.tensors[2].layer) == 2 && count(!, bk.tensors[2].layer) == 2

    # Cyclic graph (triangle) should be rejected
    Ta = randn(ComplexF64, 2, 2, 2)
    Tb = randn(ComplexF64, 2, 2, 2)
    Tc = randn(ComplexF64, 2, 2, 2)
    tensors_cyc = [Ta, Tb, Tc]
    ixs_cyc = [[:p1, :a, :c], [:a, :p2, :b], [:b, :p3, :c]]
    @test_throws Exception braket_network(tensors_cyc, ixs_cyc)
end
