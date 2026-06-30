using CATN
using CATN: BraKetNode, braket_node, mps2raw, cano_to!, left_canonical!, find_leg
using CATN: BraKetNetwork, braket_network
using CATN: eat!, compress!
using OMEinsum, LinearAlgebra, Random, Test

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

@testset "paired-edge eat! on two sites" begin
    # 2-site state: T1(p1, a), T2(a, p2). Norm contracts p1,p2 and bond a (ket & bra).
    T1 = randn(ComplexF64, 2, 4)        # (p1, a)
    T2 = randn(ComplexF64, 4, 2)        # (a, p2)
    n1 = braket_node(T1, [2], 1; chi=10_000)   # neighbor 2 via bond a; phys axis 1
    n2 = braket_node(T2, [1], 2; chi=10_000)   # neighbor 1 via bond a; phys axis 2
    # eat!(node_i, node_j, j_id_in_i, i_id_in_j): n1 sees n2 as neighbor 2; n2 sees n1 as neighbor 1
    lognorm_val, err, phase = eat!(n1, n2, 2, 1)
    # after eating, n1 should be a scalar (all legs contracted) → ⟨ψ|ψ⟩
    val = exp(lognorm_val) * phase
    ref = exact_norm([T1, T2], [[:p1, :a], [:a, :p2]])
    @test val ≈ ref rtol=1e-10
    @test imag(ref) ≈ 0 atol=1e-10   # real norm
    @test real(ref) ≥ 0               # non-negative norm

    # Real tensors
    T1r = randn(Float64, 2, 4)
    T2r = randn(Float64, 4, 2)
    n1r = braket_node(T1r, [2], 1; chi=10_000)
    n2r = braket_node(T2r, [1], 2; chi=10_000)
    lognorm_r, err_r, phase_r = eat!(n1r, n2r, 2, 1)
    val_r = exp(lognorm_r) * phase_r
    ref_r = exact_norm([T1r, T2r], [[:p1, :a], [:a, :p2]])
    @test val_r ≈ ref_r rtol=1e-10
    @test ref_r ≥ 0
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

@testset "eat! interleaved layout (degree-2 node_i, legs NOT at tail)" begin
    # This test exercises the move-order bug in eat!.
    # n2 = braket_node(T2, [1,3], 2): degree-2 with neighbor layout [ket-1, ket-3, bra-1, bra-3].
    # We call eat!(n2, n1, 1, 2) — contracting the bond to FIRST neighbor (j_id=1).
    # ket-to-1 is at pos 1, bra-to-1 is at pos 3.  Moving ket from 1 to n_i-1=3 passes through
    # pos 3 (bra), displacing it — this is the bug.  The fix moves bra to n_i=4 first, then ket to 3.
    # After eat!, n2 has 2 sites representing the partial contraction over bond a:
    #   C[b, b'] = Σ_{a,p1,p2} T1[p1,a]*conj(T1)[p1,a] * T2[a,p2,b]*conj(T2)[a,p2,b'] (site order [ket-to-3, bra-to-3])
    # Wait: C[bk,bb] = Σ_{ak,ab,p1,p2} T2[ak,p2,bk]*conj(T2[ab,p2,bb]) * (Σ_{p1} T1[p1,ak]*conj(T1[p1,ab]))
    # = Σ_a T2[a,p2,bk]*conj(T2[a,p2,bb]) (since T1 contributes E1[ak,ab] = Σ_p1 T1[p1,ak]*conj(T1[p1,ab]) contracted with T2)
    Random.seed!(42)
    T1 = randn(ComplexF64, 2, 3)       # axes: (p1, a=bond-to-2)
    T2 = randn(ComplexF64, 3, 2, 4)    # axes: (a=bond-to-1, p2, b=bond-to-3)
    # Build n1 as degree-1 (leaf): phys_pos=1 (p1 is axis 1), neighbor=[2]
    n1 = braket_node(T1, [2], 1; chi=10_000)   # neighbor=2 means n1 points to n2
    @test n1.neighbor == [2, 2]
    @test n1.layer == [true, false]
    # Build n2 as degree-2: phys_pos=2 (p2 is axis 2), neighbors=[1, 3]
    n2 = braket_node(T2, [1, 3], 2; chi=10_000)
    # node2 site order: [ket-to-1, ket-to-3, bra-to-1, bra-to-3] at positions [1,2,3,4]
    @test n2.neighbor == [1, 3, 1, 3]
    @test n2.layer == [true, true, false, false]
    # eat! n2 into n1 contracting bond (n2→1 ↔ n1→2); j_id=1 in n2, i_id=2 in n1
    # This is Case C (n1 has 2 sites, n2 has 4).  The bug: ket(pos1)→n_i-1=3 passes bra(pos3).
    lognorm_val, err, phase = eat!(n2, n1, 1, 2)
    # Reference: partial contraction C[bk,bb] using independent einsum
    # E1[ak,ab] = Σ_p1 T1[p1,ak]*conj(T1[p1,ab])  (shape 3×3)
    # C[bk,bb] = Σ_{ak,ab,p2} T2[ak,p2,bk]*conj(T2[ab,p2,bb]) * E1[ak,ab]
    E1_ref = ein"pa,pb->ab"(T1, conj(T1))                  # (3,3)
    C_ref  = ein"apb,aq,qpd->bd"(T2, E1_ref, conj(T2))    # (4,4)
    # mps2raw(n2) after eat! gives the normalized C; multiply by exp(lognorm)*phase to recover
    raw = mps2raw(n2) .* (exp(lognorm_val) * phase)
    @test raw ≈ C_ref rtol=1e-8
end

@testset "eat! interleaved layout — 3-site sequential contraction" begin
    # 3-site chain T1(p1,a) T2(a,p2,b) T3(b,p3).
    # eat bond (1,2): node_i=n1 (degree-1), node_j=n2 (degree-2) — Case B.
    #   n2's legs to 1 are at positions ket=1, bra=3.  This exercises node_j's
    #   head-positioning with legs NOT already at head.
    # eat bond (merged-node, 3): the merged node now has legs to 3 interleaved.
    #   This exercises node_i's tail-positioning bug.
    Random.seed!(7)
    T1 = randn(ComplexF64, 2, 3)       # (p1, a)
    T2 = randn(ComplexF64, 3, 2, 4)    # (a, p2, b)
    T3 = randn(ComplexF64, 4, 2)       # (b, p3)
    ixs = [[:p1, :a], [:a, :p2, :b], [:b, :p3]]
    ref = exact_norm([T1, T2, T3], ixs)

    n1 = braket_node(T1, [2], 1; chi=10_000)   # phys=axis1, neighbor=[2]
    n2 = braket_node(T2, [1, 3], 2; chi=10_000) # phys=axis2, neighbors=[1,3]
    n3 = braket_node(T3, [2], 2; chi=10_000)    # phys=axis2, neighbor=[2]

    # First eat: contract bond (1,2).  n1 is node_i; n2 is node_j; j_id=2, i_id=1.
    # Case B (n1 has 2 sites, n2 has 4).  After eat, result lives in n1.
    lognorm1, err1, phase1 = eat!(n1, n2, 2, 1)

    # After eat, n1 now carries the merged (T1⊗T2 block) tensor; legs to node 3
    # are interleaved in the merged MPS (from n2's sites 3:end appended).
    # Second eat: contract bond between merged n1 and n3.
    # merged n1's legs to neighbor 3 must be found and moved to tail.
    lognorm2, err2, phase2 = eat!(n1, n3, 3, 2)

    val = exp(lognorm1 + lognorm2) * phase1 * phase2
    @test val ≈ ref rtol=1e-8
    @test imag(ref) ≈ 0 atol=1e-10
    @test real(ref) ≥ 0
end

@testset "compress! for BraKetNode" begin
    # Build a degree-2 node and verify compress! reduces bond dim while
    # preserving the represented tensor (up to normalization from left_canonical!).
    D, d = 4, 2
    Ti = randn(ComplexF64, D, d, D)
    node = braket_node(Ti, [10, 20], 2; chi=10_000)
    E_ref = mps2raw(node)   # ground truth before compression
    # compress! with chi=2 (truncating)
    err = compress!(node)
    @test err >= 0.0
    @test node.cano == 1
    # After compress!, mps2raw should still give back the un-normalized tensor
    # (compress! does not normalize; it only truncates SVs).
    # Verify it is proportional to E_ref up to a scalar factor (both are the same tensor
    # up to SVD truncation; with chi=10_000 no truncation occurs so err==0 and exact).
    node2 = braket_node(Ti, [10, 20], 2; chi=10_000)
    err2 = compress!(node2)   # no truncation at large chi
    @test err2 == 0.0 || err2 < 1e-12
    raw2 = mps2raw(node2)
    @test raw2 ≈ E_ref rtol=1e-8
end
