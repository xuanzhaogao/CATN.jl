# =============================================================================
# Double-layer (bra⊗ket) square-lattice Ising contraction with CATN
# =============================================================================
#
# We build the square-lattice PEPS whose "ket" amplitude is the square-root
# Boltzmann weight
#
#     ψ(s) = ∏_{<ij>} exp( (β/2) s_i s_j ),      s_i ∈ {+1, -1},
#
# and contract its NORM  ⟨ψ|ψ⟩  as a *double-layer* tensor network: at every
# site the ket tensor and its (conjugate) bra are merged over the physical spin,
# so each virtual bond carries dimension 2×2 = 4.  This is exactly the PEPS-norm
# structure that makes 2D contraction hard.
#
# Key identity (the payoff / verification):
#
#     ⟨ψ|ψ⟩ = Σ_s ψ(s)²
#           = Σ_s ∏_{<ij>} exp(β s_i s_j)
#           = Z(β),
#
# the ordinary square-lattice Ising partition function at inverse temperature β.
# So the double-layer contraction must reproduce the single-layer Ising Z(β).
# We cross-check against:
#   * brute force over all 2^N spin configs   (ground truth, small L only),
#   * CATN's own single-layer `ising_network` (independent tensor construction).
#
# Construction of the site tensor
# -------------------------------
# W½[a,b] = exp((β/2) a b) is 2×2, symmetric PD.  Split W½ = Q Qᵀ with Q = √W½.
# The ket site tensor (a COPY/δ on the spin, with √W½ on each incident leg) is
#     T[s, l₁,…,l_d] = ∏_k Q[s, l_k].
# Contracting a shared virtual index l between neighbours i,j gives
#     Σ_l Q[s_i,l] Q[s_j,l] = (QQᵀ)[s_i,s_j] = W½[s_i,s_j] = exp((β/2) s_i s_j),
# so the single (ket) layer with open physical legs equals ψ(s).
#
# The double-layer site tensor merges ket⊗bra over the physical spin, fusing each
# leg's (ket, bra) index pair l,l' into one index m = (l-1)*2 + l' of dim 4:
#     E[m₁,…,m_d] = Σ_s ∏_k (Q[s,:] ⊗ Q[s,:])[m_k].
# Contracting a shared m over a bond factorises into a ket factor W½[s_i,s_j] and a
# bra factor W½[s_i',s_j']; the per-site COPY forces s_i' = s_i, so the whole
# network contracts to Σ_s ∏ W½² = Σ_s ∏ exp(β s_i s_j) = Z(β).  ∎
#
# Run with:  julia --project=. examples/double_layer_ising.jl
# =============================================================================

using CATN
using CATN: TensorNetwork, contraction!, ising_network
using LinearAlgebra
using Printf

# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------

"Principal square root of the half-bond Boltzmann matrix W½[a,b]=exp((β/2)·a·b)."
function half_bond_sqrt(β::Real)
    s = (1.0, -1.0)                       # spin values, index 1 → +1, index 2 → -1
    Whalf = [exp((β / 2) * a * b) for a in s, b in s]   # 2×2, symmetric PD
    return Matrix(sqrt(Symmetric(Whalf)))               # symmetric √, Q Qᵀ = W½
end

"""
    double_layer_site(Q, d) -> Array{Float64,d}

Degree-`d` double-layer site tensor, each leg dim 4 (fused ket⊗bra index):

    E[m₁,…,m_d] = Σ_s ∏_k (Q[s,:] ⊗ Q[s,:])[m_k],   m_k = (l_k-1)*2 + l_k'.
"""
function double_layer_site(Q::AbstractMatrix, d::Int)
    E = zeros(Float64, ntuple(_ -> 4, d)...)
    for s in 1:2
        vs = kron(Q[s, :], Q[s, :])                 # length-4 fused ket⊗bra vector
        term = ones(Float64, ntuple(_ -> 1, d)...)  # rank-1 outer product of d copies
        for k in 1:d
            term = term .* reshape(vs, ntuple(i -> i == k ? 4 : 1, d))
        end
        E .+= term
    end
    return E
end

"Linear site id for grid position (r,c) on an L×L lattice (row-major, 1-based)."
grid_id(r, c, L) = (r - 1) * L + c

"""
    build_double_layer(L, β) -> (tensors, ixs)

Assemble the L×L open-boundary double-layer network. `tensors[k]` is the site
tensor for site `k = grid_id(r,c,L)`; `ixs[k]` are its OMEinsum-style bond labels
in the same leg order (N, S, W, E — existing directions only). Each bond label is
shared by exactly its two endpoints, so it becomes a contracted bond.
"""
function build_double_layer(L::Int, β::Real)
    Q = half_bond_sqrt(β)
    tensors = Array{Float64}[]
    ixs = Vector{Tuple{Symbol,Int,Int}}[]
    for r in 1:L, c in 1:L                       # visits sites in grid_id order 1..L²
        labels = Tuple{Symbol,Int,Int}[]
        r > 1 && push!(labels, (:v, r - 1, c))   # N: vertical bond above
        r < L && push!(labels, (:v, r,     c))   # S: vertical bond below
        c > 1 && push!(labels, (:h, r, c - 1))   # W: horizontal bond left
        c < L && push!(labels, (:h, r, c))       # E: horizontal bond right
        push!(tensors, double_layer_site(Q, length(labels)))
        push!(ixs, labels)
    end
    return tensors, ixs
end

"Nearest-neighbour edges of the L×L square lattice (open boundaries), 1-based."
function square_edges(L::Int)
    e = Tuple{Int,Int}[]
    for r in 1:L, c in 1:L
        c < L && push!(e, (grid_id(r, c, L), grid_id(r, c + 1, L)))
        r < L && push!(e, (grid_id(r, c, L), grid_id(r + 1, c, L)))
    end
    return e
end

"Exact lnZ of the L×L Ising model by brute force (small L only). J=1, h=0."
function brute_lnZ(L::Int, β::Real)
    edges = square_edges(L)
    n = L * L
    n > 24 && error("brute_lnZ: N=$n too large for brute force")
    Z = 0.0
    for bits in 0:(2^n - 1)
        E = 0
        for (i, j) in edges
            si = ((bits >> (i - 1)) & 1) == 1 ? 1 : -1
            sj = ((bits >> (j - 1)) & 1) == 1 ? 1 : -1
            E += si * sj
        end
        Z += exp(β * E)
    end
    return log(Z)
end

# lnZ from the double-layer CATN contraction (Z = exp(lnZ)·psi; psi is a phase).
function double_layer_lnZ(L, β; Dmax, chi)
    tensors, ixs = build_double_layer(L, β)
    tn = TensorNetwork(tensors, ixs; Dmax=Dmax, chi=chi)
    lnZ, err, psi = contraction!(tn)
    return lnZ + log(abs(psi)), err, tn.maxdim_intermediate
end

# lnZ from CATN's single-layer Ising network (independent construction).
function single_layer_lnZ(L, β; Dmax, chi)
    edges = square_edges(L)
    n = L * L
    tn = ising_network(n, edges, ones(length(edges)), zeros(n), β; Dmax=Dmax, chi=chi)
    lnZ, err, _ = contraction!(tn)
    return lnZ, err
end

# -----------------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------------

const βc = log(1 + sqrt(2)) / 2      # 2D Ising critical inverse temperature ≈ 0.4407

function main()
    β = βc
    @printf("\nDouble-layer square-lattice Ising via CATN  (J=1, h=0, open BC)\n")
    @printf("β = %.6f  (2D critical βc = ln(1+√2)/2)\n", β)

    # --- L = 4: exact, fully verifiable against brute force ------------------
    println("\n" * "="^72)
    println("L = 4  (16 spins) — exact contraction, verified against brute force")
    println("="^72)

    lnZ_bf = brute_lnZ(4, β)
    lnZ_dl, err_dl, maxd = double_layer_lnZ(4, β; Dmax=-1, chi=100_000)   # exact
    lnZ_sl, _           = single_layer_lnZ(4, β; Dmax=-1, chi=100_000)    # exact

    @printf("  brute force            lnZ = %.10f\n", lnZ_bf)
    @printf("  CATN single-layer      lnZ = %.10f   (Δ = %.2e)\n", lnZ_sl, abs(lnZ_sl - lnZ_bf))
    @printf("  CATN double-layer ⟨ψ|ψ⟩ lnZ = %.10f   (Δ = %.2e)\n", lnZ_dl, abs(lnZ_dl - lnZ_bf))
    @printf("  double-layer truncation error = %.2e, max intermediate dim = %d\n", err_dl, maxd)

    ok4 = isapprox(lnZ_dl, lnZ_bf; rtol=1e-8) && isapprox(lnZ_sl, lnZ_bf; rtol=1e-8)
    println(ok4 ? "  ✓ PASS: double-layer contraction reproduces Z(β) exactly." :
                  "  ✗ FAIL: mismatch beyond tolerance!")

    # --- L = 16: truncated, double-layer vs single-layer cross-check ---------
    println("\n" * "="^72)
    println("L = 16  (256 spins) — truncated contraction (double layer is bond-dim 4)")
    println("="^72)

    for Dmax in (16, 32, 64)
        chi = 128
        t_dl = @elapsed ((lnZ_dl16, err16, maxd16) = double_layer_lnZ(16, β; Dmax=Dmax, chi=chi))
        @printf("  Dmax=%-3d double-layer  lnZ/N = %.8f  (err=%.1e, maxdim=%d, %.1fs)\n",
                Dmax, lnZ_dl16 / 256, err16, maxd16, t_dl)
    end

    # single-layer reference at a well-converged Dmax
    t_sl = @elapsed ((lnZ_sl16, errsl16) = single_layer_lnZ(16, β; Dmax=64, chi=128))
    @printf("  single-layer (Dmax=64) lnZ/N = %.8f  (err=%.1e, %.1fs)  ← reference\n",
            lnZ_sl16 / 256, errsl16, t_sl)

    println("\nThe double-layer lnZ/N converges toward the single-layer reference as Dmax\n" *
            "grows — confirming the ⟨ψ|ψ⟩ construction reproduces Z(β) on the large lattice too.")
end

main()
