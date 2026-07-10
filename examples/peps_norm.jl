# =============================================================================
# Random bra-ket quantum PEPS norm  ⟨ψ|ψ⟩  contracted with CATN
# =============================================================================
#
# A PEPS (Projected Entangled Pair State) on an L×L open square lattice: each
# site carries a tensor  A[p; u,d,l,r]  with a physical index p (dim `d`) and one
# virtual index per neighbour (dim `D`; corners rank-2, edges rank-3, interior
# rank-4).  Entries are i.i.d. complex Gaussian — a random quantum state.
#
# The norm is the *double-layer* network obtained by contracting each ket tensor
# against its conjugate bra over the physical index:
#
#     E[(u u'),(d d'),(l l'),(r r')] = Σ_p A[p;u,d,l,r] · conj(A[p;u',d',l',r']),
#
# so every virtual bond gets dimension D×D = D².  Contracting the resulting
# network of E tensors gives  ⟨ψ|ψ⟩ = Σ_{physical config} |ψ(config)|²  — a real,
# positive scalar.  This is the canonical hard 2D contraction (norm of a PEPS).
#
# Verification (small L), fully independent of the double-layer trick:
#   * amplitude oracle: exactly contract the SINGLE ket layer (physical legs open)
#     into the full amplitude tensor ψ via OMEinsum, then ⟨ψ|ψ⟩ = Σ|ψ|².
#   * einsum oracle:     exactly contract the E-network scalar via OMEinsum.
#   * CATN double-layer contraction — must match both, and be real & positive.
#
# For larger L we truncate (finite Dmax).  NOTE: random PEPS have flat
# entanglement spectra, so truncation is genuinely hard here — the reported error
# does NOT shrink to machine precision the way the *structured* Ising example did.
# That is the honest, expected behaviour for a random state.
#
# Run with:  julia --project=. examples/peps_norm.jl
# =============================================================================

using CATN
using CATN: TensorNetwork, contraction!
using OMEinsum
using LinearAlgebra
using Random
using Printf

# -----------------------------------------------------------------------------
# Exact reference contractor (OMEinsum, greedy order). Output = labels seen once.
# -----------------------------------------------------------------------------
function exact_contract(tensors, ixs)
    ixs_vv = [collect(ix) for ix in ixs]
    LT = eltype(eltype(ixs_vv))
    counts = Dict{LT,Int}(); order = LT[]
    for ix in ixs_vv, l in ix
        haskey(counts, l) || push!(order, l)
        counts[l] = get(counts, l, 0) + 1
    end
    iy = [l for l in order if counts[l] == 1]
    code = EinCode(ixs_vv, iy)
    sd = OMEinsum.get_size_dict(ixs_vv, tensors)
    opt = optimize_code(code, sd, GreedyMethod())
    return opt(tensors...)
end

# -----------------------------------------------------------------------------
# Building the random PEPS and its double-layer norm network
# -----------------------------------------------------------------------------

grid_id(r, c, L) = (r - 1) * L + c

"""
    double_layer_peps_site(A, deg, D, d) -> Array (deg legs, each dim D²)

Merge ket `A` with conjugate bra over the physical index:
E[m₁,…,m_deg] = Σ_p A[p,l₁,…] · conj(A[p,l₁',…]),  m_k fuses (l_k, l_k').
`A` has shape (d, D,…,D) with `deg` virtual legs.
"""
function double_layer_peps_site(A::AbstractArray, deg::Int, D::Int, d::Int)
    Amat = reshape(A, d, D^deg)                 # (p, α),  α = (l₁,…,l_deg), l₁ fastest
    M2 = transpose(Amat) * conj(Amat)           # (α, β) = Σ_p A[p,α] conj(A[p,β])
    T = reshape(M2, ntuple(_ -> D, 2deg)...)     # (l₁,…,l_deg, l₁',…,l_deg')
    perm = Int[]                                  # interleave to (l₁,l₁', l₂,l₂', …)
    for k in 1:deg
        push!(perm, k); push!(perm, deg + k)
    end
    Tp = permutedims(T, perm)
    return reshape(Tp, ntuple(_ -> D^2, deg)...)  # fuse each (l_k,l_k') → dim-D² leg
end

"""
    build_peps_norm(L, D, d, rng) -> (ket_tensors, ket_ixs, E_tensors, E_ixs)

Random complex PEPS on the L×L open lattice. `ket_*` describe the single ket
layer with OPEN physical legs (for the amplitude oracle); `E_*` describe the
double-layer norm network. Both use the SAME random tensors and a consistent
N,S,W,E leg order, with shared bond labels (:h,r,c)/(:v,r,c) and unique physical
labels (:p,r,c).
"""
function build_peps_norm(L::Int, D::Int, d::Int, rng::AbstractRNG)
    ket_tensors = Array{ComplexF64}[]; ket_ixs = Vector{Tuple{Symbol,Int,Int}}[]
    E_tensors   = Array{ComplexF64}[]; E_ixs   = Vector{Tuple{Symbol,Int,Int}}[]
    for r in 1:L, c in 1:L                       # site order = grid_id 1..L²
        bonds = Tuple{Symbol,Int,Int}[]
        r > 1 && push!(bonds, (:v, r - 1, c))    # N
        r < L && push!(bonds, (:v, r,     c))    # S
        c > 1 && push!(bonds, (:h, r, c - 1))    # W
        c < L && push!(bonds, (:h, r, c))        # E
        deg = length(bonds)
        A = randn(rng, ComplexF64, d, ntuple(_ -> D, deg)...)
        push!(ket_tensors, A)
        push!(ket_ixs, vcat([(:p, r, c)], bonds))     # physical leg first, then N,S,W,E
        push!(E_tensors, double_layer_peps_site(A, deg, D, d))
        push!(E_ixs, bonds)                            # N,S,W,E (physical merged away)
    end
    return ket_tensors, ket_ixs, E_tensors, E_ixs
end

# ⟨ψ|ψ⟩ from CATN's double-layer contraction: value = exp(lnZ)·psi.
function catn_norm(E_tensors, E_ixs; Dmax, chi)
    tn = TensorNetwork(E_tensors, E_ixs; Dmax=Dmax, chi=chi)
    lnZ, err, psi = contraction!(tn)
    return lnZ, psi, err, tn.maxdim_intermediate
end

# -----------------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------------

function verify_small(L; D=2, d=2, seed=1234)
    rng = MersenneTwister(seed)
    ket_t, ket_ix, E_t, E_ix = build_peps_norm(L, D, d, rng)

    # Oracle 2 (always cheap enough here): exact E-network scalar via OMEinsum.
    norm_ein = exact_contract(E_t, E_ix)[]                      # scalar

    # CATN exact double-layer contraction.
    lnZ, psi, err, maxd = catn_norm(E_t, E_ix; Dmax=-1, chi=100_000)
    norm_catn = exp(lnZ) * psi

    @printf("\nL=%d, D=%d, d=%d  (physical legs N=%d, double-layer bond dim=%d)\n",
            L, D, d, L * L, D^2)
    @printf("  einsum oracle   ⟨ψ|ψ⟩ = %.10f  (imag %.1e)\n", real(norm_ein), imag(norm_ein))
    @printf("  CATN double     ⟨ψ|ψ⟩ = %.10f  (imag %.1e)  [err=%.1e, maxdim=%d]\n",
            real(norm_catn), imag(norm_catn), err, maxd)

    checks = Bool[]
    push!(checks, isapprox(norm_catn, norm_ein; rtol=1e-8))
    push!(checks, real(norm_catn) > 0)
    push!(checks, abs(imag(norm_catn)) < 1e-8 * abs(real(norm_catn)))

    # Amplitude oracle (independent of the E construction) — only where cheap.
    if L * L <= 12
        psi_full = exact_contract(ket_t, ket_ix)               # ψ[p₁,…,p_N]
        norm_amp = sum(abs2, psi_full)                         # Σ|ψ|²
        @printf("  amplitude oracle Σ|ψ|² = %.10f  (validates E construction)\n", norm_amp)
        push!(checks, isapprox(real(norm_catn), norm_amp; rtol=1e-8))
    end

    ok = all(checks)
    println(ok ? "  ✓ PASS: CATN reproduces ⟨ψ|ψ⟩ exactly; result is real & positive." :
                 "  ✗ FAIL: mismatch!")
    return ok
end

function demo_large(L; D=2, d=2, seed=7, Dmaxes=(8, 16, 32), chi=128)
    rng = MersenneTwister(seed)
    _, _, E_t, E_ix = build_peps_norm(L, D, d, rng)
    @printf("\nL=%d, D=%d  (256... N=%d spins, double-layer bond dim=%d) — truncated\n",
            L, D, L * L, D^2)
    for Dmax in Dmaxes
        t = @elapsed ((lnZ, psi, err, maxd) = catn_norm(E_t, E_ix; Dmax=Dmax, chi=chi))
        @printf("  Dmax=%-3d  ln⟨ψ|ψ⟩/N = %.6f   arg(psi)=%+.1e   err=%.2e  maxdim=%d  (%.1fs)\n",
                Dmax, real(lnZ) / (L * L), angle(psi), err, maxd, t)
    end
end

function main()
    println("Random bra-ket quantum PEPS norm ⟨ψ|ψ⟩ via CATN")
    println("="^72)
    println("Small lattices — exact, verified against independent oracles")
    println("="^72)
    verify_small(3; D=2, d=2)
    verify_small(4; D=2, d=2)

    println("\n" * "="^72)
    println("Larger lattice — truncated (random PEPS ⇒ truncation is genuinely hard)")
    println("="^72)
    demo_large(8; D=2, d=2, Dmaxes=(8, 16, 32), chi=128)

    println("\nNote: for a RANDOM PEPS the singular-value spectra are flat, so the")
    println("truncation error stays sizeable and ln⟨ψ|ψ⟩/N keeps drifting with Dmax —")
    println("unlike the structured Ising case. This is the expected hard-instance behaviour.")
end

main()
