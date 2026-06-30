"""
    BraKetNode{T,AT<:AbstractArray{T,3}}

An MPS node representing the double tensor `E_i = Σ_p T_i ⊗ conj(T_i)` of a
physical site tensor `T_i`.  Physical (virtual) legs are kept **separate**:
the first `deg` MPS sites carry the ket virtual bonds (tagged `layer=true`) and
the last `deg` MPS sites carry the bra virtual bonds (tagged `layer=false`).
`neighbor` repeats the neighbor ids in each block.  The combined internal bond
(created by contracting the shared physical index) is small (≤ `d_phys`) and is
the *only* bond that is ever truncated during contraction.

**MPS leg convention (per site):** `(left_bond, physical_leg, right_bond)`.
Site order: `[ket v₁, …, ket v_deg, bra v₁, …, bra v_deg]`.

**Internal junction bond (between last ket site and first bra site):** dim ≤ `d_phys`.
  - Last ket site shape:  `(χ_l, D_deg, d)` — right bond = d
  - First bra site shape: `(d, D₁, χ_r)`   — left bond = d
Together these represent the contraction over the shared physical index `p`.
"""
mutable struct BraKetNode{T,AT<:AbstractArray{T,3}}
    mps::Vector{AT}          # combined-internal-bond MPS; each site (χ_l, D_k, χ_r)
    neighbor::Vector{Int}    # neighbor node id per MPS site
    layer::Vector{Bool}      # true = ket leg, false = bra leg, per MPS site
    cano::Int
    chi::Int
    cutoff::Float64
    norm_method::Int
    svdopt::Bool
    swapopt::Bool

    BraKetNode{T,AT}(mps, neighbor, layer, cano, chi, cutoff, norm_method, svdopt, swapopt) where {T,AT<:AbstractArray{T,3}} =
        new{T,AT}(mps, neighbor, layer, cano, chi, cutoff, norm_method, svdopt, swapopt)
end

# Outer positional constructor: infer AT from the mps vector.
function BraKetNode(mps::Vector{<:AbstractArray{T,3}}, neighbor::Vector{Int},
                    layer::Vector{Bool}, cano::Int, chi::Int, cutoff::Float64,
                    norm_method::Int, svdopt::Bool, swapopt::Bool) where {T}
    BraKetNode{T,eltype(mps)}(mps, neighbor, layer, cano, chi, cutoff, norm_method, svdopt, swapopt)
end

# ---------------------------------------------------------------------------
# Main constructor
# ---------------------------------------------------------------------------

"""
    braket_node(Ti, ket_neighbors, phys_pos; chi, cutoff, norm_method, svdopt, swapopt)
             -> BraKetNode

Build the double tensor `E_i = Σ_p T_i conj(T_i)` as an MPS with separate ket/bra
virtual legs, **without** forming the dense `D^(2·deg)` double tensor.

`Ti` is the ket site tensor; `phys_pos` is the 1-based axis index of the physical
(contracted) index.  `ket_neighbors` lists the neighbor node ids for the virtual axes
of `Ti` (all axes except `phys_pos`), in their natural order.

**Algorithm (no dense double tensor):**

1. Permute `Ti` → `Tk` of shape `(D₁, …, D_deg, d)` (virtual axes first, physical last).
2. **Ket MPS:** left-to-right SVD over axes 1…deg; physical leg `d` remains as the
   right boundary bond of the last ket site.
     - sites 1…deg-1: shape `(χ_l, D_k, χ_r)` (standard)
     - site deg (last ket): shape `(χ_l, D_deg, d)` ← right bond = d_phys
3. **Bra MPS:** permute `conj(Tk)` → `Tbra` of shape `(d, D₁, …, D_deg)` so that the
   physical leg `d` becomes the leftmost axis (= left boundary bond of the first bra site).
   Left-to-right SVD over axes D₁…D_deg:
     - site 1 (first bra): shape `(d, D₁, χ_r)` ← left bond = d_phys
     - sites 2…deg: shape `(χ_l, D_k, χ_r)` (standard)
4. **Join at p:** the last ket site and first bra site share bond dim d — no explicit
   contraction needed.  Concatenate to form the full MPS.

**Output site order:** `[ket v₁, …, ket v_deg, bra v₁, …, bra v_deg]`.
`mps2raw(node)` gives shape `(D₁_ket, …, D_deg_ket, D₁_bra, …, D_deg_bra)`.
"""
function braket_node(Ti::AbstractArray{T}, ket_neighbors::Vector{Int}, phys_pos::Int;
                     chi::Int=32, cutoff::Float64=1e-15, norm_method::Int=1,
                     svdopt::Bool=true, swapopt::Bool=false) where {T}
    nd  = ndims(Ti)
    deg = nd - 1     # number of virtual legs

    deg >= 1 || error("braket_node: degree-0 site (no virtual legs) is not supported.")
    length(ket_neighbors) == deg ||
        error("braket_node: length(ket_neighbors)=$(length(ket_neighbors)), expected $deg")

    # ------------------------------------------------------------------
    # Step 0: Permute Ti so virtual axes come first, physical axis last.
    # Tk has shape (D₁, …, D_deg, d).
    # ------------------------------------------------------------------
    virt_axes = [i for i in 1:nd if i != phys_pos]
    perm      = [virt_axes..., phys_pos]
    Tk        = permutedims(Ti, perm)     # (D₁, …, D_deg, d)
    dims_Tk   = size(Tk)
    d         = dims_Tk[deg+1]            # physical dimension

    AT_type   = typeof(similar(Ti, T, (1, 1, 1)))

    # ------------------------------------------------------------------
    # Step 1: Build ket MPS via left-to-right SVD.
    # We sweep over the deg virtual axes (axes 1…deg), carrying the physical
    # axis d as the right boundary bond of the last ket site.
    #
    # Start with R = reshape(Tk, 1, D₁, …, D_deg, d) = (1, D₁, …, D_deg, d).
    # dleft = 1 initially.
    #
    # For i = 1 … deg-1:
    #   Di = dims_Tk[i]
    #   M  = reshape(R, dleft*Di, :)   — (dleft*Di) × (remaining dims)
    #   SVD: M ≈ U*S*V'
    #   site i ← reshape(U, dleft, Di, χ)
    #   R ← reshape(S*V', χ, dims_Tk[i+1:end]...)
    #   dleft ← χ
    #
    # Last ket site (i=deg): R has shape (dleft, D_deg, d).
    #   Store as reshape(R, dleft, D_deg, d) — right bond = d.
    # ------------------------------------------------------------------
    ket_sites = AT_type[]
    R     = reshape(Tk, 1, dims_Tk...)    # (1, D₁, …, D_deg, d)
    dleft = 1

    for i in 1:deg-1
        Di = dims_Tk[i]
        M  = reshape(R, dleft * Di, :)
        U, S, V = tsvd(M; cutoff=cutoff, maxdim=chi)
        χ  = length(S)
        push!(ket_sites, reshape(U, dleft, Di, χ))
        R  = reshape(Diagonal(S) * V', χ, dims_Tk[(i+1):end]...)
        dleft = χ
    end
    # Last ket site: R has shape (dleft, D_deg, d).
    push!(ket_sites, reshape(R, dleft, dims_Tk[deg], d))

    # ------------------------------------------------------------------
    # Step 2: Build bra MPS from conj(Tk).
    # Permute conj(Tk) to Tbra of shape (d, D₁, …, D_deg) so that d is axis 1
    # (this will become the left boundary bond of the first bra site).
    #
    # Start with R = Tbra (already shape (d, D₁, …, D_deg)).
    # dleft = d initially (the physical bond dimension, not yet "split off").
    #
    # For i = 1 … deg-1  (iterating over D₁, …, D_{deg-1}):
    #   Di = dims_B[i+1]   — dims_B[1]=d, dims_B[2]=D₁, …, dims_B[deg+1]=D_deg
    #   M  = reshape(R, dleft*Di, :)
    #   SVD → site i_bra = reshape(U, dleft, Di, χ)
    #   R ← reshape(S*V', χ, dims_B[i+2:end]...)
    #   dleft ← χ
    #
    # Last bra site (i=deg): R has shape (dleft, D_deg) → store as (dleft, D_deg, 1).
    #
    # Site 1 shape: (d, D₁, χ₁)   ← left bond = d = right bond of last ket site ✓
    # Site k shape: (χ_{k-1}, D_k, χ_k)  for k=2…deg
    # ------------------------------------------------------------------
    Tbra   = permutedims(conj(Tk), (deg+1, 1:deg...))   # (d, D₁, …, D_deg)
    dims_B = size(Tbra)    # dims_B[1]=d, dims_B[2]=D₁, …, dims_B[deg+1]=D_deg

    bra_sites = AT_type[]
    R     = Tbra            # already (d, D₁, …, D_deg) — no extra reshape needed
    dleft = d               # left bond of first bra site = d

    for i in 1:deg-1
        Di = dims_B[i+1]    # D₁, D₂, …, D_{deg-1}
        M  = reshape(R, dleft * Di, :)
        U, S, V = tsvd(M; cutoff=cutoff, maxdim=chi)
        χ  = length(S)
        push!(bra_sites, reshape(U, dleft, Di, χ))
        R  = reshape(Diagonal(S) * V', χ, dims_B[(i+2):end]...)
        dleft = χ
    end
    # Last bra site: R has shape (dleft, D_deg).
    push!(bra_sites, reshape(R, dleft, dims_B[deg+1], 1))

    # ------------------------------------------------------------------
    # Step 3: Concatenate ket + bra MPS.
    # The internal bond between last ket site (right=d) and first bra site (left=d)
    # represents the contracted physical index.  No explicit contraction needed.
    # ------------------------------------------------------------------
    all_sites    = AT_type[ket_sites..., bra_sites...]
    neighbor_vec = Int[ket_neighbors..., ket_neighbors...]
    layer_vec    = Bool[trues(deg)..., falses(deg)...]
    cano_val     = 1    # will be updated by left_canonical! below

    node = BraKetNode{T,AT_type}(all_sites, neighbor_vec, layer_vec,
                                  cano_val, chi, cutoff, norm_method, svdopt, swapopt)
    left_canonical!(node)
    return node
end

# ---------------------------------------------------------------------------
# Canonicalization for BraKetNode (mirrors MPSNode's cano_to! / left_canonical!)
# ---------------------------------------------------------------------------

"""
    cano_to!(node::BraKetNode, idx)

Move the canonical center of `node.mps` to position `idx` by sweeping
left-to-right (center < idx) or right-to-left (center > idx) with rank-
preserving re-SVDs.  Uses `cutoff=0.0` (keep all columns) just like
`MPSNode.cano_to!`.  Only `node.mps` and `node.cano` are modified;
`node.neighbor` and `node.layer` are untouched (canonicalization re-SVDs
adjacent sites in place — it does not permute site order).
"""
function cano_to!(node::BraKetNode, idx::Int)
    idx == 0 && (idx = length(node.mps))
    node.cano == idx && return node
    if node.cano < idx
        # Sweep right: move center from node.cano to idx
        for i in node.cano:idx-1
            dl, d, dr = size(node.mps[i])
            U, S, V = tsvd(reshape(node.mps[i], dl*d, :); cutoff=0.0)
            nkeep = count(>(node.cutoff), S)
            nkeep = nkeep == 0 ? length(S) : nkeep
            U = U[:, 1:nkeep]; S = S[1:nkeep]; V = V[:, 1:nkeep]
            node.mps[i] = reshape(U, dl, d, nkeep)
            R = Diagonal(S) * V'
            node.mps[i+1] = ein"ij,jab->iab"(R, node.mps[i+1])
            node.cano = i + 1
        end
    else
        # Sweep left: move center from node.cano to idx
        for i in node.cano:-1:idx+1
            dl, d, dr = size(node.mps[i])
            Mt = reshape(permutedims(node.mps[i], (2, 3, 1)), d*dr, dl)
            U, S, V = tsvd(Mt; cutoff=0.0)
            nkeep = count(>(node.cutoff), S)
            nkeep = nkeep == 0 ? length(S) : nkeep
            U = U[:, 1:nkeep]; S = S[1:nkeep]; V = V[:, 1:nkeep]
            node.mps[i] = permutedims(reshape(U, d, dr, nkeep), (3, 1, 2))
            R = Diagonal(S) * V'                     # shape (nkeep, dl)
            node.mps[i-1] = ein"abc,cd->abd"(node.mps[i-1], permutedims(R, (2, 1)))
            node.cano = i - 1
        end
    end
    return node
end

left_canonical!(node::BraKetNode) = (node.cano = 1; cano_to!(node, length(node.mps)))

# ---------------------------------------------------------------------------
# mps2raw for BraKetNode — generic MPS reconstruction, tags ignored
# ---------------------------------------------------------------------------

"""
    mps2raw(node::BraKetNode{T}) -> Array{T}

Reconstruct the full tensor from the MPS by contracting all internal bonds.
The result has one axis per MPS site (in site order), shape
`(D₁_ket, …, D_deg_ket, D₁_bra, …, D_deg_bra)`.

The ket physical legs (size(s,2) for ket sites) give the ket virtual dims;
the bra physical legs give the bra virtual dims.  The physical-index bond
(internal to the MPS, between last ket and first bra sites) is contracted
automatically as a regular MPS internal bond.
"""
function mps2raw(node::BraKetNode{T}) where {T}
    mps = node.mps
    isempty(mps) && return Array{T,0}(undef)
    A   = mps[1]
    # First site: shape (left=1, phys, right) — squeeze left bond (= 1).
    cur      = reshape(A, size(A, 2), size(A, 3))   # (phys₁, right₁)
    physdims = Int[size(A, 2)]
    for k in 2:length(mps)
        B        = mps[k]
        r, p, r2 = size(B)
        Bmat     = reshape(B, r, p * r2)
        res      = ein"ab,bc->ac"(cur, Bmat)        # (pdim, p*r2)
        pdim     = size(cur, 1)
        res3     = reshape(res, pdim, p, r2)
        pdim    *= p
        cur      = reshape(res3, pdim, r2)
        push!(physdims, p)
    end
    return reshape(cur, physdims...)
end

# ---------------------------------------------------------------------------
# Accessors (mirrors MPSNode)
# ---------------------------------------------------------------------------

order(node::BraKetNode) = length(node.mps)
shape(node::BraKetNode) = [size(t, 2) for t in node.mps]

"""
    find_leg(node::BraKetNode, j, isket) -> Int

Return the 1-based MPS site index for the leg connecting to neighbor `j` with
layer tag `isket` (`true` = ket, `false` = bra), or `0` if not found.
"""
function find_leg(node::BraKetNode, j::Int, isket::Bool)
    for k in eachindex(node.mps)
        node.neighbor[k] == j && node.layer[k] == isket && return k
    end
    return 0
end
