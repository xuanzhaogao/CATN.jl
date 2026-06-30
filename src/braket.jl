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

# ---------------------------------------------------------------------------
# PairedEdge — bookkeeping for one original virtual bond
# ---------------------------------------------------------------------------

"""
    PairedEdge

Bookkeeping for one original virtual bond `(i, j)` in the double-layer network.
For each endpoint (`i` and `j`), we record which MPS leg index in the `BraKetNode`
corresponds to the ket and bra leg of that bond.

Fields:
- `i`, `j`          : node IDs of the two endpoints
- `ket_leg_i`       : 1-based MPS site index of the ket leg at node `i`
- `bra_leg_i`       : 1-based MPS site index of the bra leg at node `i`
- `ket_leg_j`       : 1-based MPS site index of the ket leg at node `j`
- `bra_leg_j`       : 1-based MPS site index of the bra leg at node `j`
"""
struct PairedEdge
    i::Int
    j::Int
    ket_leg_i::Int
    bra_leg_i::Int
    ket_leg_j::Int
    bra_leg_j::Int
end

# ---------------------------------------------------------------------------
# BraKetNetwork
# ---------------------------------------------------------------------------

"""
    BraKetNetwork{T,AT}

A double-layer (bra–ket) tensor network for computing `⟨ψ|ψ⟩`.
Each site `i` is stored as a `BraKetNode{T,AT}`.  Every original virtual bond
`(i,j)` is tracked as a `PairedEdge` holding the ket and bra MPS-leg indices
on both endpoints.

Contraction params and accumulators mirror `TensorNetwork`.
"""
mutable struct BraKetNetwork{T,AT<:AbstractArray{T,3}}
    tensors::Dict{Int,BraKetNode{T,AT}}
    edges::Vector{PairedEdge}          # one entry per original virtual bond
    lnZ::T
    psi::T
    chi::Int
    cutoff::Float64
    norm_method::Int
    svdopt::Bool
    swapopt::Bool
    maxdim_intermediate::Int
    rng::AbstractRNG
end

# ---------------------------------------------------------------------------
# braket_network constructor
# ---------------------------------------------------------------------------

"""
    braket_network(tensors, ixs; chi=64, cutoff=1e-15, norm_method=1,
                   svdopt=true, swapopt=true, seed=1) -> BraKetNetwork

Build a double-layer network from a ket state `(tensors, ixs)`.

`ixs[i]` are the index labels for `tensors[i]`.  Labels appearing in exactly
one tensor are physical indices; labels shared by two tensors are virtual bonds.
Each tensor must have **exactly one** physical index (v1).

A `BraKetNode` is built for every site. Each shared virtual bond becomes a
`PairedEdge` recording the ket and bra MPS-leg indices on both endpoints.

**Cyclic virtual-bond graphs are rejected** — v1 supports acyclic (chain/tree)
topologies only.
"""
function braket_network(tensors::Vector{<:AbstractArray},
                        ixs::Vector{<:AbstractVector};
                        chi::Int=64, cutoff::Float64=1e-15, norm_method::Int=1,
                        svdopt::Bool=true, swapopt::Bool=true, seed::Int=1)
    n = length(tensors)
    n == length(ixs) || error("braket_network: length(tensors) != length(ixs)")

    # ------------------------------------------------------------------
    # Step 1: classify labels as physical (open, count=1) or virtual (shared, count=2)
    # ------------------------------------------------------------------
    label_count   = Dict{Any,Int}()
    label_owners  = Dict{Any,Vector{Int}}()   # label -> [site_ids]
    for i in 1:n
        for l in ixs[i]
            label_count[l]  = get(label_count, l, 0) + 1
            owners          = get!(label_owners, l, Int[])
            push!(owners, i)
        end
    end

    # Validate: each virtual label appears in exactly 2 tensors
    for (l, cnt) in label_count
        cnt == 1 || cnt == 2 ||
            error("braket_network: label $l appears $cnt times (expected 1 or 2)")
    end

    # Each tensor must have exactly one physical label
    for i in 1:n
        nphys = count(l -> label_count[l] == 1, ixs[i])
        nphys == 1 || error("braket_network: tensor $i has $nphys physical (open) labels, expected 1")
    end

    # ------------------------------------------------------------------
    # Step 2: Detect cycles in the virtual-bond graph (acyclic = tree/chain only)
    # Union-Find over node IDs 1..n
    # ------------------------------------------------------------------
    parent = collect(1:n)
    function find_root(x)
        while parent[x] != x; parent[x] = parent[parent[x]]; x = parent[x]; end
        return x
    end
    function link!(x, y)
        rx, ry = find_root(x), find_root(y)
        rx == ry && error("braket_network: virtual-bond graph contains a cycle — cyclic graphs are not supported in v1")
        parent[rx] = ry
    end
    for (l, cnt) in label_count
        if cnt == 2
            i, j = label_owners[l][1], label_owners[l][2]
            link!(i, j)
        end
    end

    # ------------------------------------------------------------------
    # Step 3: Build BraKetNodes
    # For each tensor i, collect:
    #   - phys_pos : axis index of the physical label
    #   - ket_neighbors : neighbor node IDs for each virtual axis (in axis order)
    # ------------------------------------------------------------------
    T    = promote_type(map(eltype, tensors)...)

    node_dict = Dict{Int,BraKetNode}()
    for i in 1:n
        virt_axes   = Int[]
        ket_neighbors = Int[]
        phys_pos    = 0
        for (ax, l) in enumerate(ixs[i])
            if label_count[l] == 1
                phys_pos = ax
            else
                push!(virt_axes, ax)
                j = label_owners[l][1] == i ? label_owners[l][2] : label_owners[l][1]
                push!(ket_neighbors, j)
            end
        end
        phys_pos > 0 || error("braket_network: could not find physical axis for tensor $i")

        node = braket_node(tensors[i], ket_neighbors, phys_pos;
                           chi=chi, cutoff=cutoff, norm_method=norm_method,
                           svdopt=svdopt, swapopt=swapopt)
        node_dict[i] = node
    end

    # ------------------------------------------------------------------
    # Step 4: Build PairedEdge list
    # For each virtual bond (pair of sites), record the ket/bra MPS leg
    # indices on both endpoints using find_leg.
    # ------------------------------------------------------------------
    edges = PairedEdge[]
    for (l, cnt) in label_count
        cnt == 2 || continue
        i, j = label_owners[l][1], label_owners[l][2]
        ni   = node_dict[i]
        nj   = node_dict[j]
        ket_leg_i = find_leg(ni, j, true)
        bra_leg_i = find_leg(ni, j, false)
        ket_leg_j = find_leg(nj, i, true)
        bra_leg_j = find_leg(nj, i, false)
        push!(edges, PairedEdge(i, j, ket_leg_i, bra_leg_i, ket_leg_j, bra_leg_j))
    end

    # ------------------------------------------------------------------
    # Step 5: Infer type parameters and construct the network
    # ------------------------------------------------------------------
    # Get concrete node type from the first node
    first_node = first(values(node_dict))
    T_concrete  = eltype(eltype(first_node.mps))
    AT_concrete = eltype(first_node.mps)

    # Build a typed Dict
    typed_dict = Dict{Int,typeof(first_node)}()
    for (k, v) in node_dict
        typed_dict[k] = v
    end

    zero_T = zero(T_concrete)
    one_T  = one(T_concrete)
    rng    = MersenneTwister(seed)

    return BraKetNetwork{T_concrete,AT_concrete}(
        typed_dict, edges,
        zero_T,   # lnZ
        one_T,    # psi
        chi, cutoff, norm_method, svdopt, swapopt,
        0,        # maxdim_intermediate (starts at 0; updated to peak during contraction)
        rng
    )
end

# ---------------------------------------------------------------------------
# BraKetNode MPS movement helpers (mirror MPSNode's but also track `layer`)
# ---------------------------------------------------------------------------

"""
    swap!(node::BraKetNode, i, j) -> Float64

Swap adjacent MPS sites `i` and `j` (`|i-j|==1`).  Identical algorithm to
`MPSNode.swap!`, but permutes both `node.neighbor` **and** `node.layer` in
lockstep with the MPS sites so that the layer tag (ket/bra) rides along
correctly.  Returns the truncation error.
"""
function swap!(node::BraKetNode, i::Int, j::Int)
    mps = node.mps
    @assert abs(i - j) == 1 "swap!: i and j must be consecutive indices"
    @assert 1 <= i <= length(mps) && 1 <= j <= length(mps) "swap!: indices out of range"

    # Pre-move canonical center to the nearer of i or j if it's elsewhere
    if node.cano != i && node.cano != j
        target = abs(node.cano - i) < abs(node.cano - j) ? i : j
        cano_to!(node, target)
    end

    # Always work with left < right
    if i < j
        tl = mps[i]; tr = mps[j]
    else
        tl = mps[j]; tr = mps[i]
    end

    d0 = size(tl, 1)
    d1 = size(tr, 2)   # physical of tr moves to left after swap
    d2 = size(tl, 2)   # physical of tl moves to right after swap
    d3 = size(tr, 3)

    W = ein"ijk,kab->iajb"(tl, tr)    # (d0, d1, d2, d3)
    M = reshape(W, d0 * d1, d2 * d3)

    rows, cols = size(M)
    U, S, V, err = if node.swapopt && ((rows > 7000 && cols > 7000) || rows > 20000 || cols > 20000)
        U2, S2, V2 = rsvd(M, node.chi, 10, 10)
        U2, S2, V2, 0.0
    else
        tsvd(M; cutoff=node.cutoff, maxdim=node.chi)
    end

    myd = length(S)
    error = err

    if i < j  # going right: center ends at j
        node.mps[i] = reshape(U, d0, d1, myd)
        node.mps[j] = reshape(Diagonal(S) * V', myd, d2, d3)
    else  # going left: center ends at j
        node.mps[j] = reshape(U * Diagonal(S), d0, d1, myd)
        node.mps[i] = reshape(copy(V'), myd, d2, d3)
    end

    node.cano = j

    # Rotate neighbor and layer in lockstep with the two MPS sites
    node.neighbor[i], node.neighbor[j] = node.neighbor[j], node.neighbor[i]
    node.layer[i],    node.layer[j]    = node.layer[j],    node.layer[i]

    return error
end

"""
    move!(node::BraKetNode, a, b) -> Float64

Move site `a` to position `b` via sequential `swap!` calls, tracking both
`neighbor` and `layer`.  Returns accumulated truncation error.
"""
function move!(node::BraKetNode, a::Int, b::Int)
    error = 0.0
    a == b && return error
    if b > a
        for idx in a:b-1
            error += swap!(node, idx, idx + 1)
        end
    else
        for idx in a:-1:b+1
            error += swap!(node, idx, idx - 1)
        end
    end
    return error
end

"""
    move2tail!(node::BraKetNode, idx) -> Float64

Move site `idx` to the last position.  Returns accumulated truncation error.
"""
function move2tail!(node::BraKetNode, idx::Int)
    error = 0.0
    n = length(node.mps)
    idx == n && (cano_to!(node, n); return error)
    for k in idx:n-1
        error += swap!(node, k, k + 1)
    end
    cano_to!(node, n)
    return error
end

"""
    move2head!(node::BraKetNode, idx) -> Float64

Move site `idx` to the first position.  Returns accumulated truncation error.
"""
function move2head!(node::BraKetNode, idx::Int)
    error = move!(node, idx, 1)
    cano_to!(node, 1)
    return error
end

# ---------------------------------------------------------------------------
# eat! for BraKetNode — contracts the PAIRED ket + bra legs of a shared bond
# ---------------------------------------------------------------------------

"""
    eat!(node_i::BraKetNode, node_j::BraKetNode, j_id::Int, i_id::Int)
         -> (lognorm::Float64, error::Float64, phase)

Contract the paired (ket, bra) virtual legs of the bond between nodes `i` and
`j`.  `j_id` is the neighbor id of `j` as recorded in `node_i.neighbor`
(i.e. how `i` labels `j`); `i_id` is the neighbor id of `i` as recorded in
`node_j.neighbor`.

Algorithm:
1. Move `i`'s bra leg (neighbor=`j_id`, layer=false) to position `n_i` first.
2. Re-find `i`'s ket leg (neighbor=`j_id`, layer=true) and move to `n_i-1`.
   (Bra-first order is required: a move to `n_i-1` never touches position `n_i`,
   so the bra placed there remains undisturbed.  Ket-first would have the bra
   move displace the ket via the (n_i-1, n_i) swap.)
3. Move `j`'s ket leg to position 1; re-find and move bra to position 2.
4. Fuse `i`'s last two sites into `W_i` (dl, Dk, Db); fuse `j`'s first two
   into `W_j` (Dk, Db, dr); contract `W_i` and `W_j` over Dk and Db →
   `mat` of shape (dl, dr).
5. Fold `mat` into the MPS, append `j`'s remaining sites (3:end), update
   `neighbor` and `layer`, normalize, return `(lognorm, error, phase)`.
"""
function eat!(node_i::BraKetNode{T}, node_j::BraKetNode{T}, j_id::Int, i_id::Int) where {T}
    error_acc = 0.0
    n_i = length(node_i.mps)
    n_j = length(node_j.mps)

    # -----------------------------------------------------------------------
    # Step 1: bring i's paired legs to the MPS tail (positions n_i-1, n_i)
    # Order matters: move bra to n_i FIRST (its final position), then re-find
    # the ket and move it to n_i-1.  A move ending at n_i-1 only touches
    # positions ≤ n_i-1 (swaps up to (n_i-2, n_i-1)) and never disturbs
    # position n_i, so the bra stays put.  The reverse (ket first) is NOT
    # safe: moving bra to n_i afterwards does a swap at (n_i-1, n_i) which
    # displaces the ket that was just placed at n_i-1.
    # -----------------------------------------------------------------------
    bi = find_leg(node_i, j_id, false)
    bi > 0 || error("eat!: bra leg to neighbor $j_id not found in node_i")
    error_acc += move!(node_i, bi, n_i)
    # Re-find ket after the bra move (neighbor/layer are updated by swap!)
    ki = find_leg(node_i, j_id, true)
    ki > 0 || error("eat!: ket leg to neighbor $j_id not found in node_i")
    error_acc += move!(node_i, ki, n_i - 1)

    # Canonicalize to the last site so it carries the un-normalized weight
    cano_to!(node_i, n_i)

    # -----------------------------------------------------------------------
    # Step 2: bring j's paired legs to the MPS head (positions 1, 2)
    # -----------------------------------------------------------------------
    kj = find_leg(node_j, i_id, true)
    kj > 0 || error("eat!: ket leg to neighbor $i_id not found in node_j")
    error_acc += move!(node_j, kj, 1)
    # Re-find bra after the ket move
    bj = find_leg(node_j, i_id, false)
    bj > 0 || error("eat!: bra leg to neighbor $i_id not found in node_j")
    error_acc += move!(node_j, bj, 2)

    # -----------------------------------------------------------------------
    # Step 3: contract the four boundary sites
    # A_ik = node_i.mps[n_i-1]:  shape (dl, Dk, χ_mid)
    # A_ib = node_i.mps[n_i]:    shape (χ_mid, Db, 1)
    # A_jk = node_j.mps[1]:      shape (1, Dk, χ_mid2)
    # A_jb = node_j.mps[2]:      shape (χ_mid2, Db, dr)
    # -----------------------------------------------------------------------
    A_ik = node_i.mps[n_i - 1]
    A_ib = node_i.mps[n_i]
    A_jk = node_j.mps[1]
    A_jb = node_j.mps[2]

    dl    = size(A_ik, 1)
    Dk    = size(A_ik, 2)
    Db    = size(A_ib, 2)
    dr    = size(A_jb, 3)

    # Fuse i's last two: (dl, Dk, χ_mid) × (χ_mid, Db, 1) → (dl, Dk, Db)
    W_i = reshape(ein"ijk,kab->ijab"(A_ik, A_ib), dl, Dk, Db)
    # Fuse j's first two: (1, Dk, χ_mid2) × (χ_mid2, Db, dr) → (Dk, Db, dr)
    W_j = reshape(ein"ijk,kab->ijab"(A_jk, A_jb), Dk, Db, dr)

    # Contract over Dk and Db: (dl, Dk, Db) × (Dk, Db, dr) → (dl, dr)
    mat = ein"iab,abj->ij"(W_i, W_j)   # (dl, dr)

    # -----------------------------------------------------------------------
    # Step 4: fold mat back into the MPS, handle all four cases
    # -----------------------------------------------------------------------
    if n_i == 2 && n_j == 2
        # Case A: pure scalar — both nodes are exhausted
        z    = sum(mat)
        absz = abs(z)
        empty!(node_i.mps)
        empty!(node_i.neighbor)
        empty!(node_i.layer)
        if absz <= node_i.cutoff
            return (0.0, error_acc, one(T))
        end
        return (log(absz), error_acc, z / absz)

    elseif n_i == 2
        # Case B: i is exhausted; mat (1,dr) prefixes j's remaining sites
        new_left = reshape(mat, 1, 1, dr)   # (1, 1, dr): trivial phys dim
        # Absorb mat into j's third site (which becomes the new first)
        new_first = ein"ijk,kab->iab"(new_left, node_j.mps[3])  # (1, d3, dr3)
        node_i.mps     = vcat([new_first], node_j.mps[4:end])
        node_i.neighbor = node_j.neighbor[3:end]
        node_i.layer    = node_j.layer[3:end]
        node_i.cano     = 1

        # re-canonicalize to tail
        cano_to!(node_i, length(node_i.mps))

        center = node_i.mps[node_i.cano]
        if node_i.norm_method == 1
            norm = LinearAlgebra.norm(vec(center))
        elseif node_i.norm_method == 2
            norm = maximum(abs, center)
        else
            norm = one(real(T))
        end
        norm <= node_i.cutoff && return (0.0, error_acc, one(T))
        node_i.mps[end] ./= norm
        return (log(norm), error_acc, one(T))

    elseif n_j == 2
        # Case C: j is exhausted; mat (dl,1) is folded into i's site n_i-2
        new_tensor = ein"ijk,ka->ija"(node_i.mps[n_i - 2], mat)  # (dl2, d, 1)
        node_i.mps[n_i - 2] = new_tensor
        node_i.cano = n_i - 2
        # pop last 2 sites (the contracted ket + bra)
        pop!(node_i.mps); pop!(node_i.mps)
        # remove their neighbor/layer entries
        deleteat!(node_i.neighbor, length(node_i.neighbor))
        deleteat!(node_i.neighbor, length(node_i.neighbor))
        deleteat!(node_i.layer,    length(node_i.layer))
        deleteat!(node_i.layer,    length(node_i.layer))

        cano_to!(node_i, length(node_i.mps))

        center = node_i.mps[node_i.cano]
        if node_i.norm_method == 1
            norm = LinearAlgebra.norm(vec(center))
        elseif node_i.norm_method == 2
            norm = maximum(abs, center)
        else
            norm = one(real(T))
        end
        norm <= node_i.cutoff && return (0.0, error_acc, one(T))
        node_i.mps[end] ./= norm
        return (log(norm), error_acc, one(T))

    else
        # Case D: general — fold mat into i's site n_i-2, append j's 3:end
        new_tensor = ein"ijk,ka->ija"(node_i.mps[n_i - 2], mat)  # (dl2, d, dr)
        node_i.mps[n_i - 2] = new_tensor
        node_i.cano = n_i - 2
        # pop last 2 sites (contracted ket + bra of i)
        pop!(node_i.mps); pop!(node_i.mps)
        deleteat!(node_i.neighbor, length(node_i.neighbor))
        deleteat!(node_i.neighbor, length(node_i.neighbor))
        deleteat!(node_i.layer,    length(node_i.layer))
        deleteat!(node_i.layer,    length(node_i.layer))

        # append j's remaining sites (3:end)
        append!(node_i.mps,     node_j.mps[3:end])
        append!(node_i.neighbor, node_j.neighbor[3:end])
        append!(node_i.layer,    node_j.layer[3:end])

        # re-canonicalize: sweep from n_i-2 to new tail
        cano_to!(node_i, length(node_i.mps))

        center = node_i.mps[node_i.cano]
        if node_i.norm_method == 1
            norm = LinearAlgebra.norm(vec(center))
        elseif node_i.norm_method == 2
            norm = maximum(abs, center)
        else
            norm = one(real(T))
        end
        norm <= node_i.cutoff && return (0.0, error_acc, one(T))
        node_i.mps[end] ./= norm
        return (log(norm), error_acc, one(T))
    end
end

# ---------------------------------------------------------------------------
# compress! / compress_opt! for BraKetNode
# ---------------------------------------------------------------------------

"""
    compress!(node::BraKetNode) -> Float64

Compress the whole MPS by first left-canonicalizing (moving `cano` to the last
site), then sweeping right-to-left with two-site SVD truncation to `node.chi`.

For each step `j` from `length(mps)` down to `2` (with `i = j-1`):
1. Fuse sites `i` and `j` via `ein"ijk,kab->ijab"` reshaped to `(d0*d1, d2*d3)`.
2. Truncated SVD to `node.chi`.
3. Write back `mps[i] = reshape(U*Diagonal(S), d0, d1, :)`,
   `mps[j] = reshape(V', :, d2, d3)`.

Compression re-SVDs adjacent sites in place and does NOT permute site order,
so `neighbor` and `layer` are untouched.  Sets `cano = 1` and returns the
accumulated truncation error (sum of discarded singular values over all steps).
"""
function compress!(node::BraKetNode)
    error = 0.0
    isempty(node.mps) && return error
    left_canonical!(node)   # cano now at last site
    mps = node.mps
    for j in length(mps):-1:2
        i = j - 1
        tl = mps[i]
        tr = mps[j]
        d0, d1 = size(tl, 1), size(tl, 2)
        d2, d3 = size(tr, 2), size(tr, 3)
        mat = reshape(ein"ijk,kab->ijab"(tl, tr), d0 * d1, d2 * d3)
        U, S, V, err = tsvd(mat; cutoff=node.cutoff, maxdim=node.chi)
        error += err
        myd = length(S)
        mps[i] = reshape(U * Diagonal(S), d0, d1, myd)
        mps[j] = reshape(copy(V'), myd, d2, d3)
    end
    node.cano = 1
    return error
end

"""
    compress_opt!(node::BraKetNode) -> Float64

Like `compress!` but uses QR decomposition before SVD to reduce the size of
the matrix passed to SVD (mirrors `MPSNode.compress_opt!`).

Returns the accumulated truncation error.  `neighbor` and `layer` are untouched.
"""
function compress_opt!(node::BraKetNode)
    error = 0.0
    isempty(node.mps) && return error
    left_canonical!(node)   # cano now at last site
    mps = node.mps
    for j in length(mps):-1:2
        i = j - 1
        tl = mps[i]
        tr = mps[j]
        d0, d1 = size(tl, 1), size(tl, 2)
        d2, d3 = size(tr, 2), size(tr, 3)
        dd = size(tl, 3)   # == size(tr, 1)

        matl = reshape(tl, d0 * d1, dd)
        matr = reshape(tr, dd, d2 * d3)

        flag_left  = false
        flag_right = false
        local Ql, Qr

        ET = eltype(matl)
        if size(matl, 1) > size(matl, 2)
            flag_left = true
            Fl = qr(matl)
            Ql = _thin_q(Fl.Q, matl, ET, size(matl, 2))
            Rl = copy(Fl.R)
        else
            Rl = matl
        end

        if size(matr, 1) < size(matr, 2)
            flag_right = true
            Fr = qr(matr')
            Qr = _thin_q(Fr.Q, matr', ET, size(matr', 2))
            Rr = copy(Fr.R)
        else
            Rr = matr'
        end

        mat = Rl * Rr'
        U, S, V, err = tsvd(mat; cutoff=node.cutoff, maxdim=node.chi)
        error += err
        myd = length(S)

        U = U * Diagonal(S)
        flag_left  && (U = Ql * U)
        flag_right && (V = Qr * V)

        mps[i] = reshape(U, d0, d1, myd)
        mps[j] = reshape(copy(V'), myd, d2, d3)
    end
    node.cano = 1
    return error
end

# ---------------------------------------------------------------------------
# BraKet lognorm helper (mirrors TensorNetwork.network_lognorm)
# ---------------------------------------------------------------------------

"""
    braket_lognorm(bk) -> (lognorm::Float64, sign)

Sum of per-node norms for all nodes in `bk.tensors` that still have MPS sites.
A fully-contracted network leaves a single node whose MPS encodes the scalar.
"""
function braket_lognorm(bk::BraKetNetwork{T}) where {T}
    lognorm_total = zero(real(T))
    sign_total    = one(T)
    for (_, node) in bk.tensors
        isempty(node.mps) && continue
        # Canonicalize to the last site to extract the norm
        cano_to!(node, length(node.mps))
        center = node.mps[node.cano]
        if node.norm_method == 1
            nm = real(T)(LinearAlgebra.norm(vec(center)))
        elseif node.norm_method == 2
            nm = real(T)(maximum(abs, center))
        else
            nm = one(real(T))
        end
        nm <= node.cutoff && continue
        lognorm_total += log(nm)
        # sign is just 1 for real norms; for complex we accumulate the phase
        sign_total *= nm > 0 ? one(T) : -one(T)
    end
    return (lognorm_total, sign_total)
end

# ---------------------------------------------------------------------------
# BraKet contraction! — full loop for acyclic network
# ---------------------------------------------------------------------------

"""
    contraction!(bk::BraKetNetwork{T}) -> (lnZ, error, psi)

Contract the entire `BraKetNetwork` by repeatedly consuming one original virtual
bond at a time (via `eat!`) and compressing after each step.

**Edge selection (v1):** sequentially (in insertion order of `bk.edges`), with
`order(i) ≥ order(j)` enforced by swapping `i` and `j` if needed.

**Post-eat bookkeeping:** after `eat!(node_i, node_j, ...)`,
- the eaten `PairedEdge` is removed from `bk.edges`;
- any `PairedEdge` that still refers to `j` is re-pointed to `i`, and its
  MPS-leg indices are re-looked-up via `find_leg` (positions may have changed
  because `eat!` moves legs around via `swap!`);
- `node_j` is removed from `bk.tensors`.

Returns `(lnZ, error, psi)` where `⟨ψ|ψ⟩ = exp(lnZ) * psi`.
"""
function contraction!(bk::BraKetNetwork{T}) where {T}
    error_acc = 0.0
    bk.psi    = one(T)
    bk.lnZ   = zero(T)

    # We iterate over a snapshot of the edge list; the live list shrinks as we eat.
    # We process edges in order; the live bk.edges may have changed node IDs so
    # we always work from the current bk.edges.
    while !isempty(bk.edges)
        # Pick the first remaining edge (sequential selection for v1; acyclic ⇒ correctness)
        pe = bk.edges[1]
        i, j = pe.i, pe.j

        node_i = bk.tensors[i]
        node_j = bk.tensors[j]

        # Ensure order(i) >= order(j)
        if order(node_j) > order(node_i)
            i, j = j, i
            node_i, node_j = node_j, node_i
        end

        # Re-find the current leg positions (they may differ from pe's stored values
        # if a previous eat! moved them around within node_i or node_j).
        # j_id is how node_i refers to node_j (i.e. j); i_id is how node_j refers to i.
        j_id = j   # the neighbor ID of node_j as seen by node_i
        i_id = i   # the neighbor ID of node_i as seen by node_j

        # eat! j into i
        lognorm_val, eat_err, phase = eat!(node_i, node_j, j_id, i_id)
        error_acc   += eat_err
        bk.psi      *= phase
        bk.lnZ      += lognorm_val

        # compress! the merged node (environment-aware truncation of combined internal bonds)
        if bk.svdopt
            error_acc += compress_opt!(node_i)
        else
            error_acc += compress!(node_i)
        end

        # Track maxdim_intermediate
        for t in node_i.mps
            d = size(t, 2)
            if d > bk.maxdim_intermediate
                bk.maxdim_intermediate = d
            end
        end

        # --- Post-eat bookkeeping ---
        # 1. Remove the eaten edge from bk.edges (it is always the first element,
        #    but to be safe we remove the specific PairedEdge we just consumed).
        #    We search by node IDs (the current i and j, which may have been swapped).
        eaten_i = i;  eaten_j = j
        filter!(e -> !((e.i == eaten_i && e.j == eaten_j) ||
                        (e.i == eaten_j && e.j == eaten_i)), bk.edges)

        # 2. Re-point every PairedEdge that still refers to j → i.
        #    Re-find the MPS leg positions via find_leg (positions changed after eat!).
        for k in eachindex(bk.edges)
            e = bk.edges[k]
            if e.i == j
                # This edge was (j, other); now it is (i, other) since j was absorbed into i.
                other  = e.j
                # Re-find legs in merged node_i
                kl_i   = find_leg(node_i, other, true)   # ket leg to `other` in node_i
                bl_i   = find_leg(node_i, other, false)  # bra leg to `other` in node_i
                other_node = bk.tensors[other]
                kl_o   = find_leg(other_node, i, true)   # ket leg to i in other (node_j's old neighbors now point to i)
                bl_o   = find_leg(other_node, i, false)
                # If other_node still has legs pointing to j (not yet repointed), look for j too
                if kl_o == 0
                    kl_o = find_leg(other_node, j, true)
                    bl_o = find_leg(other_node, j, false)
                    # Update other_node's neighbor entries: replace j with i
                    for m in eachindex(other_node.neighbor)
                        other_node.neighbor[m] == j && (other_node.neighbor[m] = i)
                    end
                    # Re-find after update
                    kl_o = find_leg(other_node, i, true)
                    bl_o = find_leg(other_node, i, false)
                end
                bk.edges[k] = PairedEdge(i, other, kl_i, bl_i, kl_o, bl_o)

            elseif e.j == j
                # This edge was (other, j); now it is (other, i).
                other  = e.i
                # Re-find legs
                other_node = bk.tensors[other]
                kl_o   = find_leg(other_node, i, true)
                bl_o   = find_leg(other_node, i, false)
                if kl_o == 0
                    kl_o = find_leg(other_node, j, true)
                    bl_o = find_leg(other_node, j, false)
                    for m in eachindex(other_node.neighbor)
                        other_node.neighbor[m] == j && (other_node.neighbor[m] = i)
                    end
                    kl_o = find_leg(other_node, i, true)
                    bl_o = find_leg(other_node, i, false)
                end
                kl_i   = find_leg(node_i, other, true)
                bl_i   = find_leg(node_i, other, false)
                bk.edges[k] = PairedEdge(other, i, kl_o, bl_o, kl_i, bl_i)
            end
        end

        # 3. Remove j from bk.tensors
        delete!(bk.tensors, j)
    end

    # --- After loop: fold remaining per-node norms ---
    # For a fully-contracted acyclic network there is exactly one node left.
    # Its MPS encodes the (normalized) scalar; its norm is the remaining contribution.
    ln, sg = braket_lognorm(bk)
    bk.lnZ += ln
    bk.psi  *= sg

    return (bk.lnZ, error_acc, bk.psi)
end
