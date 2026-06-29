mutable struct TensorNetwork{T,AT<:AbstractArray{T,3}}
    tensors::Dict{Int,MPSNode{T,AT}}
    Dmax::Int
    chi::Int
    cutoff::Float64
    norm_method::Int
    select::Int
    reverse::Bool
    svdopt::Bool
    swapopt::Bool
    compress::Bool
    cut_bond::Bool
    edge_count::Dict{Int,Vector{Vector{Int}}}
    lnZ::T
    sign::T
    psi::T
    maxdim_intermediate::Int
    num_isolated::Int
    rng::AbstractRNG
    n::Int
    beta::Float64
end

"""
    TensorNetwork(tensors, ixs; Dmax=32, chi=32, cutoff=1e-15, norm_method=1,
                  select=1, reverse=true, svdopt=true, swapopt=true,
                  compress=false, cut_bond=false, seed=1)

Build a `TensorNetwork` from a vector of tensors and their OMEinsum-style index labels.

- A label shared by exactly two tensors becomes a bond.
- A label appearing once becomes an open leg (dangling), represented by a unique
  negative sentinel neighbor id (< 0) so it is never selected for contraction.
- A label appearing >2 times, or twice within one tensor, raises an error.

Leg→neighbor alignment: for tensor `t` with labels `ixs[t]`, MPS site `k`
corresponds to label `ixs[t][k]`, and `neighbor[k]` = the other node id sharing
that label (or a sentinel for open legs). This keeps `mps2raw(node)` leg order
aligned with `ixs[t]`.
"""
function TensorNetwork(
        tensors::Vector{<:AbstractArray{T}},
        ixs::Vector{<:AbstractVector};
        Dmax::Int=32,
        chi::Int=32,
        cutoff::Float64=1e-15,
        norm_method::Int=1,
        select::Int=1,
        reverse::Bool=true,
        svdopt::Bool=true,
        swapopt::Bool=true,
        compress::Bool=false,
        cut_bond::Bool=false,
        seed::Int=1) where {T}

    n = length(tensors)
    @assert length(ixs) == n "tensors and ixs must have the same length"

    # Validate self-loops: each label must be unique within one tensor's label list
    for t in 1:n
        seen = Set()
        for lbl in ixs[t]
            lbl in seen && error("Self-loop detected: label $lbl appears twice in tensor $t's index list")
            push!(seen, lbl)
        end
    end

    # Build label → [(node_id, leg_index), ...] map
    label_map = Dict{Any, Vector{Tuple{Int,Int}}}()
    for t in 1:n
        for (k, lbl) in enumerate(ixs[t])
            if !haskey(label_map, lbl)
                label_map[lbl] = Tuple{Int,Int}[]
            end
            push!(label_map[lbl], (t, k))
        end
    end

    # Validate: no label can appear >2 times
    for (lbl, occurrences) in label_map
        length(occurrences) > 2 && error("Label $lbl appears in $(length(occurrences)) tensors (max 2 allowed)")
    end

    # Assign unique negative sentinel ids for open legs (one per open-leg occurrence)
    sentinel_counter = Ref(0)
    next_sentinel() = (sentinel_counter[] -= 1; sentinel_counter[])

    # Build neighbor vector for each tensor
    # neighbor_vecs[t][k] = neighbor node id for leg k of tensor t
    neighbor_vecs = [Vector{Int}(undef, length(ixs[t])) for t in 1:n]
    for (lbl, occurrences) in label_map
        if length(occurrences) == 2
            # Bond: each side's neighbor is the other node
            (t1, k1), (t2, k2) = occurrences
            neighbor_vecs[t1][k1] = t2
            neighbor_vecs[t2][k2] = t1
        else
            # Open leg: assign a unique negative sentinel
            (t, k) = occurrences[1]
            neighbor_vecs[t][k] = next_sentinel()
        end
    end

    # Build MPSNode for each tensor
    AT = typeof(similar(tensors[1], T, (1, 1, 1)))
    nodes = Dict{Int,MPSNode{T,AT}}()
    for t in 1:n
        nodes[t] = MPSNode(tensors[t], neighbor_vecs[t];
                           chi=chi, cutoff=cutoff,
                           norm_method=norm_method,
                           svdopt=svdopt, swapopt=swapopt)
    end

    # Count isolated nodes: nodes with no real (non-sentinel) bond
    # A node is isolated if all its neighbor ids are negative (sentinels) or it has 0 legs
    num_isolated = count(t -> all(nb -> nb < 0, neighbor_vecs[t]), 1:n)

    TensorNetwork{T,AT}(
        nodes,
        Dmax,
        chi,
        cutoff,
        norm_method,
        select,
        reverse,
        svdopt,
        swapopt,
        compress,
        cut_bond,
        Dict{Int,Vector{Vector{Int}}}(),  # edge_count: empty, built in Task 10
        zero(T),                           # lnZ
        one(T),                            # sign
        one(T),                            # psi
        -1,                                # maxdim_intermediate
        num_isolated,
        MersenneTwister(seed),
        n,                                 # n: number of tensors (default)
        1.0                                # beta: default inverse temperature
    )
end

# ---------------------------------------------------------------------------
# Edge dimension helpers
# ---------------------------------------------------------------------------

"""
    dim_after_merge(tn, i, j) -> Int

Log₂ of the size of the intermediate tensor produced by contracting nodes i and j.
Mirrors `tn_np.py:dim_after_merge`.
"""
function dim_after_merge(tn::TensorNetwork, i::Int, j::Int)
    idx_j_in_i = find_neighbor(tn.tensors[i], j)
    return round(Int, logdim(tn.tensors[i]) + logdim(tn.tensors[j]) - 2 * logdim(tn.tensors[i], idx_j_in_i))
end

# ---------------------------------------------------------------------------
# edge_count bookkeeping
# ---------------------------------------------------------------------------

"""
    count_add_edges!(tn, edges)

For each sorted `[i,j]` pair in `edges`, compute `dim_after_merge(tn,i,j)` and
push the pair into the corresponding bucket of `tn.edge_count`.
"""
function count_add_edges!(tn::TensorNetwork, edges)
    for pair in edges
        i, j = pair[1], pair[2]
        cost = dim_after_merge(tn, i, j)
        if !haskey(tn.edge_count, cost)
            tn.edge_count[cost] = Vector{Int}[]
        end
        push!(tn.edge_count[cost], [i, j])
    end
end

"""
    count_add_nodes!(tn, nodes)

For each node id in `nodes`, collect its incident real bonds (neighbor > 0),
deduplicate into sorted `[i,j]` pairs, and call `count_add_edges!`.
Edges are processed in lexicographic order to ensure deterministic bucket ordering.
"""
function count_add_nodes!(tn::TensorNetwork, nodes)
    edges = Set{Vector{Int}}()
    for i in nodes
        for j in tn.tensors[i].neighbor
            j <= 0 && continue  # skip open legs / sentinels
            push!(edges, sort([i, j]))
        end
    end
    # Sort edges lexicographically for deterministic insertion order (matches Python reference)
    sorted_edges = sort(collect(edges))
    count_add_edges!(tn, sorted_edges)
end

"""
    count_remove_nodes!(tn, nodes)

For each node id in `nodes`, collect its incident real bonds, then remove those
pairs from the appropriate buckets in `tn.edge_count`.

!!! warning Ordering precondition
    This function MUST be called BEFORE any shape mutation (`eat!`/`merge!`) of the
    incident nodes.  It recomputes each edge's cost via `dim_after_merge`, which reads
    the current (live) node shapes.  If the shapes have already been mutated the cost
    lands in the wrong bucket and corrupts `edge_count`.
    (Mirrors the Python comment at tn_np.py:320:
     "take care of the count dictionary first because it depends on shape of tensors".)
"""
function count_remove_nodes!(tn::TensorNetwork, nodes)
    for node_id in nodes
        for nb in tn.tensors[node_id].neighbor
            nb <= 0 && continue  # skip open legs / sentinels
            i, j = sort([node_id, nb])
            cost = dim_after_merge(tn, i, j)
            if haskey(tn.edge_count, cost)
                filter!(pair -> pair != [i, j], tn.edge_count[cost])
            end
        end
    end
end

"""
    select_edge_init!(tn)

Clear `tn.edge_count` and rebuild it from all current real bonds in the network.
Mirrors `tn_np.py:select_edge_init`.
"""
function select_edge_init!(tn::TensorNetwork)
    empty!(tn.edge_count)
    count_add_nodes!(tn, collect(keys(tn.tensors)))
end

# ---------------------------------------------------------------------------
# Edge selectors
# ---------------------------------------------------------------------------

"""
    select_edge_min_dim(tn) -> (i, j)

Select the edge with the smallest `dim_after_merge` cost (cheapest contraction).
Updates `tn.maxdim_intermediate`. Mirrors `tn_np.py:select_edge_min_dim`.
"""
function select_edge_min_dim(tn::TensorNetwork)
    cost = minimum(k for (k, v) in tn.edge_count if !isempty(v))
    tn.maxdim_intermediate = max(tn.maxdim_intermediate, cost)
    pair = tn.edge_count[cost][1]
    return (pair[1], pair[2])
end

"""
    select_edge_min_dim_triangle(tn) -> (i, j)

Among the cheapest edges, pick the one whose endpoints share the most common
neighbors (triangle heuristic). Updates `tn.maxdim_intermediate`.
Mirrors `tn_np.py:select_edge_min_dim_triangle`.
"""
function select_edge_min_dim_triangle(tn::TensorNetwork)
    cost = minimum(k for (k, v) in tn.edge_count if !isempty(v))
    tn.maxdim_intermediate = max(tn.maxdim_intermediate, cost)
    candidates = tn.edge_count[cost]

    best_pair = candidates[1]
    best_count = -1

    for pair in candidates
        i, j = pair[1], pair[2]
        # Real neighbors of j (excluding i and sentinels); we check membership in i's neighbor list
        neigh_j = Set(nb for nb in tn.tensors[j].neighbor if nb > 0 && nb != i)
        # Common neighbors: neighbors of j that also neighbor i
        both = 0
        for k in neigh_j
            if find_neighbor(tn.tensors[k], i) > 0
                both += 1
            end
        end
        if both > best_count
            best_count = both
            best_pair = pair
        end
    end

    return (best_pair[1], best_pair[2])
end

"""
    select_edge_sequentially(tn) -> (i, j)

Select the real bond `(i, j)` (with `i < j`) minimising `i + j` by scanning ALL
current real edges directly from the live node neighbor lists.  This is robust to
stale `edge_count` state: it never touches `tn.edge_count` and therefore works
correctly even when `select_edge_init!` has not been called.

Mirrors `tn_np.py:select_edge_sequentially` (which iterates `self.G.edges()`).
"""
function select_edge_sequentially(tn::TensorNetwork)
    best = nothing
    bestsum = typemax(Int)
    for i in keys(tn.tensors)
        for nb in tn.tensors[i].neighbor
            nb > 0 || continue          # skip open-leg sentinels
            i < nb || continue          # each undirected edge once, canonical order
            s = i + nb
            if s < bestsum
                bestsum = s
                best = (i, nb)
            end
        end
    end
    best === nothing && error("select_edge_sequentially: no edges remain")
    return best
end

# ---------------------------------------------------------------------------
# Physical-bond truncation
# ---------------------------------------------------------------------------

"""
    cut_bondim!(tn, i, idx_j_in_i) -> Float64

SVD-truncate the shared physical bond between node `i` (leg `idx_j_in_i`) and
its neighbor `j` to `tn.Dmax`.  Returns the discarded singular-value weight.

When `tn.Dmax < 0` the function is a no-op and returns `0.0` (exact mode).

Index math (mirrors `tn_np.py:482`):
- `j = node_i.neighbor[idx_j_in_i]`
- `idx_i_in_j = find_neighbor(node_j, i)`
- `Ai = mps_i[idx_j_in_i]` shape `(da_l, d, da_r)`;
  `Aj = mps_j[idx_i_in_j]` shape `(db_l, d, db_r)`
- Form `mati = reshape(permutedims(Ai,(1,3,2)), da_l*da_r, d)`,
        `matj = reshape(permutedims(Aj,(1,3,2)), db_l*db_r, d)`
- `merged = mati * transpose(matj)`, `tsvd` → keep `min(nnz, Dmax)` singular values
- Split `mati = U * Diagonal(sqrt.(S))`,
        `matj = conj(V) * Diagonal(sqrt.(S))` (so `mati * transpose(matj) = U S V'`)
- Reshape back and store.
"""
function cut_bondim!(tn::TensorNetwork, i::Int, idx_j_in_i::Int)
    tn.Dmax < 0 && return 0.0

    node_i = tn.tensors[i]
    j = node_i.neighbor[idx_j_in_i]
    j <= 0 && error("cut_bondim!: leg $idx_j_in_i of node $i is an open leg (no neighbor)")

    node_j = tn.tensors[j]
    idx_i_in_j = find_neighbor(node_j, i)
    idx_i_in_j == 0 && error("cut_bondim!: node $i not found as neighbor of node $j")

    # Extract the physical-bond tensors
    Ai = node_i.mps[idx_j_in_i]   # (da_l, d, da_r)
    Aj = node_j.mps[idx_i_in_j]   # (db_l, d, db_r)

    da_l, d, da_r = size(Ai)
    db_l, _d, db_r = size(Aj)

    # Reshape: merge virtual bonds, physical bond last
    mati = reshape(permutedims(Ai, (1, 3, 2)), da_l * da_r, d)   # (da_l*da_r, d)
    matj = reshape(permutedims(Aj, (1, 3, 2)), db_l * db_r, d)   # (db_l*db_r, d)

    # Merge and SVD
    merged = mati * transpose(matj)   # (da_l*da_r, db_l*db_r)
    U, S, V, discarded_base = tsvd(merged; cutoff=tn.cutoff)

    # Apply Dmax truncation on top of cutoff truncation
    myd = min(length(S), tn.Dmax)
    myd == 0 && (myd = 1)

    extra_discarded = myd < length(S) ? sum(S[myd+1:end]) : 0.0
    error = discarded_base + extra_discarded

    S = S[1:myd]
    U = U[:, 1:myd]
    V = V[:, 1:myd]

    sqS = sqrt.(S)
    mati = U * Diagonal(sqS)         # (da_l*da_r, myd)
    matj = conj(V) * Diagonal(sqS)   # (db_l*db_r, myd)

    # Reshape back: (rows, myd) → (left_bond, right_bond, myd) → permute → (left_bond, myd, right_bond)
    node_i.mps[idx_j_in_i] = permutedims(reshape(mati, da_l, da_r, myd), (1, 3, 2))
    node_j.mps[idx_i_in_j] = permutedims(reshape(matj, db_l, db_r, myd), (1, 3, 2))

    return error
end

"""
    cut_bondim_opt!(tn, i, idx_j_in_i) -> Float64

Like `cut_bondim!` but first canonicalizes both nodes to the shared leg (via
`cano_to!`) and then applies QR pre-reduction before SVD to reduce the SVD
matrix size.  Mirrors `tn_np.py:598`.

When `tn.Dmax < 0` the function is a no-op and returns `0.0`.
"""
function cut_bondim_opt!(tn::TensorNetwork, i::Int, idx_j_in_i::Int)
    tn.Dmax < 0 && return 0.0

    node_i = tn.tensors[i]
    j = node_i.neighbor[idx_j_in_i]
    j <= 0 && error("cut_bondim_opt!: leg $idx_j_in_i of node $i is an open leg (no neighbor)")

    node_j = tn.tensors[j]
    idx_i_in_j = find_neighbor(node_j, i)
    idx_i_in_j == 0 && error("cut_bondim_opt!: node $i not found as neighbor of node $j")

    # Canonicalize both nodes to the shared leg
    cano_to!(node_i, idx_j_in_i)
    cano_to!(node_j, idx_i_in_j)

    # Extract the physical-bond tensors
    Ai = node_i.mps[idx_j_in_i]   # (da_l, d, da_r)
    Aj = node_j.mps[idx_i_in_j]   # (db_l, d, db_r)

    da_l, d, da_r = size(Ai)
    db_l, _d, db_r = size(Aj)

    # Reshape: merge virtual bonds, physical bond last
    mati = reshape(permutedims(Ai, (1, 3, 2)), da_l * da_r, d)   # (da_l*da_r, d)
    matj = reshape(permutedims(Aj, (1, 3, 2)), db_l * db_r, d)   # (db_l*db_r, d)

    # QR pre-reduction to reduce SVD cost (mirrors tn_np.py:632-644)
    ET = eltype(mati)
    flag_left = false
    qi = nothing
    if size(mati, 1) > size(mati, 2)
        F = qr(mati)
        qi = _thin_q(F.Q, mati, ET, size(mati, 2))
        ri = copy(F.R)
        flag_left = true
    else
        ri = mati
    end

    flag_right = false
    qj = nothing
    if size(matj, 1) > size(matj, 2)
        F = qr(matj)
        qj = _thin_q(F.Q, matj, ET, size(matj, 2))
        rj = copy(F.R)
        flag_right = true
    else
        rj = matj
    end

    # Merge and SVD
    merged = ri * transpose(rj)
    U, S, V, discarded_base = tsvd(merged; cutoff=tn.cutoff)

    # Apply Dmax truncation on top of cutoff truncation
    myd = min(length(S), tn.Dmax)
    myd == 0 && (myd = 1)

    extra_discarded = myd < length(S) ? sum(S[myd+1:end]) : 0.0
    error = discarded_base + extra_discarded

    S = S[1:myd]
    U = U[:, 1:myd]
    V = V[:, 1:myd]

    sqS = sqrt.(S)
    mati = U * Diagonal(sqS)         # (size_ri_rows, myd) or (da_l*da_r, myd)
    matj = conj(V) * Diagonal(sqS)   # (size_rj_rows, myd)

    # Re-apply Q factors if QR was used
    flag_left  && (mati = qi * mati)
    flag_right && (matj = qj * matj)

    # Reshape back
    node_i.mps[idx_j_in_i] = permutedims(reshape(mati, da_l, da_r, myd), (1, 3, 2))
    node_j.mps[idx_i_in_j] = permutedims(reshape(matj, db_l, db_r, myd), (1, 3, 2))

    return error
end

# ---------------------------------------------------------------------------
# Network lognorm helper
# ---------------------------------------------------------------------------

"""
    network_lognorm(tn) -> (lognorm, sign)

Sum of `lognorm(node)` over all nodes in `tn.tensors`.
Mirrors `tn_np.py:453`.
"""
function network_lognorm(tn::TensorNetwork)
    T = eltype(first(values(tn.tensors)).mps |> v -> isempty(v) ? [1.0] : v[1])
    lognorm_total = zero(real(T))
    sign_total = one(T)
    for (_, node) in tn.tensors
        ln, sg = lognorm(node)
        lognorm_total += ln
        sign_total *= sg
    end
    return (lognorm_total, sign_total)
end

# ---------------------------------------------------------------------------
# Main contraction loop
# ---------------------------------------------------------------------------

"""
    has_real_edges(tn) -> Bool

Return true iff any node in `tn` has at least one real (non-sentinel) neighbor.
Mirrors `self.G.number_of_edges() > 0` from tn_np.py:302.
"""
function has_real_edges(tn::TensorNetwork)
    for (_, node) in tn.tensors
        for nb in node.neighbor
            nb > 0 && return true
        end
    end
    return false
end

"""
    contraction!(tn) -> (lnZ, error, psi)

Perform the full CATN contraction loop: repeatedly select an edge (i,j), eat
node j into node i, optionally compress, and accumulate the log-partition
function `lnZ`, truncation `error`, and phase `psi`.

Mirrors `tn_np.py:294-451`. `lnZ` is initialised to zero; isolated nodes
(degree-0 spins) are already encoded as scalar `(1,1,1)` MPS sites storing
`2*cosh(β*h_k)`, so their contribution is picked up by `network_lognorm` after
the loop along with all remaining connected-node lognorms.
"""
function contraction!(tn::TensorNetwork{T}) where {T}
    error = 0.0
    tn.psi = one(T)
    # Initial lnZ = 0; isolated node contributions are already encoded in their MPS
    # tensors (each isolated spin k stores 2*cosh(β*h_k) in a (1,1,1) site and is
    # accumulated by network_lognorm after the loop).
    # NOTE: the Python reference (tn_np.py:297) adds log(2)*num_isolated here because
    # it leaves isolated-node MPS empty (valid only for h=0).  This Julia implementation
    # instead populates each isolated node's MPS with the correct scalar value, so the
    # log(2)*num_isolated offset would double-count that contribution.
    tn.lnZ = zero(T)

    # Build edge_count if not yet initialized (mirrors tn_np.py: select is called
    # inside the loop but edge_count must be valid for select=0 or select=1)
    if tn.select in (0, 1) && isempty(tn.edge_count)
        select_edge_init!(tn)
    end

    while has_real_edges(tn)
        # --- Select edge ---
        if tn.select == 0
            i, j = select_edge_min_dim(tn)
        elseif tn.select == 1
            i, j = select_edge_min_dim_triangle(tn)
        else  # select == 2
            i, j = select_edge_sequentially(tn)
        end

        # Ensure order(i) >= order(j) (mirrors tn_np.py:313-314)
        node_i = tn.tensors[i]
        node_j = tn.tensors[j]
        if order(node_j) > order(node_i)
            i, j = j, i
            node_i, node_j = node_j, node_i
        end

        # --- Bookkeeping: remove affected edges BEFORE any shape mutation ---
        # Mirrors tn_np.py:320 ("take care of the count dictionary first")
        affected = vcat([i, j], collect(node_i.neighbor), collect(node_j.neighbor))
        if tn.select in (0, 1)
            count_remove_nodes!(tn, affected)
        end

        # --- Find connection indices ---
        neigh_i = copy(node_i.neighbor)   # snapshot before any mutation
        neigh_j = copy(node_j.neighbor)   # snapshot before any mutation
        idx_j_in_i = find_neighbor(node_i, j)
        idx_i_in_j = find_neighbor(node_j, i)

        # --- Optional reverse to minimize swap cost ---
        # Mirrors tn_np.py:327-341 (but our swap!/reverse! keep neighbor in sync,
        # so we just recompute the indices after reverse!)
        if tn.reverse
            if idx_j_in_i < cld(length(node_i.neighbor), 2)   # idx_j_in_i < len//2
                reverse!(node_i)
                neigh_i = copy(node_i.neighbor)
                idx_j_in_i = find_neighbor(node_i, j)
            end
            if idx_i_in_j >= cld(length(node_j.neighbor), 2)   # idx_i_in_j >= len//2
                reverse!(node_j)
                neigh_j = copy(node_j.neighbor)
                idx_i_in_j = find_neighbor(node_j, i)
            end
        end

        # --- Re-point j's other neighbors to i, detecting duplicates ---
        # Mirrors tn_np.py:344-362.
        # NOTE: We do NOT touch node_i.neighbor here; eat! handles it internally
        # (removes contracted leg from tail, appends nodej's remaining neighbors).
        # We only update node_k.neighbor (replace j with i) and track duplicates.

        duplicate = Int[]
        for l in 1:length(neigh_j)
            l == idx_i_in_j && continue   # skip the i→j connection
            k = neigh_j[l]
            k <= 0 && continue            # skip open-leg sentinels

            # Check if k is already a neighbor of i (against ORIGINAL node_i.neighbor,
            # before eat! appends j's others) — 0 if not present
            idx_k_in_i = find_neighbor(node_i, k)

            # Update node_k: replace j with i at the same position
            node_k = tn.tensors[k]
            idx_i_in_k = find_neighbor(node_k, i)   # 0 if i not yet in k (before deletion)
            idx_j_in_k = delete_neighbor!(node_k, j) # returns former 1-based position of j
            add_neighbor!(node_k, i, idx_j_in_k)     # insert i at that position

            if idx_k_in_i > 0   # k was already a neighbor of i → duplicate
                push!(duplicate, k)
                # cross = true when idx_i_in_k > idx_j_in_k (mirrors tn_np.py:360)
                cross = idx_i_in_k > idx_j_in_k
                err = merge!(node_k, i; cross=cross)
                error += err
            end
        end

        # --- eat! j into i ---
        # mirrors tn_np.py:367-372
        lognorm_val, err, phase = eat!(node_i, node_j, idx_j_in_i, idx_i_in_j)
        error += err
        tn.psi *= phase
        tn.lnZ += lognorm_val

        # --- Per-duplicate: merge into i, then cut bond if needed ---
        # Mirrors tn_np.py:374-395
        for k in duplicate
            merge!(node_i, k; cross=false)
            idx_k_in_i = find_neighbor(node_i, k)
            if tn.svdopt
                if tn.cut_bond
                    error += cut_bondim_opt!(tn, i, idx_k_in_i)
                else
                    if tn.Dmax > 0 && size(node_i.mps[idx_k_in_i], 2) > tn.Dmax
                        error += cut_bondim_opt!(tn, i, idx_k_in_i)
                    end
                end
            else
                if tn.cut_bond
                    error += cut_bondim_opt!(tn, i, idx_k_in_i)
                else
                    if tn.Dmax > 0 && size(node_i.mps[idx_k_in_i], 2) > tn.Dmax
                        error += cut_bondim_opt!(tn, i, idx_k_in_i)
                    end
                end
            end
        end

        # --- Clear j ---
        clear!(node_j)
        # (In Python: self.G.remove_node(j) — we keep node j in tn.tensors but empty it)

        # --- Optional compress ---
        if tn.compress
            if tn.svdopt
                compress_opt!(node_i)
            else
                compress!(node_i)
            end
        end

        # --- Update bookkeeping ---
        if tn.select in (0, 1)
            count_add_nodes!(tn, vcat([i], collect(node_i.neighbor)))
        end

        # --- Track maxdim_intermediate ---
        for t in node_i.mps
            d = size(t, 2)
            if d > tn.maxdim_intermediate
                tn.maxdim_intermediate = d
            end
        end
    end

    # --- After loop: accumulate remaining lognorms ---
    # Mirrors tn_np.py:449-450
    ln, sg = network_lognorm(tn)
    tn.sign = sg
    tn.lnZ += ln

    return (tn.lnZ, error, tn.psi)
end
