mutable struct TensorNetwork{T}
    tensors::Dict{Int,MPSNode{T}}
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
    nodes = Dict{Int,MPSNode{T}}()
    for t in 1:n
        nodes[t] = MPSNode(tensors[t], neighbor_vecs[t];
                           chi=chi, cutoff=cutoff,
                           norm_method=norm_method,
                           svdopt=svdopt, swapopt=swapopt)
    end

    # Count isolated nodes: nodes with no real (non-sentinel) bond
    # A node is isolated if all its neighbor ids are negative (sentinels) or it has 0 legs
    num_isolated = count(t -> all(nb -> nb < 0, neighbor_vecs[t]), 1:n)

    TensorNetwork{T}(
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
        MersenneTwister(seed)
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
