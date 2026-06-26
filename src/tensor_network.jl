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
