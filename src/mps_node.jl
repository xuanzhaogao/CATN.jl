mutable struct MPSNode{T}
    mps::Vector{Array{T,3}}
    neighbor::Vector{Int}
    cano::Int
    chi::Int
    cutoff::Float64
    norm_method::Int
    svdopt::Bool
    swapopt::Bool
end

function MPSNode(tensor::AbstractArray{T}, neighbor::Vector{Int};
                chi::Int=32, cutoff::Float64=1e-15, norm_method::Int=1,
                svdopt::Bool=true, swapopt::Bool=true) where {T}
    mps = raw2mps(tensor, chi, cutoff)
    cano = length(mps)          # left-canonical: center at last site
    MPSNode{T}(mps, copy(neighbor), cano, chi, cutoff, norm_method, svdopt, swapopt)
end

function raw2mps(tensor::AbstractArray{T}, chi::Int, cutoff::Float64) where {T}
    nd = ndims(tensor)
    nd == 0 && return Array{T,3}[]
    dims = size(tensor)
    nd == 1 && return Array{T,3}[reshape(Array(tensor), 1, dims[1], 1)]
    mps = Array{T,3}[]
    R = reshape(Array(tensor), 1, dims...)      # (1, dims...)
    dleft = 1
    for i in 1:nd-1
        M = reshape(R, dleft * dims[i], :)
        U, S, V = tsvd(M; cutoff=cutoff, maxdim=chi)
        χ = length(S)
        push!(mps, reshape(U, dleft, dims[i], χ))
        R = reshape(Diagonal(S) * V', χ, dims[(i+1):end]...)
        dleft = χ
    end
    push!(mps, reshape(R, dleft, dims[nd], 1))
    return mps
end

function mps2raw(node::MPSNode{T}) where {T}
    mps = node.mps
    isempty(mps) && return Array{T,0}(undef)
    A = mps[1]
    cur = reshape(A, size(A, 2), size(A, 3))    # (p1, r) since left bond = 1
    pdim = size(A, 2)
    physdims = Int[size(A, 2)]
    for k in 2:length(mps)
        B = mps[k]
        r, p, r2 = size(B)
        Bmat = reshape(B, r, p * r2)
        res = ein"ab,bc->ac"(cur, Bmat)         # OMEinsum backend
        res3 = reshape(res, pdim, p, r2)
        pdim *= p
        cur = reshape(res3, pdim, r2)
        push!(physdims, p)
    end
    return reshape(cur, physdims...)            # right bond = 1 collapses
end

function cano_to!(node::MPSNode, idx::Int)
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

left_canonical!(node::MPSNode) = (node.cano = 1; cano_to!(node, length(node.mps)))

order(node::MPSNode) = length(node.mps)

function shape(node::MPSNode)
    return [size(t, 2) for t in node.mps]
end

"""
    swap!(node, i, j) -> Float64

Swap adjacent MPS sites `i` and `j` (`|i-j| == 1`). Truncates the virtual bond
to `node.chi`, keeps the canonical center at `j`, and also swaps the two
corresponding entries of `node.neighbor` so that `mps2raw` leg order and the
neighbor list stay aligned.  Returns the truncation error (sum of discarded
singular values).

If the canonical center is not at `i` or `j` it is first moved to whichever
of the two is closest.
"""
function swap!(node::MPSNode, i::Int, j::Int)
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

    # Build swapped 4-tensor: physical legs are exchanged via "ijk,kab->iajb"
    W = ein"ijk,kab->iajb"(tl, tr)    # (d0, d1, d2, d3) — column-major
    M = reshape(W, d0 * d1, d2 * d3)

    rows, cols = size(M)
    U, S, V, err = if node.swapopt && ((rows > 7000 && cols > 7000) || rows > 20000 || cols > 20000)
        U2, S2, V2 = rsvd(M, node.chi, 10, 10)
        U2, S2, V2, 0.0   # rsvd is approximate; tail weight not tracked
    else
        tsvd(M; cutoff=node.cutoff, maxdim=node.chi)
    end

    myd = length(S)
    error = err

    if i < j  # going right: center ends at j
        node.mps[i] = reshape(U, d0, d1, myd)
        node.mps[j] = reshape(Diagonal(S) * V', myd, d2, d3)
    else  # going left: fold S into lower-index site (j < i); center ends at j
        node.mps[j] = reshape(U * Diagonal(S), d0, d1, myd)
        node.mps[i] = reshape(V', myd, d2, d3)
    end

    node.cano = j

    # Rotate the two neighbor entries in lockstep with the two MPS sites
    node.neighbor[i], node.neighbor[j] = node.neighbor[j], node.neighbor[i]

    return error
end

"""
    move!(node, a, b) -> Float64

Move site `a` to position `b` by chaining `swap!` calls. Returns accumulated
truncation error.
"""
function move!(node::MPSNode, a::Int, b::Int)
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
    move2tail!(node, idx) -> Float64

Move site `idx` to the last position by chaining `swap!` calls. Returns
accumulated truncation error. Both `node.mps` and `node.neighbor` are kept
consistent.
"""
function move2tail!(node::MPSNode, idx::Int)
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
    move2head!(node, idx) -> Float64

Move site `idx` to the first position by chaining `swap!` calls. Returns
accumulated truncation error.
"""
function move2head!(node::MPSNode, idx::Int)
    error = move!(node, idx, 1)
    cano_to!(node, 1)
    return error
end

"""
    reverse!(node)

Reverse the site order of the MPS in-place. Transposes each tensor so that
left/right virtual bonds are exchanged: `node.mps[k] -> permutedims(t, (3,2,1))`
on the reversed list. Also reverses `node.neighbor` and updates `node.cano`.
"""
function reverse!(node::MPSNode)
    length(node.mps) <= 1 && return node
    node.mps = [permutedims(t, (3, 2, 1)) for t in Base.reverse(node.mps)]
    Base.reverse!(node.neighbor)
    node.cano = length(node.mps) + 1 - node.cano
    return node
end

# ---------------------------------------------------------------------------
# Neighbor helpers
# ---------------------------------------------------------------------------

"""
    find_neighbor(node, j) -> Int

Return the 1-based position of neighbor `j` in `node.neighbor`, or `0` if not
found (the Python reference returns -1 for the 0-based "not found" case; here
0 is the 1-based sentinel).
"""
function find_neighbor(node::MPSNode, j::Int)
    idx = findfirst(==(j), node.neighbor)
    return idx === nothing ? 0 : idx
end

"""
    add_neighbor!(node, n, pos=0)

Insert neighbor `n` at 1-based position `pos` (default 0 means append).
"""
function add_neighbor!(node::MPSNode, n::Int, pos::Int=0)
    if pos == 0
        push!(node.neighbor, n)
    else
        insert!(node.neighbor, pos, n)
    end
    return node
end

"""
    delete_neighbor!(node, n) -> Int

Remove neighbor `n` from `node.neighbor` and return its former 1-based
position.
"""
function delete_neighbor!(node::MPSNode, n::Int)
    idx = find_neighbor(node, n)
    idx == 0 && error("delete_neighbor!: neighbor $n not found")
    deleteat!(node.neighbor, idx)
    return idx
end

# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

"""
    logdim(node) -> Float64

Sum of log₂(physical dimension) over all sites — equivalently, log₂ of the
total number of elements of the raw tensor.
"""
function logdim(node::MPSNode)
    isempty(node.mps) && return 0.0
    return sum(log2(Float64(size(t, 2))) for t in node.mps)
end

"""
    logdim(node, idx) -> Float64

log₂ of the physical dimension of site `idx`.
"""
function logdim(node::MPSNode, idx::Int)
    return log2(Float64(size(node.mps[idx], 2)))
end

"""
    lognorm(node) -> (Float64, sign)

For a scalar MPS (single site with physical dimension 1) return `(log|z|, sign(z))`.
Calling this on a non-scalar MPS raises an error as in the reference.
"""
function lognorm(node::MPSNode)
    isempty(node.mps) && return (0.0, 1)
    if length(node.mps) == 1 && size(node.mps[1], 2) == 1
        z = node.mps[1][1, 1, 1]
        return (log(abs(z)), sign(z))
    end
    error("lognorm: computing norm of a non-scalar MPS is not supported in contraction")
end

"""
    clear!(node)

Remove all MPS tensors and neighbor entries.
"""
function clear!(node::MPSNode)
    empty!(node.mps)
    empty!(node.neighbor)
    return node
end
