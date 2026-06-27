mutable struct MPSNode{T,AT<:AbstractArray{T,3}}
    mps::Vector{AT}
    neighbor::Vector{Int}
    cano::Int
    chi::Int
    cutoff::Float64
    norm_method::Int
    svdopt::Bool
    swapopt::Bool
    # Suppress the auto-generated outer constructor to avoid method-overwrite warning
    # at precompilation when we define the outer positional constructor below.
    MPSNode{T,AT}(mps, neighbor, cano, chi, cutoff, norm_method, svdopt, swapopt) where {T, AT<:AbstractArray{T,3}} =
        new{T,AT}(mps, neighbor, cano, chi, cutoff, norm_method, svdopt, swapopt)
end

# Positional constructor: infer the array-type parameter AT from the mps vector.
function MPSNode(mps::Vector{<:AbstractArray{T,3}}, neighbor::Vector{Int}, cano::Int,
                 chi::Int, cutoff::Float64, norm_method::Int, svdopt::Bool,
                 swapopt::Bool) where {T}
    MPSNode{T,eltype(mps)}(mps, neighbor, cano, chi, cutoff, norm_method, svdopt, swapopt)
end

function MPSNode(tensor::AbstractArray{T}, neighbor::Vector{Int};
                chi::Int=32, cutoff::Float64=1e-15, norm_method::Int=1,
                svdopt::Bool=true, swapopt::Bool=true) where {T}
    mps = raw2mps(tensor, chi, cutoff)
    cano = length(mps)          # left-canonical: center at last site
    MPSNode(mps, copy(neighbor), cano, chi, cutoff, norm_method, svdopt, swapopt)
end

function raw2mps(tensor::AbstractArray{T}, chi::Int, cutoff::Float64) where {T}
    nd = ndims(tensor)
    AT = typeof(similar(tensor, T, (1, 1, 1)))   # concrete 3-D array type on tensor's device
    nd == 0 && return AT[]
    dims = size(tensor)
    nd == 1 && return AT[reshape(tensor, 1, dims[1], 1)]
    mps = AT[]
    R = reshape(tensor, 1, dims...)              # preserves device (no Array() copy)
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
        node.mps[i] = reshape(copy(V'), myd, d2, d3)
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
        # sum() performs a GPU-safe reduction that returns a plain Julia scalar,
        # avoiding scalar indexing (node.mps[1][1,1,1] would fail on CuArray).
        z = sum(node.mps[1])
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

"""
    compress!(node) -> Float64

Compress the whole MPS by first left-canonicalizing (moving `cano` to the last
site), then sweeping right-to-left with two-site SVD truncation to `node.chi`.

For each step `j` from `length(mps)` down to `2` (with `i = j-1`):
1. Fuse sites `i` and `j` via `ein"ijk,kab->ijab"` reshaped to `(d0*d1, d2*d3)`.
2. Truncated SVD to `node.chi`.
3. Write back `mps[i] = reshape(U*Diagonal(S), d0, d1, :)`,
   `mps[j] = reshape(V', :, d2, d3)`.

Sets `cano = 1` and returns the accumulated truncation error (sum of discarded
singular values over all sweeps).
"""
function compress!(node::MPSNode)
    error = 0.0
    isempty(node.mps) && return error
    left_canonical!(node)   # cano now at last site (right end)
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
    compress_opt!(node) -> Float64

Like `compress!` but uses QR decomposition before the SVD to reduce the size of
the matrix passed to SVD (mirrors `mps_node_np.py:202-283`).

- `flag_left`:  when `matl` (shape `(d0*d1, dd)`) has more rows than columns,
  QR-decompose `matl` to get a thin `Ql` and `Rl`; otherwise use `Rl = matl`.
- `flag_right`: when `matr` (shape `(dd, d2*d3)`) has fewer rows than columns,
  QR-decompose `matr'` to get `Qr` and `Rr`; otherwise use `Rr = matr'`.
- SVD is performed on `Rl * Rr'`.
- After truncation, `U` is expanded back via `Ql` (if `flag_left`) and `V` via
  `Qr` (if `flag_right`).

Returns the accumulated truncation error.
"""
function compress_opt!(node::MPSNode)
    error = 0.0
    isempty(node.mps) && return error
    left_canonical!(node)   # cano now at last site (right end)
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

"""
    eat!(node, nodej, idx, idxi) -> (lognorm::Float64, error::Float64, phase)

Contract physical leg `idx` of `node` with leg `idxi` of `nodej`, appending all
remaining sites of `nodej` to `node`.  Returns `(lognorm, error, phase)` where
`lognorm` is the accumulated log-norm extracted during normalization, `error` is
the accumulated truncation error from bond-dimension compression during swaps,
and `phase` is the sign/phase of the result.

Three cases (mirroring `mps_node_np.py:407-487`):
- (a) Both nodes are single-site leaves → scalar dot product, `node.mps` becomes empty.
- (b) `nodej` is a single-site leaf, `node` is not → fold into node's tail-1 site.
- (c) General → move contracting sites to boundaries, contract, append, re-canonicalize.
"""
function eat!(node::MPSNode{T}, nodej::MPSNode{T}, idx::Int, idxi::Int) where {T}
    error = 0.0

    # -----------------------------------------------------------------------
    # Case (a): both nodes are single-site leaves
    # -----------------------------------------------------------------------
    if length(node.mps) == 1
        # By convention (mirrors Python assert), nodej must also be a leaf
        @assert length(nodej.mps) == 1
        # mps[1] has shape (1, d, 1); contract along the physical leg
        vi = vec(node.mps[1])    # length d
        vj = vec(nodej.mps[1])   # length d
        r  = dot(vi, vj)
        absr = abs(r)
        if absr <= node.cutoff
            empty!(node.mps)
            # remove contracted leg from node.neighbor
            deleteat!(node.neighbor, idx)
            return (0.0, 0.0, one(T))
        end
        lognorm_val = log(absr)
        empty!(node.mps)
        # Remove contracted neighbors from both
        deleteat!(node.neighbor, idx)
        # nodej's remaining neighbors (after removing idxi) — none for single site
        return (lognorm_val, 0.0, r / absr)
    end

    # move contracting site of node to tail; mati = (D_left, d_phys)
    error += move2tail!(node, idx)
    mati = reshape(node.mps[end], size(node.mps[end], 1), size(node.mps[end], 2))

    # -----------------------------------------------------------------------
    # Case (b): nodej is a single-site leaf, node is not
    # -----------------------------------------------------------------------
    if length(nodej.mps) == 1
        tensorj = nodej.mps[1]
        matj = reshape(tensorj, size(tensorj, 2), 1)   # (d_phys, 1)
        mat  = mati * matj                              # (D_left_i, 1)

        # fold mat into the second-to-last site: "ijk,ka->ija"
        new_tensor = ein"ijk,ka->ija"(node.mps[end-1], mat)

        # normalize
        if node.norm_method == 1
            norm = LinearAlgebra.norm(vec(new_tensor))
        elseif node.norm_method == 2
            norm = maximum(abs, new_tensor)
        else  # norm_method == 0
            norm = one(real(T))
        end

        # Store new_tensor unconditionally (unnormalized), then pop the contracted tail
        node.mps[end-1] = new_tensor
        node.cano = node.cano - 1
        pop!(node.mps)
        # Update neighbors: remove the contracted leg (now at end after move2tail!)
        deleteat!(node.neighbor, length(node.neighbor))

        if norm <= node.cutoff
            return (0.0, error, one(T))
        end

        node.mps[end] ./= norm
        return (log(norm), error, one(T))
    end

    # -----------------------------------------------------------------------
    # Case (c): general — both nodes have multiple sites
    # -----------------------------------------------------------------------
    error += move2head!(nodej, idxi)
    matj = reshape(nodej.mps[1], size(nodej.mps[1], 2), size(nodej.mps[1], 3))  # (d_phys, D_right)

    mat = mati * matj   # (D_left_i, D_right_j)

    # fold mat into second-to-last site of node: "ijk,ka->ija"
    node.mps[end-1] = ein"ijk,ka->ija"(node.mps[end-1], mat)
    pop!(node.mps)

    # Set cano to the folded site (now the last site) BEFORE appending nodej's tail,
    # so cano_to! below performs a real right-sweep across the appended sites.
    node.cano = length(node.mps)

    # append nodej's remaining sites (2:end)
    append!(node.mps, nodej.mps[2:end])

    # Update neighbors: remove contracted leg from node (it was moved to end by move2tail!)
    # and append nodej's remaining neighbors (2:end, since idxi was moved to head)
    deleteat!(node.neighbor, length(node.neighbor))
    append!(node.neighbor, nodej.neighbor[2:end])

    # re-canonicalize to tail: cano points to the folded site (before the appended sites),
    # so this performs a real right-sweep that left-canonicalizes all appended sites.
    cano_to!(node, length(node.mps))

    center = node.mps[node.cano]

    # normalize
    if node.norm_method == 1
        norm = LinearAlgebra.norm(vec(center))
    elseif node.norm_method == 2
        norm = maximum(abs, center)
    else  # norm_method == 0
        norm = one(real(T))
    end

    if norm <= node.cutoff
        return (0.0, error, one(T))
    end

    node.mps[node.cano] = center / norm
    return (log(norm), error, one(T))
end

"""
    merge!(node, j; cross=false) -> Float64

Fuse the two MPS sites whose `neighbor == j` (a duplicate edge that arises
during network contraction) into a single rank-3 site.

Algorithm (mirrors `mps_node_np.py:354-376`):
1. Locate the two positions `idx1 < idx2` where `node.neighbor == j`.
2. Move site `idx2` to position `idx1+1` (cross=false) or `idx1` (cross=true)
   via `move!`.  Because Julia's `swap!` keeps `node.neighbor` synchronised
   with `node.mps`, we delete the duplicate neighbor entry *after* the move,
   at the final landing position of the duplicate site.
3. Canonicalise to `idx1`, fuse `mps[idx1]` and `mps[idx1+1]` via
   `ein"ijk,kab->ijab"` reshaped to `(dl, d*d', dr2)`, drop the extra tensor,
   and canonicalise to `idx1` again.

Returns the accumulated truncation error from `move!`.
"""
function merge!(node::MPSNode, j::Int; cross::Bool=false)
    positions = findall(==(j), node.neighbor)
    length(positions) == 2 || error("merge!: expected exactly 2 sites with neighbor $j, found $(length(positions))")
    idx1 = positions[1]
    idx2 = positions[2]

    # Move idx2 adjacent to idx1 (Julia's move! keeps neighbor in sync with mps)
    target = cross ? idx1 : idx1 + 1
    error = move!(node, idx2, target)

    # After move, the duplicate neighbor entry is at position `target`; remove it
    deleteat!(node.neighbor, target)

    # Canonicalise to idx1, then fuse idx1 and idx1+1
    cano_to!(node, idx1)
    dl  = size(node.mps[idx1], 1)
    dr2 = size(node.mps[idx1 + 1], 3)
    node.mps[idx1] = reshape(
        ein"ijk,kab->ijab"(node.mps[idx1], node.mps[idx1 + 1]),
        dl, :, dr2
    )
    deleteat!(node.mps, idx1 + 1)

    # Final canonicalisation
    cano_to!(node, idx1)

    return error
end
