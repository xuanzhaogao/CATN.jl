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
