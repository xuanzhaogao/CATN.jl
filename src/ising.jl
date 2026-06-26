"""
    ising_network(n, edges, weights, fields, β; Dmax=32, chi=32, kwargs...)

Build a TensorNetwork for the Ising/spin-glass partition function directly in MPS form.

Each spin i becomes a COPY/δ tensor built as a chain of MPS rank-3 sites.
Each edge (i,j) contributes a bond factor B = exp.(β .* w .* M_ij) split Q=B, R=I(2).
Field terms exp.(β .* h_i .* spin) fold into the first site (leaf or first-neighbor case).

Mirrors `construct_tensor` in tn_np.py (lines 71-167).

# Arguments
- `n`: number of spins
- `edges`: Vector of (i,j) tuples (1-based)
- `weights`: scalar coupling for each edge
- `fields`: external field h_i for each spin
- `β`: inverse temperature
- `Dmax`, `chi`, `kwargs...`: passed to contraction parameters
"""
function ising_network(
        n::Int,
        edges::Vector{<:Tuple{Int,Int}},
        weights::AbstractVector,
        fields::AbstractVector,
        β::Real;
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
        seed::Int=1)

    # Compute degree of each node
    degree = zeros(Int, n)
    for (i, j) in edges
        degree[i] += 1
        degree[j] += 1
    end

    # Initialize empty MPSNode for each spin (no mps sites yet, empty neighbor list)
    # We build mps directly, so we create nodes with empty mps and populate them
    nodes = Dict{Int, MPSNode{Float64}}()
    for k in 1:n
        node = MPSNode{Float64}(
            Array{Float64,3}[],  # mps: empty, will be populated
            Int[],               # neighbor: empty, will be populated
            0,                   # cano
            chi,
            cutoff,
            norm_method,
            svdopt,
            swapopt
        )
        nodes[k] = node
    end

    spin = [1.0, -1.0]  # spin values

    for (edge_idx, (i, j)) in enumerate(edges)
        spini = copy(spin)
        spinj = copy(spin)

        # Outer product M_ij = spini ⊗ spinj
        M_ij = spini * spinj'  # 2x2 matrix

        # Bond factor B = exp(β * w * M_ij), Q = B, R = I
        w = float(weights[edge_idx])
        B = exp.(β * w .* M_ij)  # 2x2
        Q = B
        R = Matrix{Float64}(I, 2, 2)

        # --- Process node i ---
        nodei = nodes[i]
        hi = float(fields[i])
        fieldi = Diagonal(exp.(β .* hi .* spini))  # 2x2 diagonal

        if degree[i] == 1
            # Leaf: sum over spin_i → scalar bond factors
            # mat = fieldi @ Q, then sum over rows (spin_i), shape (1, 2, 1)
            mat = Matrix(fieldi) * Q  # 2x2
            # sum over physical index (rows = spin_i): result shape (1, 2, 1) after sum axis 0 in numpy
            # In numpy: mat.sum(0) sums rows → result is shape (2,), then reshape to (1,2,1)
            # This means we sum over spin_i index, leaving only spin_j (the bond index)
            sv = dropdims(sum(mat, dims=1), dims=1)  # shape (2,)
            site = reshape(sv, 1, 2, 1)
            push!(nodei.mps, site)
        else
            if isempty(nodei.neighbor)
                # First neighbor: shape (1, 2, 2)
                # mat = (fieldi @ Q).T, reshape to (1, 2, 2)
                # In numpy C-order: .T transposes the 2x2, then reshape(1,2,2) gives
                # site[0, s_i, s_j] = (fieldi@Q).T[s_i, s_j] = (fieldi@Q)[s_j, s_i]
                mat = (Matrix(fieldi) * Q)'  # 2x2, transposed
                # reshape to (1, 2, 2) in C-order: mat is indexed [s_j, s_i] after reshape
                # In numpy: A.T.reshape(1,2,2) means site[0,r,c] = A.T[r,c] = A[c,r]
                # In Julia (Fortran/column-major): we need to be explicit
                # site has shape (left=1, phys=2, right=2)
                # site[1, s_i+1, s_j+1] should equal mat[s_i+1, s_j+1]
                site = reshape(mat, 1, 2, 2)
                push!(nodei.mps, site)
            elseif length(nodei.neighbor) == degree[i] - 1
                # Last neighbor: shape (2, 2, 1)
                # mat = Q, reshape to (2, 2, 1)
                # In numpy: Q.reshape(2, 2, 1) — Q[s_i, s_j] stored in C order
                # site[s_i, s_j, 0] = Q[s_i, s_j]
                # In Julia: reshape fills column-major, so we need to be careful
                # We want site[a, s, b] where a=left_bond, s=phys, b=right_bond
                # For last site: a = s_i_prev_bond (2), s = s_i (2), b = 1
                # site[s_i, s_j, 1] = Q[s_i, s_j]
                # reshape(Q, 2, 2, 1) in Julia fills column-major: site[1,1,1]=Q[1,1], site[2,1,1]=Q[2,1]...
                # That gives site[a, s, b] = Q[a, s] — but we want site[a, s, 1] = Q[a, s]
                # Q has shape (2, 2): Q[s_i_old, s_i_new_bond]
                # This should be site[s_i_old=a, s_i_new_bond=s, 1]
                # So yes: reshape(Q, 2, 2, 1) in Julia gives the right layout
                site = reshape(copy(Q), 2, 2, 1)
                push!(nodei.mps, site)
            else
                # Interior: shape (2, 2, 2)
                # t3[:, 0, :] = diag(Q[:, 0])  (Python 0-indexed → Julia column 1)
                # t3[:, 1, :] = diag(Q[:, 1])  (Python 1-indexed → Julia column 2)
                # In Julia: t3[a, s, b] where a=left_bond, s=phys, b=right_bond
                t3 = zeros(Float64, 2, 2, 2)
                t3[:, 1, :] = Diagonal(Q[:, 1])
                t3[:, 2, :] = Diagonal(Q[:, 2])
                push!(nodei.mps, t3)
            end
        end
        push!(nodei.neighbor, j)

        # --- Process node j ---
        nodej = nodes[j]
        hj = float(fields[j])
        fieldj = Diagonal(exp.(β .* hj .* spinj))  # 2x2 diagonal

        if degree[j] == 1
            # Leaf: sum over spin_j
            mat = Matrix(fieldj) * R  # fieldj @ I = fieldj (2x2 diagonal)
            sv = dropdims(sum(mat, dims=1), dims=1)
            site = reshape(sv, 1, 2, 1)
            push!(nodej.mps, site)
        else
            if isempty(nodej.neighbor)
                # First neighbor
                mat = (Matrix(fieldj) * R)'  # (fieldj @ R).T
                site = reshape(mat, 1, 2, 2)
                push!(nodej.mps, site)
            elseif length(nodej.neighbor) == degree[j] - 1
                # Last neighbor
                site = reshape(copy(R), 2, 2, 1)
                push!(nodej.mps, site)
            else
                # Interior
                t3 = zeros(Float64, 2, 2, 2)
                t3[:, 1, :] = Diagonal(R[:, 1])
                t3[:, 2, :] = Diagonal(R[:, 2])
                push!(nodej.mps, t3)
            end
        end
        push!(nodej.neighbor, i)
    end

    # Handle isolated nodes (degree 0) — spin contributes exp(β*h*s) summed over s
    for k in 1:n
        if degree[k] == 0
            hk = float(fields[k])
            # Z_k = exp(β*h_k*1) + exp(β*h_k*(-1)) = 2*cosh(β*h_k)
            val = exp(β * hk) + exp(-β * hk)
            site = reshape([val], 1, 1, 1)
            push!(nodes[k].mps, site)
            # No neighbor to add
        end
    end

    # Set cano to length(mps) for all nodes (left-canonical convention, mirrors MPSNode constructor)
    for (_, node) in nodes
        node.cano = length(node.mps)
    end

    # Count isolated nodes (no real bonds)
    num_isolated = count(k -> isempty(nodes[k].neighbor), 1:n)

    TensorNetwork{Float64}(
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
        Dict{Int,Vector{Vector{Int}}}(),
        0.0,   # lnZ
        1.0,   # sign
        1.0,   # psi
        -1,    # maxdim_intermediate
        num_isolated,
        MersenneTwister(seed),
        n,
        Float64(β)
    )
end

"""
    free_energy(tn) -> (lnZ_per_site, F)

Run contraction and return per-site log-partition function and free energy.

- `lnZ_per_site = lnZ / n`
- `F = -lnZ_per_site / β`

Mirrors lnz_np.py lines 171-175.
"""
function free_energy(tn::TensorNetwork)
    lnZ, _, _ = contraction!(tn)
    lnZ_per_site = lnZ / tn.n
    F = -lnZ_per_site / tn.beta
    return (lnZ_per_site, F)
end
