# Device movement via Adapt: `adapt(CuArray, tn)` / `cu(tn)` move every MPS tensor
# to the target array type. Compute then dispatches to OMEinsum's CUDAExt + cuSOLVER.
# Note: a network containing a 0-dim-scalar (empty-mps) node cannot change device
# (its AT is unknowable from no arrays); this does not arise in Ising or normal use.

function Adapt.adapt_structure(to, node::MPSNode)
    new_mps = map(t -> adapt(to, t), node.mps)
    MPSNode(new_mps, copy(node.neighbor), node.cano, node.chi,
            node.cutoff, node.norm_method, node.svdopt, node.swapopt)
end

function Adapt.adapt_structure(to, tn::TensorNetwork)
    ks = sort(collect(keys(tn.tensors)))
    new_nodes = [adapt(to, tn.tensors[k]) for k in ks]
    @assert !isempty(new_nodes) "adapt_structure: cannot adapt an empty TensorNetwork"
    NAT = eltype(new_nodes[1].mps)                       # AT after adapt
    NT  = eltype(NAT)                                    # element type AFTER the move
    tensors = Dict{Int,MPSNode{NT,NAT}}(k => n for (k, n) in zip(ks, new_nodes))
    TensorNetwork{NT,NAT}(
        tensors, tn.Dmax, tn.chi, tn.cutoff, tn.norm_method, tn.select, tn.reverse,
        tn.svdopt, tn.swapopt, tn.compress, tn.cut_bond,
        deepcopy(tn.edge_count), NT(tn.lnZ), NT(tn.sign), NT(tn.psi),
        tn.maxdim_intermediate, tn.num_isolated, deepcopy(tn.rng), tn.n, tn.beta)
end
