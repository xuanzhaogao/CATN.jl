using LinearAlgebra
using OMEinsum, Test

function exact_contract(tensors, ixs)
    # output = labels appearing exactly once, first-seen order
    ixs_vv = [collect(ix) for ix in ixs]
    LT = eltype(eltype(ixs_vv))
    counts = Dict{LT,Int}()
    order = LT[]
    for ix in ixs_vv, l in ix
        haskey(counts, l) || push!(order, l)
        counts[l] = get(counts, l, 0) + 1
    end
    iy = [l for l in order if counts[l] == 1]
    code = EinCode(ixs_vv, iy)
    sd = OMEinsum.get_size_dict(ixs_vv, tensors)
    opt = optimize_code(code, sd, GreedyMethod())
    return opt(tensors...)
end

@testset "exact oracle" begin
    A = randn(2, 3); B = randn(3, 4); C = randn(4, 2)
    # trace(A*B*C): labels a,b,c each appear twice -> scalar
    r = exact_contract([A, B, C], [[:a,:b],[:b,:c],[:c,:a]])
    @test r[] ≈ tr(A * B * C)
end
