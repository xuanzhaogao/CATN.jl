using CATN
using Test

@testset "CATN.jl" begin
    include("exact.jl")
    include("test_linalg.jl")
    include("test_mps_node.jl")
    include("test_contraction.jl")
    include("test_ising.jl")
    include("test_adapt.jl")
    include("test_gpu.jl")
    include("test_complex.jl")
end
