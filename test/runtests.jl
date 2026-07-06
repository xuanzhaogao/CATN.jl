using CATN
using Test

@testset "CATN.jl" begin
    include("exact.jl")
    include("test_linalg.jl")
    include("test_braket.jl")
    include("test_mps_node.jl")
    include("test_contraction.jl")
    include("test_openleg.jl")
    include("test_ising.jl")
    include("test_adapt.jl")
    include("test_complex.jl")
    # GPU tests run only when CUDA is available (e.g. the `test/gpu` environment,
    # see test/gpu/run_gpu_tests.jl). CUDA is not a dependency of the default test
    # environment, so this is skipped on CPU-only machines and in CI.
    if Base.find_package("CUDA") !== nothing
        include("test_gpu.jl")
    else
        @info "CUDA not in environment; skipping GPU tests (run test/gpu/run_gpu_tests.jl on a GPU machine)"
    end
end
