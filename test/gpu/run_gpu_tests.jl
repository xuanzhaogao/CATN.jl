# Run the GPU test suite on a CUDA-capable machine.
#
# CUDA is intentionally NOT a dependency of the default test environment (CI has
# no GPU, and CUDA's stack does not build on 32-bit), so the GPU tests live in
# this separate environment. Run them with:
#
#     julia test/gpu/run_gpu_tests.jl
#
# This activates `test/gpu`, dev-installs the local CATN, instantiates (downloads
# CUDA/cuTENSOR artifacts on first run), and runs the GPU testset on the device.

using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path = normpath(joinpath(@__DIR__, "..", ".."))))
Pkg.instantiate()

using Test
# `exact_contract` oracle, then the GPU testset (which has its own CUDA.functional() guard).
include(joinpath(@__DIR__, "..", "exact.jl"))
include(joinpath(@__DIR__, "..", "test_gpu.jl"))
