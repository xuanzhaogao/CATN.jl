# CATN

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://xuanzhaogao.github.io/CATN.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://xuanzhaogao.github.io/CATN.jl/dev/)
[![Build Status](https://github.com/xuanzhaogao/CATN.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/xuanzhaogao/CATN.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/xuanzhaogao/CATN.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/xuanzhaogao/CATN.jl)

## Usage

```julia
using CATN

# Generic tensor network (OMEinsum-style labels)
A, B, C = randn(2,3), randn(3,4), randn(4,2)
tn = TensorNetwork([A,B,C], [[:a,:b],[:b,:c],[:c,:a]]; Dmax=32, chi=64)
lnZ, err, psi = contraction!(tn)
value = exp(lnZ) * psi    # ≈ tr(A*B*C)

# Ising free energy on an arbitrary graph
edges = [(1,2),(2,3),(3,1)]
tn = ising_network(3, edges, ones(3), zeros(3), 0.4; Dmax=20, chi=200)
lnZ_per_site, F = free_energy(tn)   # free_energy runs the contraction internally
```

### GPU

CATN is device-agnostic. Move a network to the GPU with `cu` (or `adapt(CuArray, tn)`) and
contract there — the contractions run via OMEinsum's CUDA extension and `svd`/`qr` via cuSOLVER:

```julia
using CATN, CUDA
tn = ising_network(n, edges, w, h, β; Dmax=64, chi=256)   # built on CPU
lnZ, err, psi = contraction!(cu(tn))                       # contracted on the GPU
```

`Float32` networks (`randn(Float32, …)`) are recommended on GPU for speed. CUDA is a
test-only dependency of CATN — the core has no CUDA dependency.

## Limitations

Complex-valued tensors are not fully supported or validated in this version — the Ising application and all test coverage use real-valued tensors.
