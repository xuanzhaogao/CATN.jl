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
lnZ, _, _ = contraction!(tn)
lnZ_per_site, F = free_energy(tn)
```

## Limitations

Complex-valued tensors are not fully supported or validated in this version — the Ising application and all test coverage use real-valued tensors.
