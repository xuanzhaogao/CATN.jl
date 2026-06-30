# CATN

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://xuanzhaogao.github.io/CATN.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://xuanzhaogao.github.io/CATN.jl/dev/)
[![Build Status](https://github.com/xuanzhaogao/CATN.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/xuanzhaogao/CATN.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/xuanzhaogao/CATN.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/xuanzhaogao/CATN.jl)

A Julia implementation of **CATN** (Contracting Arbitrary Tensor Networks,
[arXiv:1912.03014](https://arxiv.org/abs/1912.03014)) — an algorithm for *approximately*
contracting tensor networks of arbitrary topology. Every tensor is represented as its own
Matrix Product State (MPS), and the network is contracted edge by edge with controlled
truncation. [OMEinsum](https://github.com/under-Peter/OMEinsum.jl) is the tensor-contraction
backend (and, on GPU, OMEinsum's CUDA extension).

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/xuanzhaogao/CATN.jl")
```

## Quick start

### Generic tensor network

Specify a network the way you would an einsum: a list of tensors plus, for each, its leg
labels. A label shared by exactly two tensors is a contracted bond; a label appearing once is
an open leg.

```julia
using CATN

A, B, C = randn(2,3), randn(3,4), randn(4,2)
tn = TensorNetwork([A, B, C], [[:a,:b], [:b,:c], [:c,:a]]; Dmax=32, chi=64)
lnZ, err, psi = contraction!(tn)
value = exp(lnZ) * psi      # ≈ tr(A*B*C)
```

`contraction!` returns `(lnZ, error, psi)`: the contracted scalar is `exp(lnZ) * psi`, where
`lnZ` accumulates the log-magnitudes (kept real) and `psi` carries the sign/phase. `error` is
the accumulated truncation error (the sum of discarded singular values). Note `contraction!`
consumes the network in place.

### Ising / spin-glass free energy

`ising_network` builds the partition-function tensor network for an Ising model on an
arbitrary graph (energy `E = Σ_{(i,j)} w_{ij} s_i s_j + Σ_i h_i s_i`, Boltzmann weight
`exp(β E)`):

```julia
edges   = [(1,2), (2,3), (3,1)]   # 1-based node indices
weights = ones(3)                 # coupling J on each edge
fields  = zeros(3)                # external field h on each spin
β       = 0.4

tn = ising_network(3, edges, weights, fields, β; Dmax=20, chi=200)
lnZ_per_site, F = free_energy(tn)   # runs the contraction internally; F = -lnZ_per_site/β
```

### Observables (magnetization & correlations)

Computed via spin pinning (each rebuilds and contracts the network internally):

```julia
m = magnetization(n, edges, weights, fields, β; Dmax=-1)   # ⟨s_i⟩, one per spin
c = correlation(n, edges, weights, fields, β; Dmax=-1)     # ⟨s_i s_j⟩, one per edge
```

## Truncation & options

Two knobs control the approximation; everything else has sensible defaults.

| keyword | meaning |
|---|---|
| `Dmax` | maximum physical bond dimension between tensors. **`Dmax < 0` ⇒ exact contraction** (no truncation). |
| `chi` | maximum virtual bond dimension *inside* each tensor's MPS. |
| `cutoff` | singular values below this are dropped (default `1e-15`). |
| `select` | edge-selection heuristic: `0` min intermediate dimension, `1` min-dim + triangle preference (default), `2` sequential. |
| `norm_method` | per-step normalization: `0` none, `1` L2 (default), `2` max-abs. |
| `reverse`, `compress`, `svdopt`, `swapopt`, `cut_bond` | performance/accuracy switches (MPS reversal to minimize swaps, whole-MPS compression, QR-before-SVD, randomized SVD for huge swaps, always-cut bonds). |
| `seed` | RNG seed (affects the randomized-SVD path only). |

## GPU

CATN is device-agnostic: it holds whatever array type you give it, and GPU compute is provided
by OMEinsum's CUDA extension (contractions) and cuSOLVER (`svd`/`qr`). Move a network to the
device with `cu` (or `adapt(CuArray, tn)`) and contract there:

```julia
using CATN, CUDA
tn = ising_network(n, edges, w, h, β; Dmax=64, chi=256)   # built on CPU
lnZ, err, psi = contraction!(cu(tn))                       # contracted on the GPU
```

For the generic path, build the network directly from `CuArray`s:

```julia
gtn = TensorNetwork([CuArray(A), CuArray(B), CuArray(C)], ixs; Dmax=64, chi=256)
```

`Float32` networks (`randn(Float32, …)`) are recommended on GPU for speed. **CATN has no CUDA
dependency** — GPU support is provided through OMEinsum's CUDA extension when you load `CUDA`
yourself, so CPU-only users install nothing extra. The GPU test suite lives in a separate
environment (`test/gpu`) and is run on a CUDA-capable machine with:

```
julia test/gpu/run_gpu_tests.jl
```

## Complex numbers

Complex-valued tensors are supported in the generic `TensorNetwork` path. Tensor-network bonds
are contracted *without* conjugation, and the result `exp(lnZ) * psi` carries the complex phase
in `psi`. The `ising_network` convenience builder is real-valued (physical couplings are real).

## How it works

1. Each input tensor is decomposed into an MPS (one site per neighbor) via successive SVDs.
2. The network is contracted edge by edge: an edge is selected (by the `select` heuristic), the
   shared bond is brought to the MPS ends and contracted (`eat!`), duplicate bonds created by
   loops are merged, and oversized bonds are truncated to `Dmax`.
3. Per-step log-norms accumulate into `lnZ`; the final value is `exp(lnZ) * psi`.

Correctness is validated against exact tensor-network contraction (OMEinsum's optimized
contraction) for small networks, and the Ising layer is checked against analytic
transfer-matrix and brute-force results.

### Double-layer (bra–ket) norm

`braket_network` builds the double-layer network for the norm `⟨ψ|ψ⟩` of a tensor-network
state `|ψ⟩` (open labels = physical indices, shared labels = virtual bonds). The ket and bra
layers are kept separate on every network edge (RAM-efficient); they combine only inside each
node's MPS, where the environment-aware `χ`-truncation happens.

```julia
using CATN
# state |ψ⟩: a chain T1(p1,a) T2(a,p2,b) T3(b,p3)
tensors = [randn(ComplexF64,2,3), randn(ComplexF64,3,2,3), randn(ComplexF64,3,2)]
ixs     = [[:p1,:a], [:a,:p2,:b], [:b,:p3]]
bk = braket_network(tensors, ixs; chi=64)
lnZ, err, psi = contraction!(bk)
norm2 = exp(lnZ) * psi      # ≈ ⟨ψ|ψ⟩  (real, ≥ 0)
```

v1 supports acyclic virtual-bond graphs (chains/trees) and the norm; operators `⟨ψ|O|ψ⟩`,
overlaps, and loopy (PEPS) states are planned follow-ons.

## Reference

*Contracting Arbitrary Tensor Networks: General Approximate Algorithm and Applications in
Graphical Models and Quantum Circuit Simulations*,
[arXiv:1912.03014](https://arxiv.org/abs/1912.03014).
