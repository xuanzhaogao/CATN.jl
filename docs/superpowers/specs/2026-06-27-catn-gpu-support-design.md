# CATN.jl — GPU (CUDA) support

**Date:** 2026-06-27
**Status:** Approved design
**Builds on:** the CPU implementation merged in `docs/superpowers/specs/2026-06-26-catn-julia-design.md`.

## 1. Goal & scope

Let CATN.jl contract tensor networks on the GPU by making the engine **device-agnostic**:
hold whatever array type the user provides and let **OMEinsum's CUDAExt** (for `ein` contractions)
and **CUDA.jl's cuSOLVER dispatch** (for `svd`/`qr`) do the GPU compute. We write *no* CUDA
kernels ourselves — GPU support is emergent from feeding `CuArray`s through the existing code.

**In scope:**
- Generalize tensor storage from the hardcoded CPU `Array{T,3}` to a parameterized array type
  `AT`, so `CuArray{T,3}` (or any `AbstractArray{T,3}`) flows through unchanged.
- Keep `raw2mps` and the truncation utilities (`tsvd`/`rsvd`) device-preserving and GPU-safe
  (no scalar indexing into device arrays).
- An `Adapt.adapt_structure` integration so `adapt(CuArray, tn)` / `cu(tn)` moves a whole
  network (generic or Ising) to the device; the Ising builder stays CPU and the user adapts
  before `contraction!`.
- `Adapt` as a light direct dependency; `CUDA` (+ optional `cuTENSOR`) as **test-only** extras;
  GPU tests guarded by `CUDA.functional()`.

**Out of scope (unchanged from CPU version):**
- Complex-tensor support (still the documented limitation: `eat!`/`cut_bondim!` use conjugating
  adjoints where the reference uses plain transpose). GPU work stays real-valued.
- Multi-GPU / distributed; AMDGPU/Metal validation (the design is device-agnostic so they
  *should* work via their OMEinsum extensions, but we only validate CUDA on the available
  RTX 6000).
- Hand-written CUDA kernels or a hard CUDA dependency.

**Target hardware (validation):** NVIDIA Quadro RTX 6000 (24 GB) on the CCM workstation.
CUDA.jl provides its own toolkit artifacts, so no system CUDA toolkit is required.

## 2. Approach

**Device-agnostic core + Adapt** (chosen over a hard CUDA dependency and over a CUDA package
extension). GPU acceleration comes from OMEinsum's `CUDAExt` and CUDA.jl's LinearAlgebra
dispatch; CATN itself stays free of CUDA code.

**Storage type:** parameterize the array type (`MPSNode{T,AT}`, `TensorNetwork{T,AT}`) rather
than using an abstract `Vector{AbstractArray{T,3}}` field. This keeps the contraction hot loop
**type-stable** (monomorphic), avoiding per-op CPU-side dynamic dispatch between GPU kernel
launches — which is exactly what would erode the GPU benefit. The cost is threading `AT`
through struct definitions, constructors, and `adapt`; this is mechanical.

## 3. What changes (component by component)

### 3.1 `mps_node.jl` — storage generalization
```julia
mutable struct MPSNode{T,AT<:AbstractArray{T,3}}
    mps::Vector{AT}
    neighbor::Vector{Int}
    cano::Int
    chi::Int
    cutoff::Float64
    norm_method::Int
    svdopt::Bool
    swapopt::Bool
end
```
- The keyword constructor `MPSNode(tensor, neighbor; ...)` infers `AT` from `raw2mps`'s output.
  Add an inner/positional constructor taking all fields (used by `adapt` and the Ising builder).
- Empty-`mps` literals (`eat!` both-leaves case, isolated nodes) become `AT[]` (or
  `Vector{AT}()`), not `Array{T,3}[]`. Where the result must be empty but `AT` isn't otherwise
  known (e.g. the both-leaves `eat!` clearing `node.mps`), reuse the node's own `AT` (the
  method has `node::MPSNode{T,AT}`), so `node.mps = AT[]`.
- `raw2mps` drops the two `Array(tensor)` conversions: `reshape(tensor, 1, dims...)` preserves a
  `CuArray`. SVD on a `CuArray` dispatches to cuSOLVER. Build the order-0/1 and final sites from
  the input array type (reshape/`similar`), never a fresh CPU `Array`.
- All other methods (`mps2raw`, `cano_to!`, `swap!`, `merge!`, `compress!`, `eat!`, `reverse!`)
  are unchanged in logic: they already use `ein"…"` (OMEinsum), `reshape`/`permutedims`,
  `Diagonal`, `*`, `norm`, `dot`, `maximum` — all device-generic. Verify no stray `Array(...)`
  or scalar-index assumptions remain.

### 3.2 `tensor_network.jl` — storage generalization
```julia
mutable struct TensorNetwork{T,AT<:AbstractArray{T,3}}
    tensors::Dict{Int,MPSNode{T,AT}}
    # ... all other fields unchanged ...
end
```
- Thread `AT` through the generic constructor `TensorNetwork(tensors, ixs; ...)` (inferred from
  the input tensors' type) and the Ising-facing inner constructor.
- The contraction loop, selectors, and bookkeeping are unchanged: `neighbor` vectors, `edge_count`,
  `lnZ`/`sign`/`psi` stay host-side. Device reductions (`norm`, `dot`, `maximum`, `sum`) return
  host scalars, so nothing device-side leaks into the accumulators.

### 3.3 `linalg_utils.jl` — GPU-safe truncation
- `tsvd(A; cutoff, maxdim)`: unchanged API and return `(U, S, V, discarded)`. Make GPU-safe:
  - `svd(A)` (cuSOLVER on `CuArray`).
  - `nkeep` via `count(>(cutoff), S)` (reduction) — no `S[i]` scalar reads.
  - `discarded = nkeep < length(S) ? sum(@view S[nkeep+1:end]) : zero(real(eltype(S)))` (reduction).
  - Slices `U[:,1:nkeep]`, `S[1:nkeep]`, `V[:,1:nkeep]` stay on-device.
- `rsvd(A, k, oversample, power; rng)`: replace `randn(rng, T, n, p)` with
  `randn!(rng, similar(A, T, (n, p)))` so the sketch follows `A`'s array type. Document: on
  `CuArray`, randomness uses CUDA's device RNG (seed via `CUDA.seed!`); the `rng` arg is honored
  on CPU. `rsvd` is only reached for very large `swap!` matrices.

### 3.4 `adapt.jl` (new file) — device movement
```julia
using Adapt
Adapt.adapt_structure(to, n::MPSNode) = MPSNode(
    [adapt(to, t) for t in n.mps], copy(n.neighbor), n.cano, n.chi,
    n.cutoff, n.norm_method, n.svdopt, n.swapopt)            # via the positional constructor
Adapt.adapt_structure(to, tn::TensorNetwork) = <rebuild TensorNetwork with
    Dict(k => adapt(to, v) for (k,v) in tn.tensors), scalars copied, AT re-inferred>
```
- Result: `cu(tn)` (when `using CUDA`) and `adapt(CuArray, tn)` / `adapt(Array, tn_gpu)` move
  the network between host and device. `adapt(to, t)` on each 3-D array does the actual transfer.
- The Ising builder is unchanged (builds tiny δ-tensors on CPU). Users do
  `contraction!(cu(tn))`.

### 3.5 `Project.toml`
- `[deps]`: add `Adapt` (light, pure-Julia). Compat `Adapt = "4"` (matches CUDA 5/6 era).
- `[extras]`: add `CUDA`, optionally `cuTENSOR`. `[targets] test = [..., "CUDA"]`.
- `src/CATN.jl`: `using Adapt`; include `adapt.jl`; no new exports required (`adapt`/`cu` come
  from Adapt/CUDA).

## 4. Testing & validation

`test/test_gpu.jl`, **entirely guarded by `CUDA.functional()`** (skipped on CPU-only CI; the
existing 123-test CPU suite is untouched):

1. **CPU type-generalization spot-check** (runs always, not GPU-gated): after the struct change,
   a CPU `TensorNetwork` is `TensorNetwork{Float64,Array{Float64,3}}` and all existing behavior
   is identical (the 123 tests already enforce this).
2. **`adapt` round-trip:** `cu(tn)` has `CuArray` MPS tensors; `adapt(Array, cu(tn))` returns a
   CPU network whose `mps2raw` matches the original.
3. **GPU == CPU == oracle:** for the generic engine (`TensorNetwork(cu.(tensors), ixs)`) and an
   `adapt`ed Ising network — chain, single loop, small 2-D grid — `contraction!` on GPU returns
   the same `lnZ`/`psi` as the CPU run and `exact_contract`, `rtol ~1e-10` (Float64), `~1e-5`
   (Float32).
4. **`tsvd`/`rsvd` on `CuArray`:** reconstruct `A ≈ U·Diagonal(S)·V'` on device (guards against
   scalar-indexing regressions).
5. **Float32 GPU path:** a `Float32` network contracts on GPU (GPUs favor F32).
6. **Perf smoke (non-gating):** a larger grid runs on GPU without error; optionally log CPU vs
   GPU wall time.

**Gate:** full CPU suite stays green (123 tests); GPU suite passes on the RTX 6000.

## 5. Key risks & decisions

1. **GPU scalar indexing** is the top regression risk — `CUDA.jl` errors on `A[i]` reads of
   device arrays by default. Audit `tsvd`/`rsvd` and the MPS methods for scalar reads; use
   reductions/views. (Construction-time CPU assignments in the Ising builder are fine — they
   run before `adapt`.)
2. **`AT` threading churn:** mechanical but touches many signatures; keep the diff focused on
   type parameters + constructors, not logic.
3. **`rsvd` RNG on device:** device RNG, documented; rare path.
4. **Type stability:** verify the contraction loop stays monomorphic for a `CuArray` network
   (no `MPSNode{T}` UnionAll leaking into the `Dict` value type).
5. **Determinism:** GPU reductions/SVD may differ from CPU at ~1e-12; tolerances account for it.

## 6. Deliverables

- `MPSNode{T,AT}` / `TensorNetwork{T,AT}` device-agnostic storage; device-preserving `raw2mps`;
  GPU-safe `tsvd`/`rsvd`; `src/adapt.jl` with `Adapt` integration.
- `Project.toml` with `Adapt` dep and `CUDA` test extra.
- `test/test_gpu.jl` passing on the RTX 6000; CPU suite still green.
- README GPU usage snippet (`contraction!(cu(tn))`).
