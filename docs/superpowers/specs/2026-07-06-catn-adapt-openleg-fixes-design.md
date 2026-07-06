# CATN.jl — GPU element-type adapt + open-leg contraction fixes

**Date:** 2026-07-06
**Status:** Approved design
**Motivation:** three defects uncovered while benchmarking (see the GPU benchmark results).

## 1. Goal & scope

Fix three defects, all in the generic (`TensorNetwork`) / GPU path:

- **A. Element-type-inferring device move.** `cu(tn)` errors (`TypeError`) because `adapt` of a
  `Float64` network to `CuArray` (which `cu` downcasts to `Float32`) violates
  `MPSNode{T,AT<:AbstractArray{T,3}}`. There is also no way to obtain a `Float32` GPU network from
  a `Float64` one — which the benchmark showed is the *only* GPU-favorable configuration.
- **B. Open-leg (partial) contraction.** `contraction!` on a network with open legs throws
  `KeyError` (a negative open-leg sentinel id is looked up as a node), and even fixed, the result
  of contracting a network with open legs is a *tensor*, not a scalar — currently unrepresentable.
- **C. Docs.** The README recommends `cu(tn)` (broken) and doesn't state where GPU actually helps.

**In scope:** the generic `TensorNetwork` path, CPU + CUDA. **Out of scope:** the double-layer
`BraKetNetwork` (no `adapt`/GPU wiring yet); operator/expectation features.

## 2. Fix A — element-type-inferring `adapt`

`src/adapt.jl`, `adapt_structure(to, tn::TensorNetwork{T})` currently hardcodes the source `T`:
```julia
NAT = eltype(new_nodes[1].mps)
tensors = Dict{Int,MPSNode{T,NAT}}(...)         # BUG: T is the SOURCE eltype
TensorNetwork{T,NAT}(tensors, ..., tn.lnZ, tn.sign, tn.psi, ...)
```
Change it to infer the element type from the *adapted* arrays:
```julia
NAT = eltype(new_nodes[1].mps)
NT  = eltype(NAT)                                # element type AFTER the move
tensors = Dict{Int,MPSNode{NT,NAT}}(...)
TensorNetwork{NT,NAT}(tensors, tn.Dmax, ..., NT(tn.lnZ), NT(tn.sign), NT(tn.psi),
                      tn.maxdim_intermediate, tn.num_isolated, deepcopy(tn.rng), tn.n, tn.beta)
```
(`MPSNode`'s `adapt_structure` already infers `T`/`AT` via its positional constructor, so no change
there.) Result:
- `adapt(Array, tn)` — no eltype change (identity), still works.
- `adapt(CuArray, tn)` — preserves `Float64` (`adapt(CuArray, ::Array{Float64})` keeps `Float64`).
- `cu(tn)` — `cu` downcasts `Float64→Float32`, so this now yields a valid
  `TensorNetwork{Float32,CuArray{Float32,3}}` — the fast GPU path.

Scalar-accumulator conversion (`NT(tn.lnZ)` etc.) matters only when the eltype changes
(`Float64→Float32`); for identity/eltype-preserving moves `NT == T` and it's a no-op. `lnZ`
carries a real magnitude; `NT(...)` of a real-valued number into `Float32`/`ComplexF32` is fine.

## 3. Fix B — open-leg (partial) contraction

### 3.1 KeyError
`count_remove_nodes!` and `count_add_nodes!` (`src/tensor_network.jl`) iterate a `nodes` list that
(from `contraction!`) includes open-leg sentinel ids (negative), then do
`tn.tensors[node_id]`. Add `node_id <= 0 && continue` at the top of each function's node loop so
sentinels are skipped. (The selection helpers and the eat re-pointing block already skip
sentinels; only these two bookkeeping functions don't.)

### 3.2 Tensor result
After `contraction!` contracts all real bonds, a network with open legs leaves one surviving node
carrying the open legs (a tensor), not a scalar.
- `network_lognorm` (the post-loop fold) must fold **only scalar** surviving nodes into `lnZ`
  (a node all of whose site physical dims are 1). Non-scalar (open-leg) surviving nodes are left
  untouched — their content is the result, not a norm.
- Add `result_tensor(tn::TensorNetwork) -> Array`: after `contraction!`, returns `mps2raw` of the
  single surviving non-scalar node, with legs in the open-label order (documented); if the network
  was fully closed (no open legs), returns a 0-dimensional array holding `one(T)`.
- **Usage:** closed network → result is `exp(lnZ) * psi` (unchanged). Open network → result is
  `result_tensor(tn) .* exp(lnZ) .* psi` (a tensor; `result_tensor` returns the normalized surviving
  tensor and `exp(lnZ)*psi` is the pulled-out norm/phase).

`contraction!`'s signature is unchanged: `(lnZ, error, psi)`. It leaves the surviving node(s) in
`tn.tensors` so `result_tensor` can read them. v1 assumes a single connected component (one
surviving open node); disconnected/multi-open-node networks are out of scope (document).

### 3.3 Leg order of the result
`result_tensor` returns the surviving node's `mps2raw`, whose leg order follows the node's MPS
site order at the end of contraction (which corresponds to the open labels). The open-leg test
verifies the *values* against `exact_contract` up to a leg permutation, and documents the order.

## 4. Fix C — README

Update the GPU section:
- `adapt(CuArray, tn)` keeps the element type (`Float64`); `cu(tn)` gives a `Float32` GPU network.
- Note (from the benchmark): the GPU pays off only for **large, dense, `Float32`** contractions
  (crossover ~D≈256, ~2.5–3× by D≈1024 on an RTX 6000); for Ising / modest bond dimensions the
  CPU is faster.
- Mention `result_tensor` for open-leg (partial) contractions.

## 5. Testing

- **Fix A (GPU, `test/gpu`):** `gtn = cu(tn)` yields `TensorNetwork{Float32,CuArray{Float32,3}}`
  (assert the type); `contraction!(gtn)` matches the CPU `Float64` contraction within a `Float32`
  tolerance (`rtol ~1e-4`); `adapt(CuArray, tn)` (no `cu`) yields a `Float64` CuArray network and
  matches to `~1e-10`. Guarded by `CUDA.functional()`.
- **Fix B (CPU, `test/test_contraction.jl` or a new `test/test_openleg.jl`):** contract a network
  with open legs (a chain `T1(p1,a) T2(a,p2,b) T3(b,p3)` and a small tree) and assert
  `result_tensor(tn) .* exp(lnZ) .* psi ≈ exact_contract(tensors, ixs)` (a tensor) up to leg
  permutation, `rtol ~1e-10`, real and complex. Also a closed-network sanity: `result_tensor` is
  the 0-d `one(T)` and `exp(lnZ)*psi` matches the oracle (existing behavior preserved).
- **Regression:** the existing 217-test suite stays green (closed-network scalar contractions and
  all Ising/complex/adapt tests unchanged).

## 6. Deliverables

- `src/adapt.jl`: element-type-inferring `adapt_structure` for `TensorNetwork`.
- `src/tensor_network.jl`: sentinel-skipping `count_remove_nodes!`/`count_add_nodes!`;
  scalar-only fold in `network_lognorm`; new `result_tensor`; export `result_tensor`.
- `test/`: GPU Float32 `cu` test; CPU open-leg-vs-oracle test.
- `README.md`: corrected GPU usage + `result_tensor` note.
- Full suite green.
