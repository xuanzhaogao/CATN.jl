# CATN.jl GPU Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CATN.jl contract tensor networks on the GPU by generalizing tensor storage to a parameterized array type, letting `CuArray`s flow through to OMEinsum's CUDAExt and cuSOLVER — no hand-written CUDA code.

**Architecture:** Parameterize `MPSNode{T,AT}` and `TensorNetwork{T,AT}` over a concrete 3-D array type `AT` (type-stable). Keep `raw2mps`/`tsvd`/`rsvd` device-preserving and scalar-index-free. Add `Adapt.adapt_structure` so `adapt(CuArray, tn)`/`cu(tn)` moves a whole network to the device; the Ising builder stays CPU and the user adapts before `contraction!`.

**Tech Stack:** Julia 1.12 (juliaup at `/mnt/home/xgao1/.juliaup/bin/julia`), OMEinsum 0.9 (CUDAExt), Adapt (dep), CUDA + cuTENSOR (test-only), LinearAlgebra, Random, Test.

**Reference:** spec `docs/superpowers/specs/2026-06-27-catn-gpu-support-design.md`. Target GPU: NVIDIA Quadro RTX 6000 (24 GB) on this CCM workstation.

## Global Constraints

- Julia via the juliaup binary `/mnt/home/xgao1/.juliaup/bin/julia` (on PATH as `julia`); never `module load julia`. Full suite: `julia --project=. -e 'using Pkg; Pkg.test()'`; faster iteration: `julia --project=. -e 'include("test/runtests.jl")'`.
- **No hand-written CUDA code.** GPU compute is emergent: OMEinsum's CUDAExt for `ein` contractions, CUDA.jl's cuSOLVER dispatch for `svd`/`qr`. CATN has **no** `CUDA` dependency in `[deps]`.
- **Type-stable storage:** `MPSNode{T,AT<:AbstractArray{T,3}}` with `mps::Vector{AT}`; `TensorNetwork{T,AT}` with `tensors::Dict{Int,MPSNode{T,AT}}`. Do NOT use an abstract `Vector{AbstractArray{T,3}}` field.
- **No scalar indexing into device arrays** (`A[i]` reads on a `CuArray` error by default). Use reductions/views in `tsvd`/`rsvd`.
- `Adapt` is a light direct dependency (compat `"4"`). `CUDA` (compat `"5, 6"`) and `cuTENSOR` are **test-only** (`[extras]` + `[targets]`), added in Task 5. GPU tests are guarded by `CUDA.functional()` and skipped on CPU-only machines.
- **The existing 123-test CPU suite must stay green after every task.** Tasks 1–4 are CPU-verifiable refactors.
- Element type stays generic (`Float64` default; `Float32` must work — GPUs favor it).
- Existing method signatures written `f(x::MPSNode{T}) where {T}` still match the 2-param struct (`MPSNode{T}` is the UnionAll `MPSNode{T,AT} where AT`); do NOT churn them. Only struct defs, constructors, and `Array{...,3}[]` literals change.
- Commit after each task with trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File Structure

```
src/
  CATN.jl            # add `using Adapt`; include adapt.jl
  linalg_utils.jl    # rsvd: device-following random sketch (Task 3)
  mps_node.jl        # MPSNode{T,AT} + positional ctor + raw2mps device-preserving (Task 1)
  tensor_network.jl  # TensorNetwork{T,AT} + generic ctor AT inference (Task 2)
  ising.jl           # node/network construction type annotations (Task 2)
  adapt.jl           # NEW: Adapt.adapt_structure for MPSNode + TensorNetwork (Task 4)
test/
  runtests.jl        # include test_gpu.jl (Task 5)
  test_gpu.jl        # NEW: GPU tests guarded by CUDA.functional() (Task 5)
Project.toml         # Adapt dep (Task 4); CUDA+cuTENSOR test extras (Task 5)
```

---

### Task 1: Parameterize `MPSNode{T,AT}`

**Files:**
- Modify: `src/mps_node.jl` (struct lines 1-10; keyword ctor 12-18; `raw2mps` 20-38; `eat!` empty-`mps` literals at lines 463, 469)
- Modify: `test/test_mps_node.jl` (add a type spot-check)

**Interfaces:**
- Produces:
  - `mutable struct MPSNode{T,AT<:AbstractArray{T,3}}` with `mps::Vector{AT}` (other fields unchanged).
  - Positional constructor `MPSNode(mps::Vector{<:AbstractArray{T,3}}, neighbor::Vector{Int}, cano::Int, chi::Int, cutoff::Float64, norm_method::Int, svdopt::Bool, swapopt::Bool) where {T}` → infers `AT = eltype(mps)`. Used by the keyword ctor, the Ising builder (Task 2), and `adapt` (Task 4).
  - Keyword constructor `MPSNode(tensor::AbstractArray{T}, neighbor::Vector{Int}; chi, cutoff, norm_method, svdopt, swapopt)` unchanged in API; now routes through the positional ctor.
  - `raw2mps(tensor, chi, cutoff)` returns `Vector{AT}` where `AT` follows `tensor`'s device (no CPU `Array` conversion).

- [ ] **Step 1: Write the failing test** — append to `test/test_mps_node.jl`

```julia
@testset "MPSNode is array-type parameterized" begin
    node = MPSNode(randn(2, 3, 4), [1, 2, 3]; chi=1000)
    @test node isa MPSNode{Float64, Array{Float64, 3}}
    @test eltype(node.mps) == Array{Float64, 3}
    @test mps2raw(node) ≈ mps2raw(node)   # still reconstructs
    # positional constructor infers AT from the mps vector
    n2 = MPSNode(node.mps, [1, 2, 3], node.cano, 1000, 1e-15, 1, true, true)
    @test n2 isa MPSNode{Float64, Array{Float64, 3}}
end
```

- [ ] **Step 2: Run, expect failure**

Run: `julia --project=. -e 'include("test/runtests.jl")'`
Expected: FAIL — `MPSNode{Float64, Array{Float64,3}}` is not valid for the current 1-parameter struct (TypeError / wrong number of parameters).

- [ ] **Step 3: Edit the struct** — `src/mps_node.jl` lines 1-10

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

- [ ] **Step 4: Add the positional constructor and update the keyword constructor** — replace the keyword ctor (lines 12-18) with:

```julia
# Positional constructor: infer the array-type parameter AT from the mps vector.
function MPSNode(mps::Vector{<:AbstractArray{T,3}}, neighbor::Vector{Int}, cano::Int,
                 chi::Int, cutoff::Float64, norm_method::Int, svdopt::Bool,
                 swapopt::Bool) where {T}
    MPSNode{T,eltype(mps)}(mps, neighbor, cano, chi, cutoff, norm_method, svdopt, swapopt)
end

function MPSNode(tensor::AbstractArray{T}, neighbor::Vector{Int};
                chi::Int=32, cutoff::Float64=1e-15, norm_method::Int=1,
                svdopt::Bool=true, swapopt::Bool=true) where {T}
    mps = raw2mps(tensor, chi, cutoff)
    cano = length(mps)          # left-canonical: center at last site
    MPSNode(mps, copy(neighbor), cano, chi, cutoff, norm_method, svdopt, swapopt)
end
```

- [ ] **Step 5: Make `raw2mps` device-preserving** — replace `raw2mps` (lines 20-38) with:

```julia
function raw2mps(tensor::AbstractArray{T}, chi::Int, cutoff::Float64) where {T}
    nd = ndims(tensor)
    AT = typeof(similar(tensor, T, (1, 1, 1)))   # concrete 3-D array type on tensor's device
    nd == 0 && return AT[]
    dims = size(tensor)
    nd == 1 && return AT[reshape(tensor, 1, dims[1], 1)]
    mps = AT[]
    R = reshape(tensor, 1, dims...)              # preserves device (no Array() copy)
    dleft = 1
    for i in 1:nd-1
        M = reshape(R, dleft * dims[i], :)
        U, S, V = tsvd(M; cutoff=cutoff, maxdim=chi)
        χ = length(S)
        push!(mps, reshape(U, dleft, dims[i], χ))
        R = reshape(Diagonal(S) * V', χ, dims[(i+1):end]...)
        dleft = χ
    end
    push!(mps, reshape(R, dleft, dims[nd], 1))
    return mps
end
```
(Only change vs current: `AT` inference and dropping the two `Array(tensor)` calls. `U, S, V = tsvd(...)` still ignores the 4th return value.)

- [ ] **Step 6: Fix `eat!` empty-`mps` literals** — at `src/mps_node.jl` lines 463 and 469, replace `node.mps = Array{T,3}[]` with `empty!(node.mps)` (clears in place, preserving `Vector{AT}`).

- [ ] **Step 7: Run the full suite, expect pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS — all prior tests plus the new type spot-check (124+ total). If any test errors with `convert`/`push!` type issues on `reshape` results, wrap the pushed site as `convert(AT, reshape(...))` (CPU `Array` should not need this).

- [ ] **Step 8: Commit**

```bash
git add src/mps_node.jl test/test_mps_node.jl
git commit -m "feat: parameterize MPSNode over array type (device-agnostic storage)"
```

---

### Task 2: Parameterize `TensorNetwork{T,AT}`

**Files:**
- Modify: `src/tensor_network.jl` (struct lines 1-22; generic ctor body around lines 105-117)
- Modify: `src/ising.jl` (node Dict + node ctor at lines 51-53; `TensorNetwork{Float64}` at line 205)
- Modify: `test/test_contraction.jl` (add a type spot-check)

**Interfaces:**
- Consumes: `MPSNode{T,AT}` and its positional constructor (Task 1).
- Produces:
  - `mutable struct TensorNetwork{T,AT<:AbstractArray{T,3}}` with `tensors::Dict{Int,MPSNode{T,AT}}` (all other fields unchanged, including `n::Int`, `beta::Float64`).
  - Generic constructor `TensorNetwork(tensors, ixs; ...)` unchanged in API; infers `AT` from the input tensors.

- [ ] **Step 1: Write the failing test** — append to `test/test_contraction.jl`

```julia
@testset "TensorNetwork is array-type parameterized" begin
    A = randn(2,3); B = randn(3,4); C = randn(4,2)
    tn = TensorNetwork([A,B,C], [[:a,:b],[:b,:c],[:c,:a]]; chi=1000)
    @test tn isa TensorNetwork{Float64, Array{Float64,3}}
    @test valtype(tn.tensors) == MPSNode{Float64, Array{Float64,3}}
end
```

- [ ] **Step 2: Run, expect failure**

Run: `julia --project=. -e 'include("test/runtests.jl")'`
Expected: FAIL — `TensorNetwork{Float64, Array{Float64,3}}` invalid for the current 1-parameter struct.

- [ ] **Step 3: Edit the struct** — `src/tensor_network.jl` lines 1-22, change the header and the `tensors` field:

```julia
mutable struct TensorNetwork{T,AT<:AbstractArray{T,3}}
    tensors::Dict{Int,MPSNode{T,AT}}
    Dmax::Int
    chi::Int
    cutoff::Float64
    norm_method::Int
    select::Int
    reverse::Bool
    svdopt::Bool
    swapopt::Bool
    compress::Bool
    cut_bond::Bool
    edge_count::Dict{Int,Vector{Vector{Int}}}
    lnZ::T
    sign::T
    psi::T
    maxdim_intermediate::Int
    num_isolated::Int
    rng::AbstractRNG
    n::Int
    beta::Float64
end
```

- [ ] **Step 4: Update the generic constructor** — in `src/tensor_network.jl`, where it currently does `nodes = Dict{Int,MPSNode{T}}()` (line ~105) and `TensorNetwork{T}(...)` (line ~117), infer `AT` and use the 2-parameter types:

```julia
    AT = typeof(similar(tensors[1], T, (1, 1, 1)))   # device/type of the inputs
    nodes = Dict{Int,MPSNode{T,AT}}()
    # ... (unchanged loop that builds nodes[t] = MPSNode(tensors[t], ...; ...)) ...
    TensorNetwork{T,AT}(
        nodes, Dmax, chi, cutoff, norm_method, select, reverse, svdopt, swapopt,
        compress, cut_bond, edge_count, lnZ, sign, psi, maxdim_intermediate,
        num_isolated, rng, n, beta)
```
Match the EXACT existing argument list/order of the `TensorNetwork{T}(...)` call — only change `{T}` → `{T,AT}` and `Dict{Int,MPSNode{T}}` → `Dict{Int,MPSNode{T,AT}}`. (Read the surrounding code; keep every other line identical.)

- [ ] **Step 5: Update `src/ising.jl`** — at lines 51-53 change the node Dict and node constructor type, and at line 205 the `TensorNetwork` constructor:

```julia
    nodes = Dict{Int, MPSNode{Float64, Array{Float64,3}}}()
    for k in 1:n
        node = MPSNode{Float64, Array{Float64,3}}(
            Array{Float64,3}[],   # empty mps, populated below (CPU; user adapts to GPU later)
            Int[], 0, chi, cutoff, norm_method, svdopt, swapopt)
        nodes[k] = node
    end
```
and at line 205, `TensorNetwork{Float64}(` → `TensorNetwork{Float64, Array{Float64,3}}(` (keep the argument list identical).

- [ ] **Step 6: Run the full suite, expect pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS — all prior tests plus the new type spot-check.

- [ ] **Step 7: Commit**

```bash
git add src/tensor_network.jl src/ising.jl test/test_contraction.jl
git commit -m "feat: parameterize TensorNetwork over array type"
```

---

### Task 3: GPU-safe `rsvd` (and confirm `tsvd`)

**Files:**
- Modify: `src/linalg_utils.jl` (`rsvd`)
- Modify: `test/test_linalg.jl` (add a determinism note test)

**Interfaces:**
- Produces: `rsvd(A, k, oversample, power; rng)` unchanged in API/return `(U, S, V)`; the random sketch now follows `A`'s array type via `similar` + `randn!`, so it works on `CuArray` (device RNG).

Background: `tsvd` already avoids scalar indexing (it uses `count(>(cutoff), S)` and `sum(@view S[...])`, both reductions), so it is GPU-safe as-is once `svd` dispatches to cuSOLVER — confirm during review, no change needed.

- [ ] **Step 1: Write the failing test** — append to `test/test_linalg.jl`

```julia
@testset "rsvd uses device-following random sketch" begin
    Random.seed!(2)
    L = randn(40, 3) * randn(3, 30)          # rank 3
    U, S, V = rsvd(L, 3, 10, 10; rng=MersenneTwister(7))
    @test U * Diagonal(S) * V' ≈ L atol=1e-8
    # CPU determinism preserved across calls with the same seed
    U2, S2, V2 = rsvd(L, 3, 10, 10; rng=MersenneTwister(7))
    @test S ≈ S2
end
```

- [ ] **Step 2: Run, expect pass-or-fail**

Run: `julia --project=. -e 'include("test/runtests.jl")'`
Expected: PASS already (current `rsvd` is correct on CPU). This test pins the determinism behavior we must preserve through the edit. (If it fails, the existing rsvd is broken — stop and report.)

- [ ] **Step 3: Edit `rsvd`** — in `src/linalg_utils.jl`, replace the sketch line `Y = A * randn(rng, T, n, p)` with a device-following allocation:

```julia
    Y = A * randn!(rng, similar(A, T, (n, p)))
```
Everything else in `rsvd` stays the same. (`randn!`/`similar` are already in scope via `using Random`/`LinearAlgebra`. On CPU `Array`, `randn!(rng, Array{T}(undef,n,p))` produces identical values to `randn(rng,T,n,p)`, so the determinism test stays green. On `CuArray`, `similar` yields a `CuArray` and CUDA.jl's `randn!` fills it on-device.)

- [ ] **Step 4: Run the full suite, expect pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS (determinism preserved).

- [ ] **Step 5: Commit**

```bash
git add src/linalg_utils.jl test/test_linalg.jl
git commit -m "feat: rsvd random sketch follows input array type (GPU-ready)"
```

---

### Task 4: `Adapt` integration + dependency

**Files:**
- Modify: `Project.toml` (add `Adapt` to `[deps]` + `[compat]`)
- Create: `src/adapt.jl`
- Modify: `src/CATN.jl` (`using Adapt`; `include("adapt.jl")`)
- Modify: `test/runtests.jl` (include a new `test_adapt.jl`) and create `test/test_adapt.jl`

**Interfaces:**
- Consumes: `MPSNode{T,AT}` positional ctor (Task 1), `TensorNetwork{T,AT}` (Task 2).
- Produces: `Adapt.adapt_structure(to, ::MPSNode)` and `Adapt.adapt_structure(to, ::TensorNetwork)`, so `adapt(Array, tn)` / `adapt(CuArray, tn)` / `cu(tn)` move a network's tensors to the target array type. Networks containing a 0-dim-scalar (empty-`mps`) node are not adaptable to a different device (documented limitation; does not arise in Ising or normal use).

- [ ] **Step 1: Add the `Adapt` dependency** — `Project.toml`. Under `[deps]` add:
```toml
Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
```
Under `[compat]` add:
```toml
Adapt = "4"
```

- [ ] **Step 2: Write the failing test** — create `test/test_adapt.jl`

```julia
using CATN
using CATN: MPSNode, TensorNetwork, mps2raw, contraction!, ising_network
using Adapt
using Test

@testset "adapt to Array round-trips on CPU" begin
    # generic network
    A = randn(2,3); B = randn(3,4); C = randn(4,2)
    tn = TensorNetwork([A,B,C], [[:a,:b],[:b,:c],[:c,:a]]; chi=1000)
    tn2 = adapt(Array, tn)
    @test tn2 isa TensorNetwork{Float64, Array{Float64,3}}
    @test mps2raw(tn2.tensors[1]) ≈ mps2raw(tn.tensors[1])
    @test tn2.n == tn.n && tn2.beta == tn.beta && tn2.Dmax == tn.Dmax

    # adapting a node preserves the represented tensor
    node = MPSNode(randn(2,3,4), [1,2,3]; chi=1000)
    nb = adapt(Array, node)
    @test mps2raw(nb) ≈ mps2raw(node)

    # adapt(Array, ...) does not change the contraction result
    g1 = ising_network(3, [(1,2),(2,3),(3,1)], ones(3), zeros(3), 0.4; Dmax=-1, chi=10_000)
    g2 = ising_network(3, [(1,2),(2,3),(3,1)], ones(3), zeros(3), 0.4; Dmax=-1, chi=10_000)
    lnZ1, = contraction!(g1)
    lnZ2, = contraction!(adapt(Array, g2))
    @test lnZ1 ≈ lnZ2
end
```

- [ ] **Step 3: Run, expect failure**

Run: `julia --project=. -e 'using Pkg; Pkg.instantiate(); include("test/runtests.jl")'` (instantiate pulls Adapt)
Expected: FAIL — `adapt_structure` for `MPSNode`/`TensorNetwork` not defined (adapt returns the object unchanged or errors), and `test_adapt.jl` not yet wired.

- [ ] **Step 4: Create `src/adapt.jl`**

```julia
# Device movement via Adapt: `adapt(CuArray, tn)` / `cu(tn)` move every MPS tensor
# to the target array type. Compute then dispatches to OMEinsum's CUDAExt + cuSOLVER.
# Note: a network containing a 0-dim-scalar (empty-mps) node cannot change device
# (its AT is unknowable from no arrays); this does not arise in Ising or normal use.

function Adapt.adapt_structure(to, node::MPSNode)
    new_mps = map(t -> adapt(to, t), node.mps)
    MPSNode(new_mps, copy(node.neighbor), node.cano, node.chi,
            node.cutoff, node.norm_method, node.svdopt, node.swapopt)
end

function Adapt.adapt_structure(to, tn::TensorNetwork{T}) where {T}
    ks = sort(collect(keys(tn.tensors)))
    new_nodes = [adapt(to, tn.tensors[k]) for k in ks]
    NAT = eltype(new_nodes[1].mps)                       # AT after adapt
    tensors = Dict{Int,MPSNode{T,NAT}}(k => n for (k, n) in zip(ks, new_nodes))
    TensorNetwork{T,NAT}(
        tensors, tn.Dmax, tn.chi, tn.cutoff, tn.norm_method, tn.select, tn.reverse,
        tn.svdopt, tn.swapopt, tn.compress, tn.cut_bond,
        deepcopy(tn.edge_count), tn.lnZ, tn.sign, tn.psi, tn.maxdim_intermediate,
        tn.num_isolated, tn.rng, tn.n, tn.beta)
end
```
(Match the `TensorNetwork{T,NAT}(...)` positional argument order to the struct field order from Task 2 exactly.)

- [ ] **Step 5: Wire the module** — `src/CATN.jl`: add `using Adapt` near the other `using`s, and `include("adapt.jl")` after `include("ising.jl")` (so the types exist).

- [ ] **Step 6: Wire the test** — `test/runtests.jl`: add `include("test_adapt.jl")` inside the `@testset`.

- [ ] **Step 7: Run the full suite, expect pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS — adapt round-trip green; existing tests unaffected.

- [ ] **Step 8: Commit**

```bash
git add Project.toml src/adapt.jl src/CATN.jl test/runtests.jl test/test_adapt.jl
git commit -m "feat: Adapt integration to move networks across devices"
```

---

### Task 5: GPU validation on CUDA + README

**Files:**
- Modify: `Project.toml` (`[extras]` + `[targets]`: add `CUDA`, `cuTENSOR`)
- Create: `test/test_gpu.jl`
- Modify: `test/runtests.jl` (include `test_gpu.jl`)
- Modify: `README.md` (GPU usage snippet)

**Interfaces:**
- Consumes: everything from Tasks 1-4 (`MPSNode{T,AT}`, `TensorNetwork{T,AT}`, `adapt`, device-safe `tsvd`/`rsvd`).
- Produces: a GPU test suite guarded by `CUDA.functional()`, and documented GPU usage.

NOTE on first run: adding CUDA/cuTENSOR makes `Pkg.test()` download CUDA artifacts (~1-2 GB, several minutes) the first time. This machine has an RTX 6000 (driver 580); `CUDA.functional()` will be `true`.

- [ ] **Step 1: Add CUDA test-only deps** — `Project.toml`. Under `[extras]` add:
```toml
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
cuTENSOR = "011b41b2-24ef-40a8-b3eb-fa098493e9e1"
```
Under `[compat]` add:
```toml
CUDA = "5, 6"
cuTENSOR = "2"
```
Change the `[targets]` line to:
```toml
[targets]
test = ["Test", "CUDA", "cuTENSOR"]
```

- [ ] **Step 2: Write `test/test_gpu.jl`**

```julia
using CATN
using CATN: TensorNetwork, mps2raw, contraction!, ising_network, tsvd, rsvd
using Adapt, LinearAlgebra, Random, Test
using CUDA

if !CUDA.functional()
    @info "CUDA not functional — skipping GPU tests"
else
    using cuTENSOR     # enables OMEinsum's GPU einsum fast path
    # exact_contract is available from exact.jl (included earlier in runtests.jl)

    @testset "GPU" begin
        @testset "adapt to CuArray round-trips" begin
            A = randn(2,3); B = randn(3,4); C = randn(4,2)
            tn = TensorNetwork([A,B,C], [[:a,:b],[:b,:c],[:c,:a]]; chi=1000)
            gtn = adapt(CuArray, tn)
            @test gtn.tensors[1].mps[1] isa CuArray
            back = adapt(Array, gtn)
            @test mps2raw(back.tensors[1]) ≈ mps2raw(tn.tensors[1])
        end

        @testset "tsvd/rsvd on CuArray" begin
            Acpu = randn(8,5); A = CuArray(Acpu)
            U, S, V = tsvd(A)
            @test Array(U * Diagonal(S) * V') ≈ Acpu atol=1e-6
            Lcpu = randn(40,3) * randn(3,30); L = CuArray(Lcpu)
            Ur, Sr, Vr = rsvd(L, 3, 10, 10; rng=Random.default_rng())
            @test Array(Ur * Diagonal(Sr) * Vr') ≈ Lcpu atol=1e-5
        end

        @testset "GPU generic contraction matches oracle" begin
            ts = [randn(3,4), randn(4,5), randn(5,3)]
            ixs = [[:a,:b],[:b,:c],[:c,:a]]
            ref = exact_contract(ts, ixs)[]
            gtn = TensorNetwork([CuArray(t) for t in ts], ixs; Dmax=-1, chi=10_000)
            lnZ, err, psi = contraction!(gtn)
            @test exp(lnZ) * psi ≈ ref rtol=1e-8
        end

        @testset "GPU Ising matches CPU + oracle" begin
            edges = [(1,2),(2,3),(3,1)]
            w = [0.5,-0.8,1.1]; h = [0.2,-0.1,0.4]; β = 0.3
            cpu = ising_network(3, edges, w, h, β; Dmax=-1, chi=10_000)
            gpu = ising_network(3, edges, w, h, β; Dmax=-1, chi=10_000)
            lnZ_cpu, = contraction!(cpu)
            lnZ_gpu, = contraction!(adapt(CuArray, gpu))
            @test lnZ_gpu ≈ lnZ_cpu rtol=1e-8
        end

        @testset "GPU 2D Ising grid (finite Dmax) ≈ CPU" begin
            L = 6; β = 0.4
            edges = Tuple{Int,Int}[]
            idx(i,j) = (i-1)*L + j
            for i in 1:L, j in 1:L
                j < L && push!(edges, (idx(i,j), idx(i,j+1)))
                i < L && push!(edges, (idx(i,j), idx(i+1,j)))
            end
            n = L*L; w = ones(length(edges)); hh = zeros(n)
            cpu = ising_network(n, edges, w, hh, β; Dmax=16, chi=32, compress=true)
            gpu = ising_network(n, edges, w, hh, β; Dmax=16, chi=32, compress=true)
            lnZ_cpu, = contraction!(cpu)
            lnZ_gpu, = contraction!(adapt(CuArray, gpu))
            @test lnZ_gpu ≈ lnZ_cpu rtol=1e-5
        end

        @testset "Float32 GPU path" begin
            ts = [randn(Float32,3,4), randn(Float32,4,5), randn(Float32,5,3)]
            ixs = [[:a,:b],[:b,:c],[:c,:a]]
            ref = exact_contract([Float64.(t) for t in ts], ixs)[]
            gtn = TensorNetwork([CuArray(t) for t in ts], ixs; Dmax=-1, chi=10_000)
            lnZ, err, psi = contraction!(gtn)
            @test exp(lnZ) * psi ≈ ref rtol=1e-3
        end
    end
end
```

- [ ] **Step 3: Wire the test** — `test/runtests.jl`: add `include("test_gpu.jl")` inside the `@testset` (last).

- [ ] **Step 4: Run the full suite on the GPU machine**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: first run downloads CUDA artifacts, then PASS — CPU tests (123+) plus the `GPU` testset green on the RTX 6000. If `using cuTENSOR` causes issues, fall back to plain `using CUDA` (OMEinsum's CUDAExt still provides GPU einsum) and note it. If a GPU contraction mismatches the oracle, debug systematically (scalar-indexing error in tsvd/rsvd; a stray `Array(...)`; non-monomorphic `tn.tensors` value type) — do NOT loosen tolerances to pass.

- [ ] **Step 5: Add the README GPU section** — append to `README.md` (after the existing Usage section):

````markdown
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
````

- [ ] **Step 6: Commit**

```bash
git add Project.toml test/test_gpu.jl test/runtests.jl README.md
git commit -m "test: GPU validation on CUDA; document GPU usage"
```

---

## Self-Review

**Spec coverage:** §3.1 MPSNode generalization → Task 1; §3.2 TensorNetwork → Task 2; §3.3 GPU-safe tsvd/rsvd → Task 3 (tsvd confirmed scalar-index-free, rsvd edited); §3.4 adapt.jl → Task 4; §3.5 Project.toml (Adapt dep → Task 4; CUDA test extras → Task 5); §4 testing (CPU type checks → Tasks 1-2; adapt round-trip → Task 4; GPU==CPU==oracle, tsvd/rsvd on CuArray, Float32, grid → Task 5); §5 risks (scalar indexing → Task 3 + Task 5 debug note; AT threading → Tasks 1-2; rsvd RNG → Task 3; type stability → AT params; determinism → tolerances in Task 5). All covered.

**Placeholder scan:** every step has concrete code/commands; the Task 2 generic-ctor and adapt steps say "match the exact existing argument order" with the field list given in Task 2's struct — no "TBD"/"similar to". No bare "add error handling".

**Type consistency:** `MPSNode{T,AT}` and `TensorNetwork{T,AT}` field orders are fixed in Tasks 1-2 and reused verbatim in the generic ctor, ising ctor, and `adapt_structure`. The positional `MPSNode(mps, neighbor, cano, chi, cutoff, norm_method, svdopt, swapopt)` signature is defined in Task 1 and used in Task 4. `tsvd` returns `(U,S,V,discarded)` (4 values) and `raw2mps`/callers destructure 3 (Julia ignores the 4th) — consistent with the pre-GPU code. `rsvd` returns `(U,S,V)` (3 values). `contraction!` returns `(lnZ, error, psi)`.
