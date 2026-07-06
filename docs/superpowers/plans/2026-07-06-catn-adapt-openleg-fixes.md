# CATN.jl adapt-eltype + open-leg fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix three benchmark-discovered defects: make `adapt`/`cu` infer the moved element type (enabling `cu(tn)` and `Float32` GPU networks), support open-leg (partial) contraction (`result_tensor`), and correct the GPU docs.

**Architecture:** Small, targeted changes in `src/adapt.jl` (element-type inference), `src/tensor_network.jl` (sentinel-skipping bookkeeping, scalar-only fold, `result_tensor`), plus README. Each task is validated against the exact OMEinsum oracle (open-leg) or the CPU result (GPU Float32).

**Tech Stack:** Julia 1.12 (juliaup `/mnt/home/xgao1/.juliaup/bin/julia`), OMEinsum, Adapt, CUDA (test/gpu env), Test.

**Reference:** spec `docs/superpowers/specs/2026-07-06-catn-adapt-openleg-fixes-design.md`.

## Global Constraints

- Julia via the juliaup binary; never `module load julia`. CPU suite: `julia --project=. -e 'using Pkg; Pkg.test()'`. GPU tests: `julia test/gpu/run_gpu_tests.jl` (RTX 6000 available).
- The existing 217-test CPU suite must stay green (closed-network scalar contraction behavior is unchanged).
- `contraction!(tn) -> (lnZ, error, psi)` signature unchanged. Closed network result = `exp(lnZ)*psi`. Open network result = `result_tensor(tn) .* exp(lnZ) .* psi`.
- `adapt_structure` must infer the element type from the *moved* arrays (`NT = eltype(eltype(new_nodes[1].mps))`), not the source `T`.
- Only the generic `TensorNetwork` path; do not touch `BraKetNetwork`.
- Validate against `exact_contract` (test/exact.jl); do not loosen tolerances.
- Commit after each task with trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File Structure

```
src/adapt.jl            # Fix A: element-type-inferring adapt_structure(TensorNetwork)
src/tensor_network.jl   # Fix B: sentinel skip + scalar-only fold + result_tensor
src/CATN.jl             # export result_tensor
test/test_openleg.jl    # Fix B: open-leg contraction vs oracle (CPU)  [new]
test/test_gpu.jl        # Fix A: cu(tn) Float32 network test
test/runtests.jl        # include test_openleg.jl
README.md               # Fix C
```

---

### Task 1: Fix B — open-leg contraction + `result_tensor`

**Files:**
- Modify: `src/tensor_network.jl` (`count_remove_nodes!`, `count_add_nodes!`, `network_lognorm`; add `result_tensor`)
- Modify: `src/CATN.jl` (export `result_tensor`)
- Create: `test/test_openleg.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Produces: `result_tensor(tn::TensorNetwork) -> Array` — after `contraction!`, the surviving non-scalar (open-leg) node's `mps2raw`; a 0-d array holding `one(T)` if the network was fully closed. Full open result = `result_tensor(tn) .* exp(lnZ) .* psi`.
- Behavior change: `count_remove_nodes!`/`count_add_nodes!` skip sentinel (`≤0`) node ids; `network_lognorm` folds only scalar surviving nodes.

- [ ] **Step 1: Write the failing test** — create `test/test_openleg.jl`

```julia
using CATN
using CATN: TensorNetwork, contraction!, result_tensor
using OMEinsum, LinearAlgebra, Test
# exact_contract available from exact.jl (included earlier in runtests.jl)

@testset "open-leg (partial) contraction" begin
    # helper: full result of an open contraction
    function open_value(ts, ixs; kwargs...)
        tn = TensorNetwork(ts, ixs; kwargs...)
        lnZ, err, psi = contraction!(tn)
        return result_tensor(tn) .* (exp(lnZ) * psi)
    end

    @testset "chain with open physical legs (exact mode) vs oracle" begin
        # T1(p1,a) T2(a,p2,b) T3(b,p3): virtual bonds a,b contracted; p1,p2,p3 open
        for T in (Float64, ComplexF64)
            T1 = randn(T, 2, 4); T2 = randn(T, 4, 2, 4); T3 = randn(T, 4, 2)
            ts = [T1, T2, T3]; ixs = [[:p1,:a],[:a,:p2,:b],[:b,:p3]]
            ref = exact_contract(ts, ixs)                    # tensor over p1,p2,p3
            got = open_value(ts, ixs; Dmax=-1, chi=10_000)
            # compare up to leg permutation: same size-multiset and sorted values
            @test sort(collect(size(got))) == sort(collect(size(ref)))
            @test vec(sort(vec(abs.(got)))) ≈ vec(sort(vec(abs.(ref)))) rtol=1e-8
        end
    end

    @testset "closed network unchanged (result_tensor is scalar one)" begin
        A = randn(2,3); B = randn(3,4); C = randn(4,2)
        ixs = [[:a,:b],[:b,:c],[:c,:a]]
        ref = exact_contract([A,B,C], ixs)[]
        tn = TensorNetwork([A,B,C], ixs; Dmax=-1, chi=10_000)
        lnZ, err, psi = contraction!(tn)
        @test exp(lnZ)*psi ≈ ref rtol=1e-10             # closed API unchanged
        @test ndims(result_tensor(tn)) == 0             # 0-d one() for closed
        @test result_tensor(tn)[] * exp(lnZ) * psi ≈ ref rtol=1e-10
    end
end
```
(If the leg order turns out to be a fixed permutation, tighten to an exact `permutedims` comparison and document the order; the sorted-abs comparison is the correctness floor.)

- [ ] **Step 2: Run, expect failure**

Run: `julia --project=. -e 'include("test/runtests.jl")'` (after wiring the include)
Expected: FAIL — currently `contraction!` on the open-leg chain throws `KeyError` (sentinel id looked up as node); `result_tensor` undefined.

- [ ] **Step 3: Skip sentinels in bookkeeping** — in `src/tensor_network.jl`, at the top of `count_remove_nodes!`'s and `count_add_nodes!`'s per-node loop, add:
```julia
        node_id <= 0 && continue   # skip open-leg sentinels (not real nodes)
```
(Use the actual loop variable name in each function.)

- [ ] **Step 4: Fold only scalar nodes in `network_lognorm`** — make the per-node fold skip non-scalar (open-leg) surviving nodes. A node is scalar iff all its site physical dims are 1:
```julia
    isscalar(node) = all(size(s, 2) == 1 for s in node.mps)
    # in the sum: only add lognorm(node) for isscalar(node); skip others
```

- [ ] **Step 5: Add `result_tensor`** — in `src/tensor_network.jl`:
```julia
"""
    result_tensor(tn::TensorNetwork{T}) -> Array

After `contraction!`, return the surviving open-leg node's dense tensor (legs in the
node's final MPS order). For a fully-closed network (no open legs), returns a
0-dimensional array holding `one(T)`. The full open-network result is
`result_tensor(tn) .* exp(lnZ) .* psi`.
"""
function result_tensor(tn::TensorNetwork{T}) where {T}
    open_nodes = [node for node in values(tn.tensors)
                  if !all(size(s, 2) == 1 for s in node.mps)]
    isempty(open_nodes) && return fill(one(T))          # 0-d one() for closed networks
    length(open_nodes) == 1 || error("result_tensor: expected one surviving open-leg node, got $(length(open_nodes)) (disconnected/multi-open networks are unsupported)")
    return mps2raw(open_nodes[1])
end
```
Export `result_tensor` from `src/CATN.jl`.

- [ ] **Step 6: Wire the test** — add `include("test_openleg.jl")` to `test/runtests.jl` (inside the `@testset`, after `test_contraction.jl`).

- [ ] **Step 7: Run the full suite, expect pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS — open-leg chain (real & complex) matches the oracle; closed sanity holds; existing 217 tests unaffected. If an open contraction mismatches beyond leg permutation, debug systematically (leg order, the scalar-fold skip, sentinel skip) — do not loosen tolerances.

- [ ] **Step 8: Commit**

```bash
git add src/tensor_network.jl src/CATN.jl test/test_openleg.jl test/runtests.jl
git commit -m "feat: open-leg (partial) contraction via result_tensor; skip sentinels in bookkeeping"
```

---

### Task 2: Fix A — element-type-inferring `adapt` (+ GPU Float32 test)

**Files:**
- Modify: `src/adapt.jl` (`adapt_structure(to, tn::TensorNetwork)`)
- Modify: `test/test_gpu.jl` (add a `cu(tn)` Float32 test)

**Interfaces:**
- Consumes: `MPSNode.adapt_structure` (already eltype-inferring), `TensorNetwork`, `contraction!`.
- Produces: `adapt(CuArray, tn)` preserves `Float64`; `cu(tn)` yields `TensorNetwork{Float32,CuArray{Float32,3}}`.

- [ ] **Step 1: Write the failing test** — add to `test/test_gpu.jl`, inside the `@testset "GPU"` block (after the existing Float32 test):

```julia
        @testset "cu(tn) yields a Float32 GPU network and matches CPU" begin
            # closed Ising network (Float64) -> cu downcasts to Float32
            edges = [(1,2),(2,3),(3,1)]
            β = 0.3
            cpu = ising_network(3, edges, [0.5,-0.8,1.1], [0.2,-0.1,0.4], β; Dmax=-1, chi=10_000)
            g   = ising_network(3, edges, [0.5,-0.8,1.1], [0.2,-0.1,0.4], β; Dmax=-1, chi=10_000)
            lnZ_cpu, = contraction!(cpu)
            gtn = cu(g)
            @test gtn isa TensorNetwork{Float32, <:CuArray{Float32,3}}
            lnZ_g, = contraction!(gtn)
            @test Float64(real(lnZ_g)) ≈ real(lnZ_cpu) rtol=1e-4       # Float32 precision
            # adapt(CuArray, ·) preserves Float64
            g2 = ising_network(3, edges, [0.5,-0.8,1.1], [0.2,-0.1,0.4], β; Dmax=-1, chi=10_000)
            a2 = adapt(CuArray, g2)
            @test a2 isa TensorNetwork{Float64, <:CuArray{Float64,3}}
            lnZ_a, = contraction!(a2)
            @test real(lnZ_a) ≈ real(lnZ_cpu) rtol=1e-10
        end
```
(Ensure `Adapt`/`adapt` is in scope in test_gpu.jl — it uses `adapt` already; add `using Adapt` if needed. `cu` comes from CUDA.)

- [ ] **Step 2: Run, expect failure**

Run the GPU env: `julia test/gpu/run_gpu_tests.jl`
Expected: FAIL — `cu(g)` throws `TypeError` (`expected AT<:AbstractArray{Float64,3}, got CuArray{Float32,3}`).

- [ ] **Step 3: Fix `adapt_structure`** — in `src/adapt.jl`, replace the `TensorNetwork` method's type handling:
```julia
function Adapt.adapt_structure(to, tn::TensorNetwork)
    ks = sort(collect(keys(tn.tensors)))
    new_nodes = [adapt(to, tn.tensors[k]) for k in ks]
    @assert !isempty(new_nodes) "adapt_structure: cannot adapt an empty TensorNetwork"
    NAT = eltype(new_nodes[1].mps)
    NT  = eltype(NAT)                       # element type AFTER the move
    tensors = Dict{Int,MPSNode{NT,NAT}}(k => n for (k, n) in zip(ks, new_nodes))
    TensorNetwork{NT,NAT}(
        tensors, tn.Dmax, tn.chi, tn.cutoff, tn.norm_method, tn.select, tn.reverse,
        tn.svdopt, tn.swapopt, tn.compress, tn.cut_bond,
        deepcopy(tn.edge_count), NT(tn.lnZ), NT(tn.sign), NT(tn.psi),
        tn.maxdim_intermediate, tn.num_isolated, deepcopy(tn.rng), tn.n, tn.beta)
end
```
(Drop the `{T}` from the signature since `T` is no longer used; infer everything from the moved nodes.)

- [ ] **Step 4: Run the GPU test, expect pass**

Run: `julia test/gpu/run_gpu_tests.jl`
Expected: PASS — `cu(g)` is a `Float32` GPU network matching CPU within `1e-4`; `adapt(CuArray, ·)` stays `Float64` matching within `1e-10`.

- [ ] **Step 5: Run the CPU suite, confirm no regression**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS — the existing `adapt(Array, tn)` round-trip test (`test/test_adapt.jl`) still passes (identity move: `NT == T`).

- [ ] **Step 6: Commit**

```bash
git add src/adapt.jl test/test_gpu.jl
git commit -m "fix: adapt infers element type from moved arrays (cu(tn) -> Float32 GPU network)"
```

---

### Task 3: Fix C — README GPU docs

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the GPU section** — correct the device-move guidance and add the benchmark-derived note. Replace the GPU usage lines with:

````markdown
### GPU

Move a network to the device and contract there — contractions run via OMEinsum's CUDA
extension and `svd`/`qr` via cuSOLVER:

```julia
using CATN, CUDA
tn = ising_network(n, edges, w, h, β; Dmax=64, chi=256)
lnZ, err, psi = contraction!(adapt(CuArray, tn))   # keeps Float64
# or, for the faster Float32 path:
lnZ, err, psi = contraction!(cu(tn))               # cu downcasts to Float32
```

`adapt(CuArray, tn)` preserves the element type (`Float64`); `cu(tn)` produces a `Float32`
network. **The GPU only pays off for large, dense, `Float32` contractions** (crossover around
bond dimension ~256, ~2.5–3× by ~1024 on an RTX 6000); for Ising / modest bond dimensions the
CPU is faster because the contraction is many small tensor operations. CUDA is a test-only
dependency (see the `test/gpu` environment); the core package has no CUDA dependency.
````

- [ ] **Step 2: Add a `result_tensor` note** — in the Usage or Limitations section, add:
```markdown
For a network with open legs (a partial contraction), `contraction!` contracts all internal
bonds and the result is a tensor: `result_tensor(tn) .* exp(lnZ) .* psi`. Fully-closed networks
return the scalar `exp(lnZ) * psi`.
```

- [ ] **Step 3: Verify + commit** — sanity-check the package still loads (`julia --project=. -e 'using CATN'`), then:
```bash
git add README.md
git commit -m "docs: correct GPU adapt/cu usage; document result_tensor for open-leg contraction"
```

---

## Self-Review

**Spec coverage:** §2 Fix A → Task 2; §3.1 sentinel skip → Task 1 Step 3; §3.2 scalar-only fold + result_tensor → Task 1 Steps 4-5; §3.3 leg order → Task 1 test (up-to-permutation); §4 README → Task 3; §5 testing (open-leg oracle real+complex + closed sanity → Task 1; GPU Float32 cu + Float64 adapt → Task 2; regression = full suite each task). All covered.

**Placeholder scan:** every step has concrete code/commands. The open-leg test uses a sorted-abs value comparison (correctness floor) with a note to tighten to a permutation if the order is fixed — not a placeholder. No "TBD"/"add error handling".

**Type/name consistency:** `result_tensor(tn) -> Array` defined in Task 1, used in its test; `contraction!` return `(lnZ,error,psi)` unchanged; `adapt_structure` infers `NT=eltype(NAT)` consistently; the open result formula `result_tensor(tn) .* exp(lnZ) .* psi` is used identically in spec, test, and README. GPU test asserts `TensorNetwork{Float32,<:CuArray{Float32,3}}` matching Fix A's output type.
