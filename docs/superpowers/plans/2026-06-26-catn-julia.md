# CATN.jl Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A faithful Julia port of the CATN algorithm (arXiv:1912.03014) — approximate contraction of arbitrary tensor networks via per-node MPS — with OMEinsum as the tensor-contraction backend, plus the Ising/graphical-model application layer.

**Architecture:** Two mutable structs mirroring the Python reference: `MPSNode{T}` (a network tensor stored as an MPS chain) and `TensorNetwork{T}` (the multigraph + the edge-by-edge contraction loop). Contractions route through OMEinsum; SVD/QR through LinearAlgebra. Validation is by exact contraction of the same network via OMEinsum.

**Tech Stack:** Julia 1.12 (juliaup at `/mnt/home/xgao1/.juliaup/bin/julia`), OMEinsum 0.9, LinearAlgebra, Random, Test.

**Reference:** Python in `/mnt/home/xgao1/project/tnmp/catn/` — primarily `mps_node_np.py`, `tn_np.py`, `lnz_np.py`, `npsvd.py`. Cited as `mps_node_np.py:LINE` below. The Python is the source of truth for any mechanical detail not spelled out here.

## Global Constraints

- Julia ≥ 1.10.10 (Project.toml compat); run with the juliaup binary, never `module load julia`.
- OMEinsum is the backend for all *contractions* (multi-leg einsum-shaped operations and the exact validator). Use the `ein"..."` macro for fixed rank-≤3 patterns and `einsum(EinCode(ixs, iy), tensors)` where ranks vary. SVD/QR use `LinearAlgebra`; plain 2-D matrix products may use `*`.
- 1-based indexing throughout. The reference is 0-based — translate index arithmetic carefully (`len÷2` thresholds, `idx+1`, etc.).
- SVD convention: a `tsvd` wrapper returns `(U, S, V)` with `A ≈ U * Diagonal(S) * V'` (matches `npsvd.svd`, which returns `V = Vh'`). Note `LinearAlgebra.svd` returns `Vt`, so `V = F.V` (not `F.Vt`).
- Memory layout: Julia is column-major; numpy is row-major. Do NOT transcribe numpy `reshape`/`transpose` literally. Use the column-major reshape conventions proven in Tasks 3/4 and route pure contractions through OMEinsum (layout-independent).
- Generic element type `T` (default `Float64`; `Complex` must work for the exact path).
- All mutating functions end in `!`. Run tests with `julia --project=. -e 'using Pkg; Pkg.test()'` or a targeted `include`.
- Commit after each task with the `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer.

## File Structure

```
src/
  CATN.jl            # module: usings, includes, exports
  linalg_utils.jl    # tsvd, rsvd, qr helpers
  mps_node.jl        # MPSNode struct + methods
  tensor_network.jl  # TensorNetwork struct, construction, contraction! loop, selection, cut_bondim!
  ising.jl           # ising_network, free_energy, magnetization, correlation
test/
  runtests.jl        # @testset wiring; includes the files below
  exact.jl           # exact_contract reference oracle (OMEinsum)
  test_linalg.jl
  test_mps_node.jl
  test_contraction.jl
  test_ising.jl
```

---

### Task 1: Package skeleton + linear-algebra utilities

**Files:**
- Modify: `Project.toml` (add `LinearAlgebra`, `Random` to `[deps]`)
- Create: `src/linalg_utils.jl`
- Modify: `src/CATN.jl`
- Create: `test/test_linalg.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Produces:
  - `tsvd(A::AbstractMatrix; cutoff::Real=1e-15, maxdim::Int=typemax(Int)) -> (U, S, V)` where `A ≈ U*Diagonal(S)*V'`, `S` real vector sorted descending, truncated to entries `> cutoff` and at most `maxdim`. If all singular values `≤ cutoff`, keep the single largest.
  - `rsvd(A::AbstractMatrix, k::Int, oversample::Int=10, power::Int=10; rng=Random.default_rng()) -> (U, S, V)` — randomized SVD, port of `npsvd.py:14`.

- [ ] **Step 1: Add deps to Project.toml**

Under `[deps]` (keep the existing `OMEinsum` line), add:
```toml
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
```
Under `[compat]` add `LinearAlgebra` and `Random` are stdlibs — leave as-is (no compat entry needed beyond julia).

- [ ] **Step 2: Write the failing test** — `test/test_linalg.jl`

```julia
using CATN: tsvd, rsvd
using LinearAlgebra, Random, Test

@testset "linalg_utils" begin
    A = randn(8, 5)
    U, S, V = tsvd(A)
    @test U * Diagonal(S) * V' ≈ A
    @test issorted(S, rev=true)

    # maxdim truncation keeps the leading subspace
    U2, S2, V2 = tsvd(A; maxdim=2)
    @test length(S2) == 2
    @test S2 ≈ S[1:2]

    # cutoff drops tiny singular values
    B = U * Diagonal([1.0, 1e-3, 1e-20]) * V[:, 1:3]'
    _, S3, _ = tsvd(B; cutoff=1e-10)
    @test length(S3) == 2

    # rsvd approximates the leading singular triple of a low-rank matrix
    Random.seed!(1)
    L = randn(50, 4) * randn(4, 40)      # rank 4
    Ur, Sr, Vr = rsvd(L, 4, 10, 10; rng=MersenneTwister(0))
    @test Ur * Diagonal(Sr) * Vr' ≈ L atol=1e-8
end
```

- [ ] **Step 3: Run the test, expect failure**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL — `tsvd` not defined / `CATN` has no such binding.

- [ ] **Step 4: Implement `src/linalg_utils.jl`**

```julia
using LinearAlgebra
using Random

"""
    tsvd(A; cutoff=1e-15, maxdim=typemax(Int)) -> (U, S, V)

Truncated thin SVD with `A ≈ U * Diagonal(S) * V'`. Falls back to the QR-iteration
algorithm if the divide-and-conquer driver fails (cf. npsvd.py gesvd fallback).
"""
function tsvd(A::AbstractMatrix; cutoff::Real=1e-15, maxdim::Int=typemax(Int))
    F = try
        svd(A)
    catch
        svd(A; alg=LinearAlgebra.QRIteration())
    end
    S = F.S
    nkeep = count(>(cutoff), S)
    nkeep = nkeep == 0 ? 1 : min(nkeep, maxdim, length(S))
    return F.U[:, 1:nkeep], S[1:nkeep], F.V[:, 1:nkeep]
end

"""
    rsvd(A, k, oversample=10, power=10; rng) -> (U, S, V)

Randomized SVD (port of npsvd.py:rsvd) with `A ≈ U * Diagonal(S) * V'`.
"""
function rsvd(A::AbstractMatrix{T}, k::Int, oversample::Int=10, power::Int=10;
             rng::AbstractRNG=Random.default_rng()) where {T}
    m, n = size(A)
    p = min(n, oversample * k)
    Y = A * randn(rng, T, n, p)
    for _ in 1:power
        Y = A * (A' * Y)
    end
    Q = Matrix(qr(Y).Q)
    B = Q' * A
    F = svd(B)
    kk = min(k, size(F.U, 2))
    return (Q * F.U)[:, 1:kk], F.S[1:kk], F.V[:, 1:kk]
end
```

- [ ] **Step 5: Wire `src/CATN.jl`**

```julia
module CATN

using OMEinsum
using LinearAlgebra
using Random

include("linalg_utils.jl")

export tsvd, rsvd

end # module
```

- [ ] **Step 6: Wire `test/runtests.jl`**

```julia
using CATN
using Test

@testset "CATN.jl" begin
    include("test_linalg.jl")
end
```

- [ ] **Step 7: Run tests, expect pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS — `linalg_utils` testset green.

- [ ] **Step 8: Commit**

```bash
git add Project.toml src/CATN.jl src/linalg_utils.jl test/test_linalg.jl test/runtests.jl
git commit -m "feat: linalg utilities (tsvd, rsvd)"
```

---

### Task 2: Exact-contraction reference oracle

**Files:**
- Create: `test/exact.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Produces: `exact_contract(tensors::Vector{<:AbstractArray}, ixs::Vector{<:AbstractVector}) -> Array` — contracts the whole network exactly via OMEinsum; output legs are labels appearing exactly once, in first-seen order; returns a 0-dim array when fully contracted (use `result[]` for the scalar). Test-only helper.

- [ ] **Step 1: Write the failing test** — append to `test/exact.jl`

```julia
using OMEinsum, Test

function exact_contract(tensors, ixs)
    # output = labels appearing exactly once, first-seen order
    counts = Dict{Any,Int}()
    order = Any[]
    for ix in ixs, l in ix
        haskey(counts, l) || push!(order, l)
        counts[l] = get(counts, l, 0) + 1
    end
    iy = [l for l in order if counts[l] == 1]
    code = EinCode([collect(ix) for ix in ixs], iy)
    sd = OMEinsum.get_size_dict([collect(ix) for ix in ixs], tensors)
    opt = optimize_code(code, sd, GreedyMethod())
    return opt(tensors...)
end

@testset "exact oracle" begin
    A = randn(2, 3); B = randn(3, 4); C = randn(4, 2)
    # trace(A*B*C): labels a,b,c each appear twice -> scalar
    r = exact_contract([A, B, C], [[:a,:b],[:b,:c],[:c,:a]])
    @test r[] ≈ tr(A * B * C)
end
```
(Add `using LinearAlgebra` at the top for `tr`.)

- [ ] **Step 2: Run, expect failure** (file not included yet)

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL — `exact oracle` testset absent.

- [ ] **Step 3: Wire into `test/runtests.jl`**

Add `include("exact.jl")` as the first include inside the `@testset` (other tests will reuse `exact_contract`).

- [ ] **Step 4: Run, expect pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add test/exact.jl test/runtests.jl
git commit -m "test: exact-contraction reference oracle via OMEinsum"
```

---

### Task 3: `MPSNode` struct + `raw2mps` + `mps2raw`

**Files:**
- Create: `src/mps_node.jl`
- Modify: `src/CATN.jl` (include + exports)
- Create: `test/test_mps_node.jl`
- Modify: `test/runtests.jl`

**Reference:** `mps_node_np.py:7` (struct), `:33` (`raw2mps`), `:58` (`mps2raw`).

**Interfaces:**
- Produces:
  - `mutable struct MPSNode{T}` with fields `mps::Vector{Array{T,3}}`, `neighbor::Vector{Int}`, `cano::Int`, `chi::Int`, `cutoff::Float64`, `norm_method::Int`, `svdopt::Bool`, `swapopt::Bool`.
  - `MPSNode(tensor::AbstractArray{T}, neighbor::Vector{Int}; chi=32, cutoff=1e-15, norm_method=1, svdopt=true, swapopt=true)` — builds via `raw2mps`, sets `cano = order` (left-canonical).
  - `raw2mps(tensor, chi, cutoff) -> Vector{Array{T,3}}`
  - `mps2raw(node::MPSNode) -> Array{T}`
  - `order(node) -> Int` (= `length(node.mps)`), `shape(node) -> Vector{Int}` (physical dims; `[1]` if single site).

**Column-major reshape facts (use these, do not transcribe numpy):**
- `reshape(T, d1, :)` puts axis-1 index in the rows (axis 1 is fastest in column-major), so it cleanly peels the *first* physical index — exactly what left-to-right MPS construction needs.
- After SVD `M = U*Diagonal(S)*V'` with `M = reshape(R, dleft*d_i, :)`, `reshape(U, dleft, d_i, χ)` correctly splits rows back to `(dleft, d_i)`; `reshape(Diagonal(S)*V', χ, dims[i+1:end]...)` correctly forms the next residual.

- [ ] **Step 1: Write the failing test** — `test/test_mps_node.jl`

```julia
using CATN: MPSNode, mps2raw, raw2mps, order, shape
using LinearAlgebra, Test

@testset "raw2mps/mps2raw round-trip" begin
    for dims in [(4,), (3, 5), (2, 3, 4), (2, 3, 2, 3)]
        T = randn(dims...)
        nb = collect(1:length(dims))
        node = MPSNode(T, nb; chi=1000, cutoff=1e-15)
        @test order(node) == length(dims)
        @test shape(node) == [dims...] || (length(dims) == 1 && shape(node) == [dims[1]])
        @test mps2raw(node) ≈ T
    end

    # chi truncation on a genuinely low-rank tensor is (near) lossless
    U = randn(6, 2); V = randn(2, 6)
    M = U * V                      # rank 2 matrix, viewed as order-2 tensor
    node = MPSNode(M, [1, 2]; chi=2, cutoff=1e-15)
    @test mps2raw(node) ≈ M atol=1e-10
end
```

- [ ] **Step 2: Run, expect failure**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL — `MPSNode` not defined.

- [ ] **Step 3: Implement `src/mps_node.jl`**

```julia
mutable struct MPSNode{T}
    mps::Vector{Array{T,3}}
    neighbor::Vector{Int}
    cano::Int
    chi::Int
    cutoff::Float64
    norm_method::Int
    svdopt::Bool
    swapopt::Bool
end

function MPSNode(tensor::AbstractArray{T}, neighbor::Vector{Int};
                chi::Int=32, cutoff::Float64=1e-15, norm_method::Int=1,
                svdopt::Bool=true, swapopt::Bool=true) where {T}
    mps = raw2mps(tensor, chi, cutoff)
    cano = length(mps)          # left-canonical: center at last site
    MPSNode{T}(mps, copy(neighbor), cano, chi, cutoff, norm_method, svdopt, swapopt)
end

function raw2mps(tensor::AbstractArray{T}, chi::Int, cutoff::Float64) where {T}
    nd = ndims(tensor)
    nd == 0 && return Array{T,3}[]
    dims = size(tensor)
    nd == 1 && return Array{T,3}[reshape(Array(tensor), 1, dims[1], 1)]
    mps = Array{T,3}[]
    R = reshape(Array(tensor), 1, dims...)      # (1, dims...)
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

function mps2raw(node::MPSNode{T}) where {T}
    mps = node.mps
    isempty(mps) && return Array{T,0}(undef)
    A = mps[1]
    cur = reshape(A, size(A, 2), size(A, 3))    # (p1, r) since left bond = 1
    pdim = size(A, 2)
    physdims = Int[size(A, 2)]
    for k in 2:length(mps)
        B = mps[k]
        r, p, r2 = size(B)
        Bmat = reshape(B, r, p * r2)
        res = ein"ab,bc->ac"(cur, Bmat)         # OMEinsum backend
        res3 = reshape(res, pdim, p, r2)
        pdim *= p
        cur = reshape(res3, pdim, r2)
        push!(physdims, p)
    end
    return reshape(cur, physdims...)            # right bond = 1 collapses
end

order(node::MPSNode) = length(node.mps)

function shape(node::MPSNode)
    length(node.mps) == 1 && return [1]
    return [size(t, 2) for t in node.mps]
end
```
Note: the `shape` rule mirrors `mps_node_np.py:345` (a single-site MPS reports `[1]`). The round-trip test's `shape` assertion tolerates this.

- [ ] **Step 4: Update `src/CATN.jl`**

Add `include("mps_node.jl")` after the linalg include and export `MPSNode, raw2mps, mps2raw, order, shape`.

- [ ] **Step 5: Wire `test/runtests.jl`** — add `include("test_mps_node.jl")`.

- [ ] **Step 6: Run, expect pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS — round-trip green.

- [ ] **Step 7: Commit**

```bash
git add src/mps_node.jl src/CATN.jl test/test_mps_node.jl test/runtests.jl
git commit -m "feat: MPSNode with raw2mps/mps2raw round-trip"
```

---

### Task 4: Canonicalization — `cano_to!`, `left_canonical!`

**Files:**
- Modify: `src/mps_node.jl`
- Modify: `test/test_mps_node.jl`

**Reference:** `mps_node_np.py:110` (`cano_to`), `:161` (`left_canonical`).

**Interfaces:**
- Produces:
  - `cano_to!(node, idx::Int)` — move the orthogonality center to site `idx` (`idx == 0` means "last site", mirroring the reference's `-1`). Uses `tsvd`; updates `node.cano`. Preserves the represented dense tensor.
  - `left_canonical!(node)` — set `cano = 1` then `cano_to!(node, length(mps))`.

**1-based translation of `cano_to` (reference `mps_node_np.py:110-159`):** sweeping right (`cano < idx`), for each site `i` from `cano` to `idx-1`: reshape `mps[i]` `(dl, d, dr)` to `(dl*d, dr)`, `tsvd`, set `mps[i] = reshape(U, dl, d, :)`, fold `R = Diagonal(S)*V'` into `mps[i+1]` via `ein"ij,jab->iab"(R, mps[i+1])`, set `cano = i+1`. Sweeping left is the mirror: reshape `(dl, d, dr)` transposed to `(d*dr, dl)` — in Julia use `reshape(permutedims(mps[i], (2,3,1)), d*dr, dl)`, `tsvd`, `mps[i] = permutedims(reshape(U, d, dr, :), (3,1,2))` giving `(:, d, dr)`, and fold into `mps[i-1]` via `ein"abc,cd->abd"(mps[i-1], (Diagonal(S)*V')')`. Match the reference's truncation: keep `S .> cutoff`, but if none, keep all of `U`'s columns (`mps_node_np.py:127-132`).

- [ ] **Step 1: Write the failing test** — append to `test/test_mps_node.jl`

```julia
using CATN: cano_to!, left_canonical!

@testset "canonicalization preserves tensor" begin
    T = randn(2, 3, 4, 2)
    node = MPSNode(T, [1,2,3,4]; chi=1000)
    ref = mps2raw(node)
    for idx in [1, 2, 3, 4, 0]
        cano_to!(node, idx)
        @test mps2raw(node) ≈ ref
    end
    left_canonical!(node)
    @test node.cano == 4
    @test mps2raw(node) ≈ ref
    # left-isometry of all but the center after left_canonical!
    for i in 1:3
        A = node.mps[i]; dl, d, dr = size(A)
        M = reshape(A, dl*d, dr)
        @test M' * M ≈ I atol=1e-8
    end
end
```

- [ ] **Step 2: Run, expect failure**

Run: `julia --project=. -e 'include("test/runtests.jl")'`
Expected: FAIL — `cano_to!` not defined.

- [ ] **Step 3: Implement** `cano_to!` and `left_canonical!` in `src/mps_node.jl` following the 1-based translation above. Use this skeleton (fill both sweep directions):

```julia
function cano_to!(node::MPSNode, idx::Int)
    idx == 0 && (idx = length(node.mps))
    node.cano == idx && return node
    if node.cano < idx
        for i in node.cano:idx-1
            dl, d, dr = size(node.mps[i])
            U, S, V = tsvd(reshape(node.mps[i], dl*d, :); cutoff=node.cutoff)
            node.mps[i] = reshape(U, dl, d, :)
            R = Diagonal(S) * V'
            node.mps[i+1] = ein"ij,jab->iab"(R, node.mps[i+1])
            node.cano = i + 1
        end
    else
        for i in node.cano:-1:idx+1
            dl, d, dr = size(node.mps[i])
            Mt = reshape(permutedims(node.mps[i], (2, 3, 1)), d*dr, dl)
            U, S, V = tsvd(Mt; cutoff=node.cutoff)
            node.mps[i] = permutedims(reshape(U, d, dr, :), (3, 1, 2))
            R = Diagonal(S) * V'                     # (χ, dl)
            node.mps[i-1] = ein"abc,cd->abd"(node.mps[i-1], permutedims(R, (2,1)))
            node.cano = i - 1
        end
    end
    return node
end

left_canonical!(node::MPSNode) = (node.cano = 1; cano_to!(node, length(node.mps)))
```
Note: when `tsvd`'s cutoff would drop all values, fall back to keeping `U`'s full column count (guard `nkeep` as in `tsvd`; the reference does this at `mps_node_np.py:127-132`).

- [ ] **Step 4: Export** `cano_to!, left_canonical!` from `src/CATN.jl`.

- [ ] **Step 5: Run, expect pass**

Run: `julia --project=. -e 'include("test/runtests.jl")'`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mps_node.jl src/CATN.jl test/test_mps_node.jl
git commit -m "feat: MPS canonicalization (cano_to!, left_canonical!)"
```

---

### Task 5: `swap!`, move helpers, `reverse!`, neighbor/query helpers

**Files:**
- Modify: `src/mps_node.jl`
- Modify: `test/test_mps_node.jl`

**Reference:** `mps_node_np.py:287` (`swap`), `:69` (`move2tail`), `:87` (`move`), `:402` (`move2head`), `:489` (`reverse`), `:22` (`find_neighbor`), `:500` (`add_neighbor`), `:506` (`delete_neighbor`), `:378` (`logdim`), `:511` (`lognorm`), `:521` (`clear`).

**Interfaces:**
- Produces:
  - `swap!(node, i, j)` — swap adjacent sites `i,j` (`|i-j|==1`), truncate virtual bond to `chi`, keep canonical center at `j`; uses `rsvd` when `swapopt && ((rows>7000 && cols>7000) || rows>20000 || cols>20000)`. Returns truncation error (Float).
  - `move!(node, a, b)`, `move2tail!(node, idx)`, `move2head!(node, idx)` — reposition site by chained `swap!`.
  - `reverse!(node)` — reverse site order: `node.mps = [permutedims(t,(3,2,1)) for t in reverse(node.mps)]`, `reverse!(node.neighbor)`, `node.cano = length(mps)+1-node.cano`.
  - `find_neighbor(node, j) -> Int` (0 if absent — reference returns -1; use 0 for 1-based "not found", document it), `add_neighbor!(node, n, pos=0)`, `delete_neighbor!(node, n) -> Int` (returns the deleted 1-based position).
  - `logdim(node) -> Float64`, `logdim(node, idx) -> Float64`, `lognorm(node) -> (Float64, sign)`, `clear!(node)`.

**Swap index math (reference `mps_node_np.py:287-343`).** For `i<j` with `tl=mps[i] (d0,?,dd)`, `tr=mps[j] (dd,?,d3)`: numpy forms `einsum("ijk,kab->iajb", tl, tr).reshape(d0*d1, d2*d3)` where `d1=tr.shape[1]` (the physical that moves left), `d2=tl.shape[1]` (moves right). In Julia, build the swapped 4-tensor with OMEinsum then merge legs with explicit ordering:
```julia
d0 = size(tl,1); d1 = size(tr,2); d2 = size(tl,2); d3 = size(tr,3)
W = ein"ijk,kab->iajb"(tl, tr)             # (d0, d1, d2, d3)  -- physical legs swapped
M = reshape(W, d0*d1, d2*d3)               # column-major: (i,a) rows, (j,b) cols
U, S, V = (use rsvd or tsvd)               # truncate to chi
# going right (i<j): center ends at j
node.mps[i] = reshape(U, d0, d1, :)
node.mps[j] = reshape(Diagonal(S)*V', :, d2, d3)
node.cano = j
```
For `i>j` (going left) mirror per `mps_node_np.py:338-341`: `U = U*Diagonal(S)` folded into the lower-index site, the other gets `reshape(V', ...)`; center ends at `j`. Keep the canonical pre-move (`if cano ∉ {i,j}: cano_to!(node, nearest)`), `mps_node_np.py:299-300`.

- [ ] **Step 1: Write the failing test** — append to `test/test_mps_node.jl`

```julia
using CATN: swap!, reverse!, move2tail!, move2head!, find_neighbor,
            add_neighbor!, delete_neighbor!

@testset "swap/reverse preserve tensor (up to leg permutation)" begin
    T = randn(2, 3, 4, 5)
    node = MPSNode(T, [10,20,30,40]; chi=1000)
    swap!(node, 2, 3)                       # swap legs 2 and 3
    @test node.neighbor == [10,30,20,40]
    @test mps2raw(node) ≈ permutedims(T, (1,3,2,4))

    node2 = MPSNode(T, [10,20,30,40]; chi=1000)
    reverse!(node2)
    @test node2.neighbor == [40,30,20,10]
    @test mps2raw(node2) ≈ permutedims(T, (4,3,2,1))

    node3 = MPSNode(T, [10,20,30,40]; chi=1000)
    move2tail!(node3, 1)                    # move first leg to the end
    @test node3.neighbor[end] == 10
    @test mps2raw(node3) ≈ permutedims(T, (2,3,4,1))
end

@testset "neighbor helpers" begin
    node = MPSNode(randn(2,2,2), [5,6,7])
    @test find_neighbor(node, 6) == 2
    @test find_neighbor(node, 99) == 0
    delete_neighbor!(node, 6)
    @test node.neighbor == [5,7]
end
```
(For `move2tail!`, note the reference rearranges only the MPS, not `neighbor`; the test re-implements the expected neighbor order by also moving the neighbor entry — implement `move2tail!` to move the site only, and have the *caller* manage neighbors as the reference does. To make this test self-consistent, have the test move the neighbor manually OR add a `move2tail!` that also rotates `neighbor`. Decision: `swap!` rotates both `mps` and `neighbor` together so `mps2raw` legs and `neighbor` stay aligned; verify against the reference, which arranges neighbors separately in `tn_np`. Keep `swap!` swapping both, and document that `tensor_network.jl` must not double-swap neighbors.)

- [ ] **Step 2: Run, expect failure.** Run: `julia --project=. -e 'include("test/runtests.jl")'` — FAIL (`swap!` undefined).

- [ ] **Step 3: Implement** `swap!`, `move!`, `move2tail!`, `move2head!`, `reverse!`, and the neighbor/query helpers in `src/mps_node.jl` per the reference and the swap index math above. Have `swap!` swap the two `neighbor` entries in lockstep with the two sites. `find_neighbor` returns `0` when absent. `logdim(node) = sum(log2(size(t,2)) for t in mps)`, `logdim(node, idx) = log2(size(mps[idx],2))`.

- [ ] **Step 4: Export** the new names from `src/CATN.jl`.

- [ ] **Step 5: Run, expect pass.** Run: `julia --project=. -e 'include("test/runtests.jl")'` — PASS.

- [ ] **Step 6: Commit**

```bash
git add src/mps_node.jl src/CATN.jl test/test_mps_node.jl
git commit -m "feat: MPS swap!, move helpers, reverse!, neighbor helpers"
```

---

### Task 6: `merge!`

**Files:**
- Modify: `src/mps_node.jl`
- Modify: `test/test_mps_node.jl`

**Reference:** `mps_node_np.py:354` (`merge`).

**Interfaces:**
- Produces: `merge!(node, j::Int; cross::Bool=false) -> Float64` — find the two sites whose `neighbor == j`, bring them adjacent (`move!` site 2 to `idx1+1` if `!cross` else `idx1`), fuse them into one site via `ein"ijk,kab->ijab"` then reshape `(dl, d*d', dr)`, drop the second neighbor entry, recanonicalize to `idx1`. Returns accumulated error.

- [ ] **Step 1: Write the failing test** — append to `test/test_mps_node.jl`

```julia
using CATN: merge!

@testset "merge! fuses duplicate-neighbor legs" begin
    # order-3 tensor, legs 1 and 3 both point to neighbor 7 (a duplicate)
    T = randn(2, 5, 3)
    node = MPSNode(T, [7, 8, 7]; chi=1000)
    merge!(node, 7)                 # fuse legs to neighbor 7
    @test count(==(7), node.neighbor) == 1
    # remaining represented tensor: legs (7-fused, 8). Reference brings the two
    # 7-legs adjacent (idx1, idx1+1) and fuses with the first kept in place.
    raw = mps2raw(node)
    # cross=false default fuses leg3 next to leg1: combined index runs (i1 fast, i3)
    expected = reshape(permutedims(T, (1,3,2)), 2*3, 5)
    @test reshape(raw, size(raw,1), size(raw,2)) ≈ expected
end
```
(If the fused-index ordering differs, adjust `expected`'s `permutedims`/`reshape` to match the implementation; the invariant that matters is that `merge!` followed by `cut_bondim!` reproduces exact-network values, verified in Task 11/12.)

- [ ] **Step 2: Run, expect failure.** FAIL — `merge!` undefined.

- [ ] **Step 3: Implement** `merge!` per `mps_node_np.py:354-376`, 1-based. Use `findall(==(j), node.neighbor)` to get the two positions; `deleteat!(node.neighbor, idx2)`; `move!(node, idx2, cross ? idx1 : idx1+1)`; `cano_to!(node, idx1)`; fuse `node.mps[idx1] = reshape(ein"ijk,kab->ijab"(node.mps[idx1], node.mps[idx1+1]), dl, :, dr2)`; `deleteat!(node.mps, idx1+1)`; `cano_to!(node, idx1)`.

- [ ] **Step 4: Export** `merge!`.

- [ ] **Step 5: Run, expect pass.**

- [ ] **Step 6: Commit**

```bash
git add src/mps_node.jl src/CATN.jl test/test_mps_node.jl
git commit -m "feat: MPS merge! for duplicate-neighbor legs"
```

---

### Task 7: `compress!` / `compress_opt!`

**Files:**
- Modify: `src/mps_node.jl`
- Modify: `test/test_mps_node.jl`

**Reference:** `mps_node_np.py:165` (`compress`), `:202` (`compress_opt`).

**Interfaces:**
- Produces:
  - `compress!(node) -> Float64` — `left_canonical!`, then sweep `j` from last down to 2: two-site fuse `ein"ijk,kab->ijab"(mps[j-1], mps[j])` reshaped `(d0*d1, d2*d3)`, `tsvd` to `chi`, write back `mps[j-1]=reshape(U*Diagonal(S), d0,d1,:)`, `mps[j]=reshape(V', :, d2, d3)`; set `cano=1`. Returns error.
  - `compress_opt!(node) -> Float64` — QR-before-SVD variant (`mps_node_np.py:202-283`): `flag_left` when `matl` rows > cols, `flag_right` when `matr` rows < cols; otherwise identical.

- [ ] **Step 1: Write the failing test** — append to `test/test_mps_node.jl`

```julia
using CATN: compress!, compress_opt!

@testset "compress preserves tensor" begin
    T = randn(2,3,4,3,2)
    for f! in (compress!, compress_opt!)
        node = MPSNode(T, collect(1:5); chi=1000)
        ref = mps2raw(node)
        f!(node)
        @test node.cano == 1
        @test mps2raw(node) ≈ ref
    end
    # compress removes inflated bonds on a low-rank chain
    A = randn(3,2); M = kron(A, A')         # contrived low-rank-ish
    node = MPSNode(reshape(M, 3,2,2,3), collect(1:4); chi=1000)
    pre = maximum(size(t,3) for t in node.mps[1:end-1])
    compress!(node)
    @test mps2raw(node) ≈ reshape(M, 3,2,2,3)
end
```

- [ ] **Step 2–6:** Run→fail; implement both per the reference (column-major reshapes as established; `ein` for the two-site fuse; `tsvd` for truncation; `qr` for the opt variant via `LinearAlgebra.qr`, taking `Matrix(F.Q)` and `F.R`); export; run→pass; commit:

```bash
git add src/mps_node.jl src/CATN.jl test/test_mps_node.jl
git commit -m "feat: MPS compress!/compress_opt!"
```

---

### Task 8: `eat!`

**Files:**
- Modify: `src/mps_node.jl`
- Modify: `test/test_mps_node.jl`

**Reference:** `mps_node_np.py:407` (`eat`).

**Interfaces:**
- Produces: `eat!(node, nodej::MPSNode, idx::Int, idxi::Int) -> (lognorm::Float64, error::Float64, phase)` — contract physical leg `idx` of `node` with leg `idxi` of `nodej`, append `nodej`'s remaining sites to `node`. Three cases (reference): (a) both single-site leaves → scalar dot, sets `node.mps=[]`, returns `(log|r|, 0, r/|r|)`; (b) `nodej` single-site leaf → fold into `node`'s tail-1 site; (c) general → `move2tail!(node, idx)`, `move2head!(nodej, idxi)`, contract the boundary physical legs as a matrix product, append the rest, normalize per `norm_method`, accumulate `log(norm)`.

**Boundary contraction (case c):** after `move2tail!(node, idx)`, `mati = reshape(node.mps[end], size(...,1), size(...,2))` (right bond is 1). After `move2head!(nodej, idxi)`, `matj = reshape(nodej.mps[1], size(...,2), size(...,3))` (left bond is 1). `mat = ein"ab,bc->ac"(mati, matj)`. Fold into the new tail: `node.mps[end-1] = ein"ijk,ka->ija"(node.mps[end-1], mat)`; `pop!(node.mps)`; then `append!(node.mps, nodej.mps[2:end])`; `cano_to!(node, length(mps))`; normalize the center site; return `(log(norm), error, 1)`. Mirror cases (a)/(b) exactly from the reference.

- [ ] **Step 1: Write the failing test** — append to `test/test_mps_node.jl`

```julia
using CATN: eat!

@testset "eat! equals direct contraction" begin
    # general case: node i (legs a,b,c -> neighbors 1,2,3), node j (legs c',d -> 3,4)
    Ti = randn(2,3,4); Tj = randn(4,5)
    nodei = MPSNode(Ti, [1,2,3]; chi=1000, norm_method=0)
    nodej = MPSNode(Tj, [3,4]; chi=1000, norm_method=0)
    idx  = find_neighbor(nodei, 3)     # = 3
    idxi = find_neighbor(nodej, 3)     # = 1
    lognorm, err, phase = eat!(nodei, nodej, idx, idxi)
    @test nodei.neighbor == sort(nodei.neighbor)  # legs now 1,2,4 (order per reference)
    raw = mps2raw(nodei) .* exp(lognorm)
    expected = ein"abc,cd->abd"(Ti, Tj)           # legs a,b,d
    @test sort(collect(size(raw))) == sort(collect(size(expected)))
    @test vec(sort(vec(raw))) ≈ vec(sort(vec(expected))) atol=1e-8
end

@testset "eat! both leaves -> scalar" begin
    u = randn(3); v = randn(3)
    ni = MPSNode(u, [2]; norm_method=0); nj = MPSNode(v, [1]; norm_method=0)
    lognorm, err, phase = eat!(ni, nj, 1, 1)
    @test isempty(ni.mps)
    @test exp(lognorm) * phase ≈ dot(u, v)
end
```
(The element-multiset comparison avoids depending on the exact final leg order; Tasks 11–12 pin down ordering through the full pipeline.)

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Implement** `eat!` per `mps_node_np.py:407-487`, 1-based, with `norm_method` ∈ {0,1,2} (0→`norm=1`, 1→`norm(reshape(center,:),2)`, 2→`maximum(abs, center)`). Guard `norm ≤ cutoff` returning `(0.0, error, 1)`.

- [ ] **Step 4: Export** `eat!`.

- [ ] **Step 5: Run, expect pass.**

- [ ] **Step 6: Commit**

```bash
git add src/mps_node.jl src/CATN.jl test/test_mps_node.jl
git commit -m "feat: MPS eat! (leg contraction + append)"
```

---

### Task 9: `TensorNetwork` struct + construction from OMEinsum labels

**Files:**
- Create: `src/tensor_network.jl`
- Modify: `src/CATN.jl`
- Create: `test/test_contraction.jl`
- Modify: `test/runtests.jl`

**Reference:** `tn_np.py:19` (struct/`__init__`), neighbor/graph setup; node construction here is from generic tensors (not the Ising `construct_tensor`, which is Task 13).

**Interfaces:**
- Produces:
  - `mutable struct TensorNetwork{T}` with fields: `tensors::Dict{Int,MPSNode{T}}`, `Dmax::Int`, `chi::Int`, `cutoff::Float64`, `norm_method::Int`, `select::Int`, `reverse::Bool`, `svdopt::Bool`, `swapopt::Bool`, `compress::Bool`, `cut_bond::Bool`, `edge_count::Dict{Int,Vector{Vector{Int}}}`, `lnZ::T`, `sign::T`, `psi::T`, `maxdim_intermediate::Int`, `num_isolated::Int`, `rng::AbstractRNG`.
  - `TensorNetwork(tensors::Vector{<:AbstractArray}, ixs::Vector{<:AbstractVector}; Dmax=32, chi=32, cutoff=1e-15, norm_method=1, select=1, reverse=true, svdopt=true, swapopt=true, compress=false, cut_bond=false, seed=1)` — builds one `MPSNode` per tensor; derives the multigraph from shared labels (a label on exactly two tensors → a bond; each node's `neighbor` lists the *other* node id per leg, ordered to match that tensor's leg order so it aligns with the MPS sites). A label appearing once → an open leg, represented by appending a size-`d` dangling site whose `neighbor` entry is a unique negative sentinel id (never selected for contraction). Rejects a label appearing >2 times or twice within one tensor.

**Leg→neighbor alignment:** for tensor `t` with labels `ixs[t]`, site `k` corresponds to label `ixs[t][k]`; its `neighbor[k]` is the *other* tensor id sharing that label (or a sentinel for open legs). This guarantees `mps2raw(node)` leg order matches `ixs[t]`.

- [ ] **Step 1: Write the failing test** — `test/test_contraction.jl`

```julia
using CATN: TensorNetwork, MPSNode, mps2raw, order
using Test
# exact_contract is available from exact.jl (included earlier in runtests.jl)

@testset "TensorNetwork construction" begin
    A = randn(2,3); B = randn(3,4); C = randn(4,2)
    ixs = [[:a,:b],[:b,:c],[:c,:a]]
    tn = TensorNetwork([A,B,C], ixs; chi=1000)
    @test length(tn.tensors) == 3
    @test mps2raw(tn.tensors[1]) ≈ A
    @test mps2raw(tn.tensors[2]) ≈ B
    # node 1 neighbors: leg :a -> node 3, leg :b -> node 2
    @test tn.tensors[1].neighbor == [3, 2]
    @test tn.num_isolated == 0
end
```

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Implement** `src/tensor_network.jl` — the struct and constructor. Map labels→(node,leg) occurrences; build each `MPSNode(tensors[t], neighbor_t; chi, cutoff, norm_method, svdopt, swapopt)`. Use integer node ids `1:length(tensors)`. Set `lnZ = zero(T)`, `sign = one(T)`, `psi = one(T)`, `maxdim_intermediate = -1`, `num_isolated = count(t -> order(node)==0 ...)` (nodes with no real bonds), `rng = MersenneTwister(seed)`. Do **not** build `edge_count` yet (Task 10).

- [ ] **Step 4: Update `src/CATN.jl`** — `include("tensor_network.jl")`, export `TensorNetwork`.

- [ ] **Step 5: Wire `test/runtests.jl`** — add `include("test_contraction.jl")`.

- [ ] **Step 6: Run, expect pass.**

- [ ] **Step 7: Commit**

```bash
git add src/tensor_network.jl src/CATN.jl test/test_contraction.jl test/runtests.jl
git commit -m "feat: TensorNetwork struct + construction from einsum labels"
```

---

### Task 10: Edge selection + `edge_count` bookkeeping

**Files:**
- Modify: `src/tensor_network.jl`
- Modify: `test/test_contraction.jl`

**Reference:** `tn_np.py:169` (`dim_after_merge`), `:178`/`:200`/`:206`/`:284` (selectors), `:235`–`:259` (`count_add_*`/`count_remove_*`/`select_edge_init`).

**Interfaces:**
- Produces:
  - `dim_after_merge(tn, i, j) -> Int` = `round(logdim(i) + logdim(j) - 2*logdim(node_i, leg_to_j))`.
  - `select_edge_init!(tn)` — populate `edge_count` from all current bonds.
  - `count_add_edges!`, `count_add_nodes!`, `count_remove_nodes!`.
  - `select_edge_min_dim(tn) -> (i,j)`, `select_edge_min_dim_triangle(tn) -> (i,j)`, `select_edge_sequentially(tn) -> (i,j)`.

- [ ] **Step 1: Write the failing test** — append to `test/test_contraction.jl`

```julia
using CATN: dim_after_merge, select_edge_init!, select_edge_min_dim

@testset "edge selection" begin
    A = randn(2,2); B = randn(2,2); C = randn(2,8); D = randn(8,2)
    # chain 1-2-3-4 with a fat bond between 3 and 4
    ixs = [[:a,:x],[:x,:y],[:y,:z],[:z,:a]]   # a loop 1-2-3-4-1
    tn = TensorNetwork([A,B,C,D], ixs; chi=1000)
    select_edge_init!(tn)
    i, j = select_edge_min_dim(tn)
    @test Set([i,j]) != Set([3,4])             # the fat bond is the costliest, not chosen first
end
```

- [ ] **Step 2–6:** Run→fail; implement the selectors and bookkeeping exactly per the reference (use a `Dict{Int,Vector{Vector{Int}}}` cost→`[i,j]` pairs, with sorted `[i,j]`; the triangle heuristic counts common neighbors as in `tn_np.py:206-232`); call `select_edge_init!` and ensure `select` chooses the right selector; export the names; run→pass; commit:

```bash
git add src/tensor_network.jl src/CATN.jl test/test_contraction.jl
git commit -m "feat: edge selection heuristics + edge_count bookkeeping"
```

---

### Task 11: `cut_bondim!` / `cut_bondim_opt!`

**Files:**
- Modify: `src/tensor_network.jl`
- Modify: `test/test_contraction.jl`

**Reference:** `tn_np.py:482` (`cut_bondim`), `:598` (`cut_bondim_opt`).

**Interfaces:**
- Produces:
  - `cut_bondim!(tn, i, idx_j_in_i) -> Float64` — SVD-truncate the shared physical bond between node `i` (leg `idx_j_in_i`) and its neighbor `j` to `Dmax` (skip when `Dmax < 0`). Returns discarded-weight error.
  - `cut_bondim_opt!(tn, i, idx_j_in_i) -> Float64` — canonicalize both nodes to that leg, QR-before-SVD variant.

**Index math (reference `tn_np.py:598-692`, 1-based):** let `j = neighbor_i[idx_j_in_i]`, `idx_i_in_j = find_neighbor(node_j, i)`. `cano_to!` both to those legs (opt variant). With `Ai = mps_i[idx_j_in_i]` shape `(da_l, d, da_r)`, `Aj = mps_j[idx_i_in_j]` shape `(db_l, d, db_r)`: form `mati = reshape(permutedims(Ai,(1,3,2)), da_l*da_r, d)`, `matj = reshape(permutedims(Aj,(1,3,2)), db_l*db_r, d)`. QR-reduce when rows>cols (`flag_left`/`flag_right`), `merged = ein"ab,cb->ac"(ri, rj)` (i.e. `ri*rj'`), `tsvd` to `Dmax`, split `mati=U*sqrt(S)`, `matj=V*sqrt(S)` (so `mati*matj' = U S V'`), re-apply Q factors, reshape back via `permutedims(reshape(mati, da_l, da_r, :),(1,3,2))`. Store back into both nodes.

- [ ] **Step 1: Write the failing test** — append to `test/test_contraction.jl`

```julia
using CATN: cut_bondim!, cut_bondim_opt!

@testset "cut_bondim is exact when bond is low-rank" begin
    # bond between two order-2 nodes is rank 2 but stored as dim 4
    P = randn(3,4); Qm = randn(4,3)
    # make P*Qm have rank 2
    P = randn(3,2)*randn(2,4); Qm = randn(4,2)*randn(2,3)
    ixs = [[:a,:m],[:m,:b]]
    for cutter! in (cut_bondim!, cut_bondim_opt!)
        tn = TensorNetwork([P,Qm], ixs; chi=1000, Dmax=2)
        before = mps2raw(tn.tensors[1])      # sanity
        cutter!(tn, 1, 1)
        @test size(tn.tensors[1].mps[1], 2) == 2   # physical bond truncated to 2
        # network contraction value preserved
        got = ein"am,mb->ab"(mps2raw(tn.tensors[1]), mps2raw(tn.tensors[2]))
        @test got ≈ P*Qm atol=1e-8
    end
end
```

- [ ] **Step 2–6:** Run→fail; implement both cutters per the reference (1-based, OMEinsum for the merge contraction, `tsvd` for the decomposition, `qr` for the opt reductions); `Dmax<0` short-circuits to no-op returning `0.0`; export; run→pass; commit:

```bash
git add src/tensor_network.jl src/CATN.jl test/test_contraction.jl
git commit -m "feat: cut_bondim!/cut_bondim_opt! physical-bond truncation"
```

---

### Task 12: `contraction!` loop

**Files:**
- Modify: `src/tensor_network.jl`
- Modify: `test/test_contraction.jl`

**Reference:** `tn_np.py:294` (`contraction`), `:453` (`lognorm`).

**Interfaces:**
- Produces: `contraction!(tn) -> (lnZ, error, psi)` — the full edge-by-edge loop (steps as in the design spec §6 and `tn_np.py:294-451`): select edge; ensure `order(i) ≥ order(j)`; `count_remove_nodes!`; optional `reverse!` to minimize swaps; re-point `j`'s other neighbors to `i`, detecting & `merge!`-ing duplicates; `eat!`; per-duplicate `merge!` + `cut_bondim*!`; clear `j`; optional `compress*!`; `count_add_nodes!`; track `maxdim_intermediate`. After the loop add remaining `lognorm`s and `log(2)*num_isolated`. Calls `select_edge_init!` at the start if `edge_count` is empty.

- [ ] **Step 1: Write the failing test** — append to `test/test_contraction.jl`

```julia
using CATN: contraction!

function catn_value(tensors, ixs; kwargs...)
    tn = TensorNetwork(tensors, ixs; kwargs...)
    lnZ, err, psi = contraction!(tn)
    return exp(lnZ) * psi
end

@testset "contraction! exact mode matches oracle" begin
    networks = [
        # chain
        ([randn(3,4), randn(4,5), randn(5,3)], [[:a,:b],[:b,:c],[:c,:a]]),
        # star/tree
        ([randn(2,3,4), randn(2), randn(3), randn(4)],
         [[:a,:b,:c],[:a],[:b],[:c]]),
        # single loop of 4
        ([randn(2,3),randn(3,2),randn(2,3),randn(3,2)],
         [[:a,:b],[:b,:c],[:c,:d],[:d,:a]]),
        # loopy: two triangles sharing an edge
        ([randn(2,2,2),randn(2,2,2),randn(2,2,2),randn(2,2)],
         [[:a,:b,:e],[:b,:c,:f],[:c,:a,:g],[:e,:f]]),  # adjust to be fully contractible
    ]
    for (ts, ixs) in networks[1:3]
        ref = exact_contract(ts, ixs)[]
        for sel in 0:2, rev in (false,true), comp in (false,true)
            got = catn_value(ts, ixs; Dmax=-1, chi=10_000,
                             select=sel, reverse=rev, compress=comp, norm_method=1)
            @test got ≈ ref rtol=1e-8
        end
    end
end

@testset "contraction! finite Dmax is close on a contractible loop" begin
    ts, ixs = ([randn(4,4),randn(4,4),randn(4,4),randn(4,4)],
               [[:a,:b],[:b,:c],[:c,:d],[:d,:a]])
    ref = exact_contract(ts, ixs)[]
    got = catn_value(ts, ixs; Dmax=4, chi=64, select=1, compress=true)
    @test got ≈ ref rtol=1e-6
end
```
(Fix the 4th network so every label appears exactly twice before enabling it; the first three are sufficient for the exact-mode gate.)

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Implement** `contraction!` per the reference, 1-based, reusing all prior methods. Be meticulous with the neighbor re-pointing block (`tn_np.py:344-362`) and the duplicate handling (`tn_np.py:374-395`). Adjacency is maintained implicitly through each node's `neighbor` vector — there is no separate graph object; "edges remain" ⇔ any node has a non-sentinel neighbor. Track remaining edges by scanning `neighbor` lists.

- [ ] **Step 4: Export** `contraction!`.

- [ ] **Step 5: Run, expect pass.** Investigate any mismatch with systematic-debugging; common culprits: leg-order/`permutedims`, SVD `V` vs `Vt`, off-by-one in `reverse`/`len÷2` thresholds.

- [ ] **Step 6: Commit**

```bash
git add src/tensor_network.jl src/CATN.jl test/test_contraction.jl
git commit -m "feat: CATN contraction! loop (exact mode matches oracle)"
```

---

### Task 13: Ising network + free energy

**Files:**
- Create: `src/ising.jl`
- Modify: `src/CATN.jl`
- Create: `test/test_ising.jl`
- Modify: `test/runtests.jl`

**Reference:** `tn_np.py:71` (`construct_tensor`), `lnz_np.py` (driver, free energy).

**Interfaces:**
- Produces:
  - `ising_network(n::Int, edges::Vector{<:Tuple{Int,Int}}, weights::AbstractVector, fields::AbstractVector, β::Real; Dmax=32, chi=32, kwargs...) -> TensorNetwork` — builds the partition-function TN directly in MPS form (does NOT go through the generic dense constructor): per `construct_tensor`, each node accumulates rank-3 COPY/δ sites, edges contribute `B = exp.(β .* w .* M_ij)` split `Q=B, R=I` across endpoints, fields fold `exp.(β .* h .* spin)` into the leaf/first site. Handle the four degree cases (`tn_np.py:123-166`).
  - `free_energy(tn) -> (lnZ_per_site, F)` with `F = -lnZ_per_site/β` after `contraction!` (store `n`, `β` on the network or pass through).

**Design note:** keep `n` and `β` available to `free_energy` — add them as fields on `TensorNetwork` (default `n = length(tensors)`, `β = 1`) set by `ising_network`, or return them from `ising_network`. Choose adding optional `n::Int` and `beta::Float64` fields with sensible defaults to avoid breaking Task 9's constructor.

- [ ] **Step 1: Write the failing test** — `test/test_ising.jl`

```julia
using CATN: ising_network, free_energy, contraction!
using Test

# analytic lnZ of an open Ising chain of L spins, coupling J, no field:
# Z = 2^L * (cosh(βJ))^(L-1) * ... ; use transfer matrix for the ground truth
function chain_lnZ(L, J, β)
    T = [exp(β*J) exp(-β*J); exp(-β*J) exp(β*J)]
    v = ones(2)
    M = v' * T^(L-1) * v
    return log(M)
end

@testset "Ising chain lnZ" begin
    L, J, β = 6, 0.7, 0.4
    edges = [(i, i+1) for i in 1:L-1]
    tn = ising_network(L, edges, fill(J, L-1), zeros(L), β; Dmax=-1, chi=10_000)
    lnZ, err, psi = contraction!(tn)
    @test lnZ ≈ chain_lnZ(L, J, β) rtol=1e-9
end

@testset "Ising small loop matches brute force" begin
    # triangle with random couplings and fields, brute-force Z
    β = 0.3
    edges = [(1,2),(2,3),(3,1)]
    w = [0.5, -0.8, 1.1]; h = [0.2, -0.1, 0.4]
    function brute(edges, w, h, β, n)
        Z = 0.0
        for bits in 0:(2^n-1)
            s = [(bits >> (k-1)) & 1 == 1 ? 1 : -1 for k in 1:n]
            E = sum(w[e]*s[edges[e][1]]*s[edges[e][2]] for e in eachindex(edges)) +
                sum(h[k]*s[k] for k in 1:n)
            Z += exp(β*E)
        end
        return log(Z)
    end
    tn = ising_network(3, edges, w, h, β; Dmax=-1, chi=10_000)
    lnZ, _, _ = contraction!(tn)
    @test lnZ ≈ brute(edges, w, h, β, 3) rtol=1e-9
end
```

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Implement** `src/ising.jl` porting `construct_tensor` (build each node's `mps` directly; the interior δ-site is `t3` with `t3[:,1,:]=Diagonal(Q[:,1])`, `t3[:,2,:]=Diagonal(Q[:,2])`; leaf/first/last cases per `tn_np.py:123-166`). Build a `TensorNetwork` from the resulting nodes (add an internal constructor that takes pre-built `Dict{Int,MPSNode}` + `n`,`β`). Add `n`,`β` fields to `TensorNetwork` (Task 9) with defaults. Implement `free_energy`.

- [ ] **Step 4: Update `src/CATN.jl`** — include + export `ising_network, free_energy`.

- [ ] **Step 5: Wire `test/runtests.jl`** — `include("test_ising.jl")`.

- [ ] **Step 6: Run, expect pass.** Use systematic-debugging on mismatches; verify the δ-tensor construction by comparing a single node's `mps2raw` to the explicit COPY tensor for small degree.

- [ ] **Step 7: Commit**

```bash
git add src/ising.jl src/CATN.jl test/test_ising.jl test/runtests.jl
git commit -m "feat: Ising network construction + free energy"
```

---

### Task 14: Magnetization + correlation via pinning

**Files:**
- Modify: `src/ising.jl`
- Modify: `test/test_ising.jl`

**Reference:** `lnz_np.py:200-231` (`calc_mag`, `calc_cor`), `tn_np.py:71` (`construct_tensor` with `pos1,val1,pos2,val2`).

**Interfaces:**
- Produces:
  - `magnetization(n, edges, weights, fields, β; kwargs...) -> Vector{Float64}` — for each site, contract twice with that spin pinned to each value (`construct_tensor(pos,val)` masks the spin via `spin[val]=0`), combine `exp(lnZ_pinned - lnZ)` with sign `(+1 for s=+1, -1 for s=-1)` → `⟨s_i⟩`.
  - `correlation(n, edges, weights, fields, β; kwargs...) -> Vector{Float64}` — per edge, four pinned contractions combined per `calc_cor`.

(Take couplings/fields directly so each pinned network is rebuilt cleanly, mirroring `lnz_np.py` which rebuilds the TN per pin.)

- [ ] **Step 1: Write the failing test** — append to `test/test_ising.jl`

```julia
using CATN: magnetization, correlation

@testset "magnetization & correlation vs brute force" begin
    β = 0.35
    edges = [(1,2),(2,3),(3,1)]
    w = [0.4, 0.6, -0.3]; h = [0.1, -0.2, 0.05]; n = 3
    function brute_obs(edges, w, h, β, n)
        Z = 0.0; m = zeros(n); c = zeros(length(edges))
        for bits in 0:(2^n-1)
            s = [(bits >> (k-1)) & 1 == 1 ? 1 : -1 for k in 1:n]
            E = sum(w[e]*s[edges[e][1]]*s[edges[e][2]] for e in eachindex(edges)) +
                sum(h[k]*s[k] for k in 1:n)
            p = exp(β*E); Z += p
            for k in 1:n; m[k] += p*s[k]; end
            for e in eachindex(edges); c[e] += p*s[edges[e][1]]*s[edges[e][2]]; end
        end
        return m./Z, c./Z
    end
    m_ref, c_ref = brute_obs(edges, w, h, β, n)
    m = magnetization(n, edges, w, h, β; Dmax=-1, chi=10_000)
    c = correlation(n, edges, w, h, β; Dmax=-1, chi=10_000)
    @test m ≈ m_ref rtol=1e-7
    @test c ≈ c_ref rtol=1e-7
end
```

- [ ] **Step 2–6:** Run→fail; implement `magnetization`/`correlation` (rebuild pinned `ising_network`s — extend `ising_network`/`construct_tensor` to accept optional `pos1,val1,pos2,val2` pinning args, defaulting to none); export; run→pass; commit:

```bash
git add src/ising.jl src/CATN.jl test/test_ising.jl
git commit -m "feat: Ising magnetization & correlation via pinning"
```

---

### Task 15: Final integration, docs, full-suite green

**Files:**
- Modify: `README.md`
- Modify: `test/runtests.jl` (ensure all includes present)

- [ ] **Step 1: Run the whole suite**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: all testsets PASS.

- [ ] **Step 2: Add a usage section to `README.md`** — a minimal example:

````markdown
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
````

- [ ] **Step 3: Commit**

```bash
git add README.md test/runtests.jl
git commit -m "docs: usage example; finalize CATN.jl test suite"
```

---

## Self-Review

**Spec coverage:** §4 linalg→Task 1; §8 oracle→Task 2; §5 MPSNode (raw2mps/mps2raw→T3, cano→T4, swap/move/reverse→T5, merge→T6, compress→T7, eat→T8); §6 TensorNetwork (struct/construct→T9, select/bookkeeping→T10, cut_bondim→T11, contraction!→T12); §7 Ising (network/free energy→T13, magnetization/correlation→T14); §8 testing folded into each task + T12/T13/T14 integration; §9 risks addressed via column-major reshape conventions (T3/T4), SVD wrapper (T1), OMEinsum routing (throughout), 1-based notes (each task). Full parity items: 3 select modes→T10/T12; norm_method 0/1/2→T8; svdopt/compress_opt→T7/T11; rsvd in swap→T5; Dmax<0 exact→T11/T12; pinning mag/corr→T14. All covered.

**Placeholder scan:** test code is complete; implementation steps that defer to the reference give exact file:line pointers plus the 1-based index math and OMEinsum patterns for the error-prone parts. No "TBD"/"add error handling"/"similar to Task N".

**Type consistency:** `MPSNode{T}` fields and `TensorNetwork{T}` fields are fixed in T3/T9 and reused unchanged; `tsvd`/`rsvd` return `(U,S,V)` consistently; `eat!` returns `(lognorm,error,phase)` and `contraction!` returns `(lnZ,error,psi)` consistently; `cut_bondim!`/`cut_bondim_opt!` share signature `(tn, i, idx_j_in_i)`. `find_neighbor` returns `0` when absent (documented in T5). `TensorNetwork` gains `n`,`β` fields in T13 with defaults so T9's constructor stays valid.
