# CATN.jl Double-Layer (bra–ket) Implementation Plan — v1 (norm core)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A layer-aware double-layer tensor-network type that computes `⟨ψ|ψ⟩` for a tensor-network state, keeping ket/bra separate on every network edge and combining/truncating only inside each node's MPS (environment-aware) — validated against exact contraction.

**Architecture:** New `src/braket.jl` with `BraKetNode` (an MPS whose physical legs are tagged ket/bra and ≤`D`, whose internal bonds are combined bra⊗ket and ≤`χ`) and `BraKetNetwork`. Build each node's double tensor `E_i = Σ_p T_i conj(T_i)` *without* forming a dense object (join ket and bra MPS at the physical index). Contract edge-by-edge; `eat!` contracts the paired ket+bra edges and folds into combined MPS; compression truncates only the combined internal bonds.

**Tech Stack:** Julia 1.12 (juliaup `/mnt/home/xgao1/.juliaup/bin/julia`), OMEinsum, LinearAlgebra, Test. Reuses `tsvd` from `src/linalg_utils.jl` and the MPS algorithm patterns from `src/mps_node.jl`.

**Reference:** spec `docs/superpowers/specs/2026-06-30-catn-double-layer-design.md`.

## Global Constraints

- Julia via the juliaup binary; never `module load julia`. Full suite: `julia --project=. -e 'using Pkg; Pkg.test()'`; faster: `julia --project=. -e 'include("test/runtests.jl")'`.
- **Separate on edges, combined inside:** every inter-node bond is a pair (ket edge, bra edge), each ≤ `D`; every MPS-internal bond is combined bra⊗ket, ≤ `χ`. **Never** truncate a single-layer inter-node bond in isolation; the only truncation is canonical-form MPS compression of the combined internal bonds.
- **No dense double tensor:** build `E_i`'s MPS by joining the ket MPS (from `T_i`) and bra MPS (from `conj(T_i)`) at the contracted physical index — never materialize the `D^{2·deg}` object.
- **Bra = `conj`**, and contraction is non-conjugating (already correct in the codebase), so the joined object computes `⟨ψ|ψ⟩`.
- v1 handles **acyclic** virtual-bond graphs (chains/trees); the constructor rejects cyclic ones with a clear error.
- Column-major reshape conventions and the `(U,S,V)` SVD convention (`A ≈ U·Diagonal(S)·V'`) match the rest of the package.
- Result: `⟨ψ|ψ⟩ = exp(lnZ) * psi`; tests compare to a direct exact double-layer contraction via the existing `exact_contract`.
- Reuse `tsvd` (returns `(U,S,V,discarded)`); MPS internal ops should track the per-site `layer` tag in lockstep with `neighbor` under any permutation, exactly as `MPSNode.neighbor` does.
- Commit after each task with trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File Structure

```
src/
  braket.jl          # BraKetNode, braket_node, BraKetNetwork, braket_network, contraction!
  CATN.jl            # include + exports
test/
  test_braket.jl     # node-construction + norm-vs-oracle tests
  runtests.jl        # include test_braket.jl
```

---

### Task 1: `BraKetNode` struct + single-node construction (`braket_node`)

**Files:**
- Create: `src/braket.jl`
- Modify: `src/CATN.jl` (include + export `BraKetNode`, `braket_node`, plus helpers)
- Create: `test/test_braket.jl`
- Modify: `test/runtests.jl`

**Interfaces:**
- Produces:
  - `mutable struct BraKetNode{T,AT<:AbstractArray{T,3}}` with `mps::Vector{AT}`, `neighbor::Vector{Int}`, `layer::Vector{Bool}` (true=ket, false=bra), `cano::Int`, `chi::Int`, `cutoff::Float64`, `norm_method::Int`, `svdopt::Bool`, `swapopt::Bool`.
  - `braket_node(Ti::AbstractArray{T}, ket_neighbors::Vector{Int}, phys_pos::Int; chi, cutoff, …) -> BraKetNode` — `Ti` is the ket site tensor; `phys_pos` is which axis of `Ti` is the physical index; the remaining axes are virtual bonds, one per entry of `ket_neighbors`. Builds `E_i = Σ_p Ti conj(Ti)` as an MPS with site order `[ket virtual legs …, bra virtual legs …]`, physical legs tagged ket (`true`) for the first `deg` sites and bra (`false`) for the last `deg` sites, `neighbor` repeating the neighbor ids in each block. Built by joining ket & bra MPS at the physical index — **no dense double tensor**.
  - `mps2raw(node::BraKetNode)`, `order`, `shape`, `find_leg(node, j, isket)`.

- [ ] **Step 1: Write the failing test** — create `test/test_braket.jl`

```julia
using CATN
using CATN: BraKetNode, braket_node, mps2raw
using OMEinsum, LinearAlgebra, Test

@testset "braket" begin
    @testset "braket_node builds the double tensor E_i" begin
        # degree-2 site: axes (v1, p, v2); physical index is axis 2
        D, d = 3, 2
        Ti = randn(ComplexF64, D, d, D)
        node = braket_node(Ti, [10, 20], 2; chi=10_000)   # neighbors 10 (v1), 20 (v2)
        # exact E_i[v1,v1', v2,v2'] = Σ_p Ti[v1,p,v2] conj(Ti)[v1',p,v2']
        E = ein"apb,cpd->abcd"(Ti, conj(Ti))              # (v1,v2,v1',v2')
        # reconstruct from the node: legs in node order [ket v1, ket v2, bra v1', bra v2']
        raw = mps2raw(node)                                # dims (v1, v2, v1', v2') per site order
        @test size(raw) == (D, D, D, D)
        @test raw ≈ E
        @test node.layer == [true, true, false, false]    # ket block then bra block
        @test node.neighbor == [10, 20, 10, 20]
        # physical legs are ≤ D, internal junction bond ≤ d_phys
        @test all(size(s,2) ≤ D for s in node.mps)
        @test maximum(size(s,3) for s in node.mps[1:end-1]) ≤ max(D, d)
    end
end
```
(Adjust the `ein"apb,cpd->abcd"` index order and the asserted `mps2raw` leg order to whatever the implementation produces, but keep it a genuine `raw ≈ E` value check against the independently-formed `E`. Document the chosen leg order.)

- [ ] **Step 2: Run, expect failure**

Run: `julia --project=. -e 'include("test/runtests.jl")'` (after wiring includes)
Expected: FAIL — `braket_node`/`BraKetNode` not defined.

- [ ] **Step 3: Implement the struct + `braket_node` in `src/braket.jl`.** Approach (no dense `E_i`):
  - Permute `Ti` so the physical axis is last: `Tk = permutedims(Ti, (virtual…, phys))`, shape `(D₁,…,D_deg, d)`.
  - Build a **ket MPS** of `Tk` over its virtual axes with the physical axis as the trailing boundary leg: i.e. raw2mps-style left-to-right SVD on the virtual axes, leaving `p` attached to the last ket site as an extra leg (so the last ket site is `(χ, D_deg, d)` carrying `p`). Each ket site is tagged `layer=true`, `neighbor=ket_neighbors[k]`.
  - Build a **bra MPS** of `conj(Tk)` likewise, with `p` as the **leading** boundary leg of the first bra site, tagged `layer=false`.
  - **Join at `p`:** contract the ket chain's trailing `p`-leg with the bra chain's leading `p`-leg (`ein` over `p`), producing one MPS whose internal junction bond has dim ≤ `d`. Concatenate `mps`, `neighbor`, `layer`.
  - Set `cano` to a valid canonical position. Reuse `tsvd` and column-major reshape conventions from `mps_node.jl`. `mps2raw` is the generic MPS reconstruction (same as `MPSNode`'s, ignoring tags).
  - Degree-1 leaf: `Ti` is `(D, d)` → `E_i` is `(D_ket, D_bra)` (two physical legs, one ket one bra), a 2-site MPS.

- [ ] **Step 4: Wire `src/CATN.jl`** (`include("braket.jl")` after `mps_node.jl`; export `BraKetNode, braket_node`) and `test/runtests.jl` (`include("test_braket.jl")`).

- [ ] **Step 5: Run, expect pass.** `raw ≈ E` confirms `E_i` is correctly represented with separate ket/bra legs and small internal bonds. If the leg order differs, fix the test's `permutedims`/`ein` to match (keep it a real value check). Full suite stays green.

- [ ] **Step 6: Commit**

```bash
git add src/braket.jl src/CATN.jl test/test_braket.jl test/runtests.jl
git commit -m "feat: BraKetNode + braket_node (double tensor as separate-leg MPS)"
```

---

### Task 2: `BraKetNetwork` construction + exact double-layer oracle

**Files:**
- Modify: `src/braket.jl` (`BraKetNetwork`, `braket_network`)
- Modify: `src/CATN.jl` (exports)
- Modify: `test/test_braket.jl`

**Interfaces:**
- Consumes: `braket_node` (Task 1), `exact_contract` (test/exact.jl).
- Produces:
  - `mutable struct BraKetNetwork{T,AT}` holding `tensors::Dict{Int,BraKetNode{T,AT}}`, the paired-edge bookkeeping, contraction params, and accumulators (`lnZ::T`, `sign::T`, `psi::T`, `chi`, `cutoff`, `norm_method`, `svdopt`, `swapopt`, `maxdim_intermediate`).
  - `braket_network(tensors::Vector{<:AbstractArray}, ixs::Vector{<:AbstractVector}; chi=64, cutoff=1e-15, norm_method=1, svdopt=true, swapopt=true, seed=1) -> BraKetNetwork` — open labels = physical (exactly one per tensor in v1), shared labels = virtual bonds; builds a `BraKetNode` per site; each shared bond becomes a paired (ket edge, bra edge). **Rejects cyclic** virtual-bond graphs with a clear error.
  - Test helper `exact_norm(tensors, ixs)` — builds the explicit full double-layer network (`tensors` + `conj.(tensors)`, physical indices linking `T_i`↔`conj(T_i)`, ket/bra virtual bonds distinct) and returns `exact_contract(...)[]` = exact `⟨ψ|ψ⟩`.

- [ ] **Step 1: Write the failing test** — append to `test/test_braket.jl`

```julia
using CATN: BraKetNetwork, braket_network

# Exact ⟨ψ|ψ⟩ via a direct double-layer contraction (independent oracle).
function exact_norm(tensors, ixs)
    n = length(tensors)
    # relabel: ket keeps labels; bra gets distinct labels for virtual bonds, shares physical.
    counts = Dict{Any,Int}(); for ix in ixs, l in ix; counts[l]=get(counts,l,0)+1; end
    bra_ix = [map(l -> counts[l]==2 ? (l, :bra) : l, ix) for ix in ixs]  # virtual→distinct, physical shared
    all_t   = vcat(collect(tensors), [conj(t) for t in tensors])
    all_ix  = vcat([collect(ix) for ix in ixs], [collect(b) for b in bra_ix])
    return exact_contract(all_t, all_ix)[]
end

@testset "braket_network construction" begin
    # 3-site chain (acyclic): T1(p1,a) T2(a,p2,b) T3(b,p3)
    T1 = randn(ComplexF64, 2, 3)
    T2 = randn(ComplexF64, 3, 2, 3)
    T3 = randn(ComplexF64, 3, 2)
    tensors = [T1, T2, T3]
    ixs = [[:p1, :a], [:a, :p2, :b], [:b, :p3]]
    bk = braket_network(tensors, ixs; chi=10_000)
    @test bk isa BraKetNetwork
    @test length(bk.tensors) == 3
    # each node reconstructs its E_i (spot-check node 2 has 4 physical legs: 2 ket + 2 bra)
    @test count(bk.tensors[2].layer) == 2 && count(!, bk.tensors[2].layer) == 2
end
```

- [ ] **Step 2–6:** Run→fail; implement `BraKetNetwork`/`braket_network` (detect physical=open vs virtual=shared labels; build nodes via `braket_node`; record paired edges; cycle-detect the virtual-bond graph and `error` if cyclic); export; wire; run→pass; commit:

```bash
git add src/braket.jl src/CATN.jl test/test_braket.jl
git commit -m "feat: BraKetNetwork construction + exact double-layer test oracle"
```

---

### Task 3: paired-edge `eat!` (the two-node core)

**Files:**
- Modify: `src/braket.jl` (`eat!` for `BraKetNode`, plus the MPS helpers it needs: `cano_to!`, `swap!`/move, `compress!` adapted to carry the `layer` tag)
- Modify: `test/test_braket.jl`

**Interfaces:**
- Produces:
  - `cano_to!`, `swap!`, `move2tail!`, `move2head!`, `compress!` for `BraKetNode` — same algorithms as `MPSNode` but permuting `layer` in lockstep with `neighbor`. (Reuse logic; the tag is extra metadata.)
  - `eat!(node_i, node_j, bond)` — contract the **paired** ket leg AND bra leg of the shared `bond` between `i` and `j`: bring `i`'s ket-leg-to-`j` and bra-leg-to-`j` to the MPS tail, `j`'s corresponding legs to its head, contract both, append `j`'s remaining (separate) ket/bra legs to `i`, normalize, return `(lognorm, error, phase)`.

**Two-node correctness target:** for two sites sharing one virtual bond, contracting that bond in the double layer must equal the exact `⟨ψ|ψ⟩` of the 2-site state (a scalar, since both physical indices and the shared bond are contracted).

- [ ] **Step 1: Write the failing test** — append to `test/test_braket.jl`

```julia
using CATN: eat!, find_leg, mps2raw

@testset "paired-edge eat! on two sites" begin
    # 2-site state: T1(p1, a), T2(a, p2). Norm contracts p1,p2 and bond a (ket & bra).
    T1 = randn(ComplexF64, 2, 4)        # (p1, a)
    T2 = randn(ComplexF64, 4, 2)        # (a, p2)
    n1 = braket_node(T1, [2], 1; chi=10_000)   # neighbor 2 via bond a; phys axis 1
    n2 = braket_node(T2, [1], 2; chi=10_000)   # neighbor 1 via bond a; phys axis 2
    lognorm, err, phase = eat!(n1, n2, 1)      # contract the bond shared with neighbor that links them
    # after eating, n1 should be a scalar (all legs contracted) → ⟨ψ|ψ⟩
    val = exp(lognorm) * phase
    ref = exact_norm([T1, T2], [[:p1, :a], [:a, :p2]])
    @test val ≈ ref rtol=1e-10
    @test imag(ref) ≈ 0 atol=1e-10 && real(ref) ≥ 0      # it's a norm
end
```
(Adapt the `eat!` call signature to the implementation — it takes whatever identifies the shared bond/neighbor. The binding check is `val ≈ ref`.)

- [ ] **Step 2: Run, expect failure.**

- [ ] **Step 3: Implement** the tag-carrying MPS helpers and `eat!`. Key points: when bringing legs to the ends, move BOTH the ket leg and the bra leg of the shared bond; contract them with `j`'s matching legs; the new internal bonds combine bra⊗ket. Mirror `MPSNode.eat!`'s structure but contracting two legs (ket then bra, adjacent) rather than one. Normalize per `norm_method`.

- [ ] **Step 4: Run, expect pass.** If `val ≉ ref`: debug systematically — likely culprits are leg ordering between ket/bra blocks, the join-bond orientation, or moving only one of the paired legs. Do NOT loosen the tolerance.

- [ ] **Step 5: Commit**

```bash
git add src/braket.jl test/test_braket.jl
git commit -m "feat: paired-edge eat! for BraKetNode (two-site norm matches exact)"
```

---

### Task 4: full `contraction!` loop (exact mode)

**Files:**
- Modify: `src/braket.jl` (`contraction!`, edge selection)
- Modify: `test/test_braket.jl`

**Interfaces:**
- Produces: `contraction!(bk::BraKetNetwork) -> (lnZ, error, psi)` — loop over original virtual bonds, `eat!` each (ensuring `order(i) ≥ order(j)`), `compress!` after each, accumulate. v1 selection: pick the bond giving the smallest resulting internal dimension, else sequential (acyclic ⇒ order is not critical for correctness). After the loop, fold remaining per-node norms; `⟨ψ|ψ⟩ = exp(lnZ)·psi`.

- [ ] **Step 1: Write the failing test** — append to `test/test_braket.jl`

```julia
using CATN: contraction!

function braket_value(tensors, ixs; kwargs...)
    bk = braket_network(tensors, ixs; kwargs...)
    lnZ, err, psi = contraction!(bk)
    return exp(lnZ) * psi
end

@testset "norm matches exact (exact mode)" begin
    # random MPS-state chains of various lengths/bond dims, and a small tree
    cases = [
        ([randn(ComplexF64,2,3), randn(ComplexF64,3,2,3), randn(ComplexF64,3,2,3), randn(ComplexF64,3,2)],
         [[:p1,:a],[:a,:p2,:b],[:b,:p3,:c],[:c,:p4]]),                       # 4-chain
        ([randn(2,2), randn(2,2,2), randn(2,2)],                            # real 3-chain
         [[:p1,:a],[:a,:p2,:b],[:b,:p3]]),
        ([randn(ComplexF64,2,2,2), randn(ComplexF64,2,2), randn(ComplexF64,2,2), randn(ComplexF64,2,2)],
         [[:a,:b,:p1],[:a,:p2],[:b,:p3],[:p4, :?]]),                        # tree (fix labels so acyclic & valid)
    ]
    for (ts, ixs) in cases[1:2]
        ref = exact_norm(ts, ixs)
        val = braket_value(ts, ixs; chi=10_000)
        @test val ≈ ref rtol=1e-10
        @test real(ref) ≥ 0
    end
end
```
(Use the first two cases for the gate; construct a genuinely acyclic, fully-valid tree for the third before enabling it.)

- [ ] **Step 2–5:** Run→fail; implement `contraction!` + selection; run→pass (chains and a tree match the oracle to `rtol 1e-10` in exact mode, real and complex). Debug systematically on mismatch; do not loosen tolerances. Commit:

```bash
git add src/braket.jl test/test_braket.jl
git commit -m "feat: BraKet contraction! loop (norm matches exact oracle)"
```

---

### Task 5: finite-`χ` truncation + invariants + docs

**Files:**
- Modify: `test/test_braket.jl`
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above. No new exports.

- [ ] **Step 1: Write the tests** — append to `test/test_braket.jl`

```julia
@testset "finite-chi truncation" begin
    # Low-Schmidt-rank state: truncation to sufficient chi is lossless.
    # Build a chain whose internal (combined) bonds are genuinely low rank, set chi to cover it.
    ts = [randn(ComplexF64,2,2), randn(ComplexF64,2,2,2), randn(ComplexF64,2,2,2), randn(ComplexF64,2,2)]
    ixs = [[:p1,:a],[:a,:p2,:b],[:b,:p3,:c],[:c,:p4]]
    ref = exact_norm(ts, ixs)
    @test braket_value(ts, ixs; chi=64) ≈ ref rtol=1e-8     # chi large enough ⇒ lossless
end

@testset "structure invariants" begin
    ts = [randn(ComplexF64,2,3), randn(ComplexF64,3,2,3), randn(ComplexF64,3,2)]
    ixs = [[:p1,:a],[:a,:p2,:b],[:b,:p3]]
    bk = braket_network(ts, ixs; chi=8)
    # every physical leg ≤ D (max state bond), every internal bond ≤ chi — check fresh nodes
    D = 3
    for node in values(bk.tensors)
        @test all(size(s,2) ≤ D for s in node.mps)               # physical legs separate, ≤ D
    end
end

@testset "cyclic graph rejected (v1)" begin
    # A loop: T1-T2-T3-T1 in virtual bonds → should error in v1.
    ts = [randn(2,2,2), randn(2,2,2), randn(2,2,2)]
    ixs = [[:a,:c,:p1],[:a,:b,:p2],[:b,:c,:p3]]   # bonds a,b,c form a cycle
    @test_throws Exception braket_network(ts, ixs; chi=8)
end
```

- [ ] **Step 2: Run, expect pass** (implement any small guards needed — the cycle check should already exist from Task 2). 

- [ ] **Step 3: Add a README section** documenting `braket_network`:

````markdown
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
````

- [ ] **Step 4: Run the full suite; commit**

```bash
git add test/test_braket.jl README.md
git commit -m "test: finite-chi + invariants for double layer; docs"
```

---

## Self-Review

**Spec coverage:** §3.1 BraKetNode → Task 1; §3.2 node construction (no dense tensor) → Task 1; §3.3 BraKetNetwork/braket_network + oracle → Task 2; §3.4 contraction! → Tasks 3 (eat!) + 4 (loop); §2 truncation (combined internal only) → compress in Tasks 3–4 + invariants in Task 5; §4 testing (node unit, exact-mode norm real/complex, finite-χ, invariants, cycle rejection) → Tasks 1–5; §5 risks (construction validated first in T1, paired-edge eat validated on 2 sites in T3, no single-layer cut enforced, acyclic guard in T2/T5). All covered.

**Placeholder scan:** test bodies are concrete; two test cases are marked "fix labels to be acyclic/valid before enabling" (the tree case in T4) — the gating cases (chains) are complete. Implementation steps that defer mechanics give a concrete approach (join-at-physical-index, paired-leg eat) and a "validate first / debug systematically" instruction rather than vague placeholders.

**Type consistency:** `BraKetNode{T,AT}` fields fixed in T1 and reused; `braket_node(Ti, ket_neighbors, phys_pos; …)`, `braket_network(tensors, ixs; …)`, `eat!(node_i, node_j, bond)`, `contraction!(bk) -> (lnZ, error, psi)` consistent across tasks. `exact_norm` defined in T2, reused in T3–T5. Value comparisons use `exp(lnZ)*psi ≈ exact_norm(...)`, consistent with the package pattern.
