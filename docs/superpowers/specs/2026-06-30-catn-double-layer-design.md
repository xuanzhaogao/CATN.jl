# CATN.jl — Double-layer (bra–ket) tensor network

**Date:** 2026-06-30
**Status:** Approved design (v1 = norm core)
**Builds on:** the generic engine, complex support, and GPU support already on `main`.

## 1. Goal & scope

Add a **distinct, layer-aware** tensor-network type for double-layer (bra–ket) systems —
computing `⟨ψ|ψ⟩` for a tensor-network state `|ψ⟩` — with its own contraction *rules* that
differ from the generic `TensorNetwork`:

- the ket and bra layers are stored **separately on every network edge** (two `D`-bonds, never
  a fused `D²` bond) — the RAM win;
- ket/bra are **combined only inside a node's MPS**, and that combined internal bond is the
  **only** thing that gets truncated — an environment-aware truncation (canonical-form SVD),
  which is the physically meaningful one. Single-layer bonds are never truncated in isolation.

**In scope (v1):**
- A `BraKetNode`/`BraKetNetwork` type and a `braket_network(tensors, ixs)` constructor that takes
  the ket state and builds the double layer for the **norm `⟨ψ|ψ⟩`**.
- `contraction!` for the double-layer type, returning `⟨ψ|ψ⟩` (real, ≥0 up to truncation).
- Validation against a direct exact double-layer contraction (OMEinsum) on small states.
- Works for real and complex tensors (complex already supported), CPU.
- Focus on states whose **virtual-bond graph is acyclic (chains/trees)** for v1, so the
  eat order and paired-edge bookkeeping are unambiguous. Loopy states are a documented follow-on.

**Out of scope (follow-on):**
- Operator insertion `⟨ψ|O|ψ⟩`, overlaps `⟨φ|ψ⟩`, open physical legs.
- Loopy virtual-bond graphs (PEPS) — needs duplicate doubled-bond merging.
- GPU for the double-layer type (the building blocks are device-agnostic, but not validated here).

## 2. The algorithm (confirmed)

A TN state `|ψ⟩` has site tensors `T_i` with one physical index `p_i` (open) and virtual bonds
(dim `D`) to neighbors. The norm is
`⟨ψ|ψ⟩ = Σ (∏_i T_i)(∏_i conj(T_i))`, contracting each physical index `p_i` between `T_i` and
`conj(T_i)`, and each virtual bond within the ket layer and (separately) within the bra layer.

**Representation.** Each physical site becomes one `BraKetNode` holding a single MPS that
represents the *double tensor* `E_i = Σ_{p_i} T_i conj(T_i)`, but with:
- **physical legs kept separate** — one **ket leg** and one **bra leg** per neighbor (each ≤ `D`),
  each tagged by layer; and
- **combined internal (virtual) bonds** — small in the bulk (the ket-block and bra-block of the
  node are joined only through the physical-index contraction, an internal bond of dim ≤ `d_phys`),
  and growing — then truncated to `χ` — as the node absorbs neighbors.

So in the network: **every edge is a pair (ket edge, bra edge), each ≤ `D`, kept separate**;
**every MPS-internal bond is combined bra⊗ket, ≤ `χ`**. The dim of a node's internal bonds is
exactly the "separate vs combined" indicator: tiny for a bulk node, up to `χ` for a grown one.

**Contraction.** CATN-style: repeatedly pick an original virtual bond `(i,j)` and `eat!` it.
`eat!` contracts **both** the paired ket edge and bra edge between `i` and `j` and folds `j`'s MPS
into `i`'s; the newly internal bonds **combine** bra⊗ket. `i`'s remaining edges to other neighbors
stay separate ket/bra. After folding, the MPS is re-canonicalized and **compressed**, truncating
the combined internal bonds to `χ`.

**Truncation.** Only MPS-internal (combined) bonds are truncated, via canonical-form SVD
(`compress!`/`swap!`/`cano_to!`) → environment-aware. There is **no** single-layer
`cut_bondim` on inter-node bonds (that would be the meaningless single-layer truncation).
Inter-node ket/bra bonds stay at the state's `D`.

**Result.** `contraction!` accumulates per-step log-norms; `⟨ψ|ψ⟩ = exp(lnZ) * psi` (for a true
norm: real and ≥ 0 up to truncation, so `psi ≈ 1`).

## 3. Components (`src/braket.jl`)

### 3.1 `BraKetNode{T}`
```julia
mutable struct BraKetNode{T,AT<:AbstractArray{T,3}}
    mps::Vector{AT}          # combined-internal-bond MPS
    neighbor::Vector{Int}    # neighbor node id per site
    layer::Vector{Bool}      # true = ket leg, false = bra leg, per site
    cano::Int
    chi::Int
    cutoff::Float64
    norm_method::Int
    svdopt::Bool
    swapopt::Bool
end
```
Each neighbor `j` appears as up to two sites — one ket (`layer=true`) and one bra
(`layer=false`). `find_leg(node, j, isket) -> Int` returns the site index for that (neighbor,
layer); `order`, `shape`, `logdim` etc. mirror `MPSNode`. The MPS mechanics (`cano_to!`,
`swap!`, `compress!`) are reused at the algorithm level (they operate on a tagged MPS; the
tags ride along under permutations exactly like `neighbor` does in `MPSNode`).

### 3.2 Node construction (no dense double tensor)
`braket_node(T_i, ket_neighbors, phys_dim)` builds `E_i`'s MPS **without** forming the
`D^{2·deg}` dense double tensor:
1. Build the **ket MPS** from `T_i` over its virtual legs, carrying the physical leg `p` as the
   chain's trailing (boundary) leg.
2. Build the **bra MPS** from `conj(T_i)` likewise, with `p` as the leading leg.
3. **Join** at `p`: contract the ket chain's `p`-leg with the bra chain's `p`-leg → one MPS whose
   site order is `[ket legs …][bra legs …]`, with the junction internal bond = `d_phys`.
Physical legs are tagged ket/bra; internal bonds are combined and small. (For a degree-1 leaf,
`E_i` is a `(D_ket, D_bra)` object — two physical legs.)

### 3.3 `BraKetNetwork{T,AT}` + `braket_network(tensors, ixs; chi, cutoff, …)`
- `(tensors, ixs)` define the ket: open labels = physical indices (exactly one per site for v1),
  shared labels = virtual bonds.
- Builds a `BraKetNode` per site; every shared virtual bond becomes a **paired** (ket edge, bra
  edge) between the two nodes. Bookkeeping tracks, for each original bond, its ket and bra legs on
  both endpoints.
- Holds the contraction params and accumulators (`lnZ`, `psi`, …) like `TensorNetwork`.

### 3.4 `contraction!(bk) -> (lnZ, error, psi)`
Loop over remaining original virtual bonds (selection: simplest reasonable order for v1 — min
resulting internal dimension, falling back to sequential; trees make this non-critical):
1. Pick bond `(i,j)`; ensure `order(i) ≥ order(j)`.
2. `eat!`: bring the paired ket & bra legs to the MPS ends, contract both, append `j`'s remaining
   (still-separate) ket/bra legs, normalize, accumulate `lnZ`/`psi`.
3. `compress!` the merged MPS → truncate combined internal bonds to `χ`.
After the loop, combine remaining per-node norms. Return `(lnZ, error, psi)`; `⟨ψ|ψ⟩ = exp(lnZ)·psi`.

## 4. Testing & validation (`test/test_braket.jl`)

Independent oracle: build the **full double-layer network explicitly** — `tensors` and
`conj.(tensors)` wired so each physical index links `T_i`↔`conj(T_i)` and ket/bra virtual bonds are
distinct — and contract it with the existing `exact_contract` (OMEinsum). That value is the exact
`⟨ψ|ψ⟩`.

- **Unit:** `braket_node` of a small `T_i` reconstructs `E_i` (its `mps2raw`, regrouped by
  ket/bra legs, equals `Σ_p T_i conj(T_i)`); physical legs are correctly tagged and ≤ `D`; internal
  junction bond ≤ `d_phys`.
- **Norm, exact mode (large `χ`):** random MPS-state chains (e.g. `L=4..8`, `D=2..4`, `d_phys=2`)
  and small trees — `exp(lnZ)·psi ≈ ⟨ψ|ψ⟩` from the oracle, `rtol ~1e-10`. Real and complex.
- **Sanity:** `⟨ψ|ψ⟩` real and ≥ 0; equals `Σ |amplitudes|²` for a tiny enumerable state.
- **Finite `χ`:** for a state with low Schmidt rank, truncation to a sufficient `χ` is lossless
  (matches exact); for higher rank, close within a loose tolerance.
- **RAM/structure invariant:** inter-node bonds stay ≤ `D` throughout; only internal bonds reach
  `χ` (assert max internal bond ≤ `χ`, max physical leg ≤ `D`).

## 5. Risks & decisions

1. **Node construction without the dense double tensor** is the core enabler — build ket/bra MPS
   from `T_i`/`conj(T_i)` and join at `p`. Validate on a single node first.
2. **Paired-edge `eat!`** — both ket and bra legs of the chosen bond must be brought to the MPS
   ends and contracted together so the subsequent compression sees the combined object; the layer
   tags must track through swaps/moves exactly like neighbor ids.
3. **Environment-aware truncation** is realized as canonical-form MPS compression of the combined
   internal bonds — this is the established correctness mechanism (already used and tested in the
   generic engine); v1 reuses it rather than inventing new truncation math.
4. **No single-layer cut.** The double-layer type must never truncate an inter-node ket/bra bond
   on its own — only the combined internal bonds.
5. **Acyclic v1.** Loopy states create duplicate doubled bonds (the same pair of nodes connected by
   two virtual bonds → four edges) that need merging; deferred. Constructor should detect and
   reject (or warn on) cyclic virtual-bond graphs in v1.
6. **Complex.** `conj` for the bra layer + non-conjugating contraction (already correct) gives the
   right `⟨ψ|ψ⟩`.

## 6. Deliverables (v1)

- `src/braket.jl`: `BraKetNode`, `braket_node`, `BraKetNetwork`, `braket_network`, double-layer
  `contraction!`; exported from `CATN`.
- `test/test_braket.jl`: node-construction unit tests + norm-vs-exact-oracle integration tests
  (real & complex, exact and finite-`χ`), wired into `runtests.jl`.
- README section documenting `braket_network` and the separate-bulk / combined-truncation rules.
- Full suite stays green.
