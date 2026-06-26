# CATN.jl — Julia implementation of Contracting Arbitrary Tensor Networks

**Date:** 2026-06-26
**Status:** Approved design
**Reference:** Python implementation in `../catn` (numpy path: `tn_np.py`, `mps_node_np.py`, `lnz_np.py`, `npsvd.py`, `raw_node.py`, `args.py`), paper [arXiv:1912.03014](https://arxiv.org/abs/1912.03014).

## 1. Goal & scope

Implement the CATN algorithm in Julia, using **OMEinsum** as the backend for tensor
contractions. CATN approximately contracts a tensor network of arbitrary topology by
representing every network tensor as its own MPS and contracting the network edge by edge
with controlled truncation.

**In scope (this version):**

- The core MPS-node contraction engine (`MPSNode` + `TensorNetwork`), with **full parity**
  of the reference's algorithmic features:
  - Truncation knobs `Dmax` (physical bond dim `D` between tensors) and `chi` (virtual bond
    dim χ inside each MPS), with `cutoff` for discarding singular values.
  - `eat`, `swap`, `merge`, `cano_to`, `compress`/`compress_opt`,
    `cut_bondim`/`cut_bondim_opt`, `reverse`, the move helpers.
  - All three edge-selection heuristics (`select` = 0 min-dim, 1 min-dim+triangle,
    2 sequential) with the `edge_count` bookkeeping.
  - All three normalization methods (`norm_method` = 0 none, 1 L2, 2 max-abs).
  - QR-before-SVD optimization (`svdopt`) and randomized SVD (`rsvd`) for huge matrices in
    `swap` (`swapopt`).
  - Exact mode (`Dmax < 0`).
- The graphical-model (Ising / spin-glass) application layer: build the tensor network for a
  partition function from `(edges, J, h, β)`, compute `lnZ` and free energy, and compute
  magnetization and pairwise correlations via pinning (`calc_mag` / `calc_cor`).

**Out of scope (this version):** quantum-circuit (`qc.py`) and PEPS (`peps.py`)
applications; mean-field/BP baselines (`bp_mf.py`); exact FVS / Kac-Ward solvers
(`exact.py`); graph generators (RRG, small-world, etc. — the user supplies the edge list).
Tiny grid/complete-graph helpers may exist in tests only.

## 2. Approach

**Faithful structural port** (chosen over a whole-network EinCode rewrite, which would be a
different algorithm, and over a functional/immutable rewrite, which fights the inherently
stateful algorithm). Mirror the reference's two-struct design with methods mapping ~1:1 to
the Python, so the Julia can be reviewed against `../catn`.

**Backend split:**

- **Contractions** (the rank-3 einsum patterns in `swap`/`merge`/`eat`/`cano_to`/`compress`/
  `mps2raw`, the Ising δ-tensor builds, and the exact validator) go through **OMEinsum**.
- **Decompositions** (SVD, QR) use `LinearAlgebra`.
- Plain 2-D matrix products may use `*`.

Element type is generic (`Float64` default; `Complex` supported — relevant to the exact
path and future quantum-circuit use).

## 3. Module layout

```
src/
  CATN.jl            # module: exports, includes, `using OMEinsum, LinearAlgebra`
  linalg_utils.jl    # tsvd, rsvd, qr helpers, leg merge/split helpers
  mps_node.jl        # MPSNode struct + methods
  tensor_network.jl  # TensorNetwork struct, construction, contraction! loop,
                     #   selection heuristics, edge_count bookkeeping, cut_bondim!
  ising.jl           # graphical-model layer: ising_network, free_energy,
                     #   magnetization, correlation
test/
  runtests.jl        # includes the test files below
  exact.jl           # exact_contract via OMEinsum (reference)
  test_mps_node.jl   # unit tests for MPSNode operations
  test_contraction.jl# integration tests vs exact_contract
  test_ising.jl      # Ising lnZ / free energy / magnetization / correlation
```

## 4. Component: `linalg_utils.jl`

- `tsvd(A; cutoff, maxdim) -> (U, S, V)` — thin SVD with the convention
  `A ≈ U * Diagonal(S) * V'` (matching `npsvd.svd`, which returns `V = Vh'`). Falls back to
  the `gesvd`-style divide-and-conquer alternative on failure (LinearAlgebra's
  `svd(A; alg=LinearAlgebra.QRIteration())`). Truncates by `cutoff` and optional `maxdim`.
- `rsvd(A, k, oversample, power) -> (U, S, V)` — randomized SVD, port of `npsvd.rsvd`; uses a
  seedable RNG for deterministic tests.
- Leg helpers to pin down index semantics regardless of column-major layout:
  - `merge_legs(A, groups)` / a `combine`/`split` pair implemented with `permutedims` +
    `reshape`, taking explicit index orderings so a "merge axes (a,b)" is unambiguous.
  - These replace literal transcription of numpy `reshape`/`transpose` chains.

## 5. Component: `mps_node.jl`

```julia
mutable struct MPSNode{T}
    mps::Vector{Array{T,3}}   # chain of (left, physical, right); one site per neighbor
    neighbor::Vector{Int}     # neighbor[k] = node-id reached by physical leg of site k
    cano::Int                 # canonical center (1-based); 0 ⇒ empty
    chi::Int
    cutoff::Float64
    norm_method::Int
    svdopt::Bool
    swapopt::Bool
end
```

Methods (all mutating use `!`), 1:1 with the reference:

| Method | Role | Backend |
|---|---|---|
| `raw2mps(tensor, chi, cutoff)` | build MPS from dense tensor via sequential SVD; left-canonical | SVD |
| `mps2raw(node)` | reconstruct dense tensor | OMEinsum |
| `cano_to!(node, idx)` | move canonical center via SVD sweeps | SVD + OMEinsum |
| `left_canonical!(node)` | center to last site | SVD |
| `swap!(node, i, j)` | swap adjacent sites; truncate to `chi`; `rsvd` when huge & `swapopt` | OMEinsum + SVD/rSVD |
| `move!`, `move2head!`, `move2tail!` | reposition a site by chained swaps | — |
| `merge!(node, j; cross)` | fuse two sites pointing to the same neighbor `j` | OMEinsum |
| `eat!(node, nodej, idx, idxi)` | contract shared leg, append `nodej`'s remaining sites, normalize → `(lognorm, error, phase)` | OMEinsum + SVD |
| `compress!` / `compress_opt!` | two-site SVD sweep truncating to `chi` (opt = QR first) | OMEinsum + SVD/QR |
| `reverse!(node)` | reverse site order (`permutedims(t,(3,2,1))`, reverse neighbor) | — |
| `find_neighbor`, `add_neighbor!`, `delete_neighbor!` | neighbor bookkeeping | — |
| `logdim`, `order`, `shape`, `lognorm`, `clear!` | queries | — |

Fidelity points:

- `eat!` handles the three reference cases: both leaves → scalar dot (returns `log|result|`,
  phase `result/|result|`); `j` is a leaf; general case (append `j`'s remaining sites).
- Normalization in `eat!` follows `norm_method`; the accumulated `log(norm)` is returned to
  the caller for the running `lnZ`.
- `swap!` uses `rsvd` when `swapopt` and `(rows>7000 && cols>7000) || (rows>20000 || cols>20000)`.
- `merge!`'s `cross` flag chooses `move(idx2, idx1)` vs `move(idx2, idx1+1)`.

## 6. Component: `tensor_network.jl`

```julia
mutable struct TensorNetwork{T}
    tensors::Dict{Int,MPSNode{T}}
    Dmax::Int                 # <0 ⇒ exact
    chi::Int
    cutoff::Float64
    norm_method::Int
    select::Int               # 0 min-dim, 1 min-dim+triangle, 2 sequential
    reverse::Bool
    svdopt::Bool; swapopt::Bool; compress::Bool; cut_bond::Bool
    edge_count::Dict{Int,Vector{Vector{Int}}}  # cost ⇒ candidate [i,j] pairs
    lnZ::T; sign::T; psi::T
    maxdim_intermediate::Int
    num_isolated::Int
    rng::AbstractRNG          # for rsvd determinism
end
```

**Construction from OMEinsum-style labels.** Public entry:

```julia
TensorNetwork(tensors::Vector{<:AbstractArray}, ixs::Vector{<:AbstractVector};
              Dmax, chi, cutoff, norm_method, select, reverse,
              svdopt, swapopt, compress, cut_bond, seed)
```

`ixs[t]` lists the leg labels of `tensors[t]` (labels are arbitrary `hashable`; e.g. `Int`,
`Char`, `Symbol`). A label shared by exactly two tensors is a bond → a `neighbor` link
between those two nodes; a label appearing once is an open/output leg → represented as an
order-1 dangling site (mirrors the leaf handling), so open networks contract correctly.
Each input tensor becomes an `MPSNode` via `raw2mps`. The adjacency (multigraph) is derived
from shared labels. Self-loops / repeated labels within one tensor are validated/rejected
(unsupported, consistent with the reference).

**Contraction loop `contraction!(tn) -> (lnZ, error, psi)`** — faithful port of
`tn_np.contraction`:

1. Select an edge via `select_edge_min_dim` / `select_edge_min_dim_triangle` /
   `select_edge_sequentially`; ensure `order(i) ≥ order(j)` (swap roles otherwise).
2. `count_remove_nodes!` for `[i, j] ∪ neighbors(i) ∪ neighbors(j)` *before* shapes change.
3. If `reverse`, `reverse!` `i` and/or `j` so the shared leg sits in the near half
   (`idx_j_in_i ≥ len/2`, `idx_i_in_j < len/2`), minimizing swap count.
4. Delete the `i–j` link; re-point each other neighbor `k` of `j` to `i`
   (`add_neighbor!`/`delete_neighbor!`, update adjacency). If `k` was already a neighbor of
   `i` (a **duplicate**), record it and `merge!` it on the `k` side
   (`cross = idx_i_in_k > idx_j_in_k`).
5. `eat!` `j` into `i`; `lnZ += lognorm`, `psi *= phase`, `error += err`.
6. For each duplicate `k`: `merge!` on the `i` side, then `cut_bondim_opt!`/`cut_bondim!`
   if that bond exceeds `Dmax` (or always, when `cut_bond`).
7. Clear node `j`, remove it from the graph.
8. If `compress`, `compress_opt!`/`compress!` node `i`.
9. `count_add_nodes!` for `[i] ∪ neighbors(i)`; track `maxdim_intermediate`.
10. After the loop: add remaining per-node `lognorm`s and `log(2)·num_isolated`; return
    `(lnZ, error, psi)`.

**Selection / bookkeeping:** `dim_after_merge(i,j)`, `select_edge_init!`, `count_add_edges!`,
`count_add_nodes!`, `count_remove_nodes!`, and the three `select_edge_*` exactly as the
reference (including the triangle-count tie-break in `select_edge_min_dim_triangle`).

**`cut_bondim!` / `cut_bondim_opt!`:** SVD-truncate the shared physical bond between two
nodes to `Dmax`; the `opt` variant canonicalizes both nodes to that leg and does QR before
SVD. `Dmax < 0` (and large `chi`) ⇒ no truncation ⇒ exact contraction.

## 7. Component: `ising.jl`

```julia
ising_network(edges, J, h, β; Dmax, chi, kwargs...) -> TensorNetwork
free_energy(tn)   -> (lnZ_per_site, F)        # F = -lnZ/(n·β)
magnetization(tn) -> Vector                    # ⟨s_i⟩ via pinning (calc_mag)
correlation(tn)   -> Vector                    # ⟨s_i s_j⟩ per edge via pinning (calc_cor)
```

- Build per the reference `construct_tensor`: each spin → COPY/δ tensor built directly in MPS
  form (interior site = `t3[:,1,:]=Diagonal(Q[:,1])`, `t3[:,2,:]=Diagonal(Q[:,2])`); each
  edge → bond factor `B = exp(β · J · M_ij)` split `Q⊗R` (reference uses `Q=B`, `R=I`)
  across endpoints; field `exp(β h s)` folded into the leaf/first site. Replicate all four
  degree cases (leaf, first-neighbor, last-neighbor, interior) and the `M_ij` spin masking
  used for pinning (`construct_tensor(pos1,val1,pos2,val2)`).
- `magnetization` / `correlation`: rebuild with one/two spins pinned, contract, and combine
  `exp(lnZ_pinned − lnZ)` with the sign pattern from `calc_mag` / `calc_cor`.

## 8. Testing — exact OMEinsum reference

`test/exact.jl`: `exact_contract(tensors, ixs)` contracts the whole network exactly via
OMEinsum (`optimize_code` for a contraction order, then `einsum`) — the reference oracle.

- **Unit (`test_mps_node.jl`):** `raw2mps`∘`mps2raw` round-trip; `swap!`, `reverse!`,
  `merge!`, `cano_to!`, `compress!` preserve the represented dense tensor (within `cutoff`);
  `eat!` of two nodes equals a direct OMEinsum contraction of their dense forms.
- **Integration (`test_contraction.jl`):** small networks (chain, tree, single loop,
  small loopy graph) — CATN with `Dmax<0` (exact) matches `exact_contract` to ~1e-10;
  CATN with finite `Dmax/chi` matches within a loose tolerance; all three `select` modes,
  `reverse` on/off, and `compress` on/off agree in exact mode.
- **Ising (`test_ising.jl`):** 1×n chain `lnZ` matches the analytic transfer-matrix result;
  small 2-D grid and complete graph match `exact_contract` of the same Ising network in
  exact mode; finite-`Dmax` results are close; `magnetization`/`correlation` match exact
  (brute-force enumeration) for a tiny system.
- **Determinism:** seed the `rng`; the `rsvd` path is reproducible.

## 9. Key implementation risks & decisions

1. **Memory layout (numpy row-major vs Julia column-major).** Do *not* transcribe
   `reshape`/`transpose` literally. Route pure contractions through OMEinsum (layout- and
   index-correct) and perform every leg merge/split through the `linalg_utils.jl` helpers
   with explicit index orderings.
2. **Indexing.** 1-based throughout; carefully translate the reference's 0-based index
   arithmetic (e.g. `len//2` thresholds, `idx1+1`, `permutedims(t,(3,2,1))` for reverse).
3. **SVD convention.** Wrap SVD to return `(U, S, V)` with `A ≈ U·Diagonal(S)·V'`, matching
   `npsvd.svd`. Mind that LinearAlgebra's `svd` returns `Vt`.
4. **OMEinsum as the contraction backend.** Use the `ein"..."` string macro for the fixed
   rank-3 patterns; use dynamic `EinCode`/`einsum` where ranks vary (δ-tensor builds, exact
   validator). SVD/QR remain in LinearAlgebra (decompositions, not contractions).
5. **Generic element type** so the exact/complex path and future complex tensors work.

## 10. Deliverables

- `src/CATN.jl` + the four component files, exporting the public API
  (`MPSNode`, `TensorNetwork`, `contraction!`, `ising_network`, `free_energy`,
  `magnetization`, `correlation`).
- `test/` suite passing via `Pkg.test()`.
- `Project.toml` updated with needed stdlib deps (`LinearAlgebra`, `Random`) and `Test` extra
  (OMEinsum already present).
