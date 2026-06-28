# CATN.jl — Complex-number support

**Date:** 2026-06-28
**Status:** Approved design
**Builds on:** the CPU implementation (`2026-06-26-catn-julia-design.md`) and GPU support (`2026-06-27-catn-gpu-support-design.md`).

## 1. Goal & scope

Make CATN correct for complex-valued tensors by fixing the three spots where a
tensor-network *bond* contraction wrongly uses a conjugating adjoint. Bonds in a tensor
network are contracted **without** conjugation (pure tensor contraction, not bra–ket inner
products); the current code conjugates in three places, which is a no-op for real tensors but
wrong for complex.

**In scope:**
- Fix `eat!` case (a), `cut_bondim!`, and `cut_bondim_opt!` to contract bonds non-conjugated.
- Validate complex correctness on CPU (generic networks vs the OMEinsum oracle, exact mode AND
  the finite-Dmax cut path) and a GPU smoke test (the fixes are device-agnostic).
- Remove the "complex unsupported" limitation from README/spec docs.

**Out of scope:** any algorithmic change beyond the conjugation fixes; complex Ising couplings
(the Ising layer stays real — physical β·J is real); performance work.

## 2. Root cause & fixes

A tensor-network bond shared by two sites is summed without conjugation. Three sites violate
this:

### 2.1 `eat!` case (a) — `src/mps_node.jl` (~line 475)
Both nodes are single-site leaves; their shared bond is contracted to a scalar.
- **Now:** `r = dot(vi, vj)` — `dot` conjugates `vi`.
- **Fix:** `r = sum(vi .* vj)` (non-conjugating; a GPU-safe reduction). Equivalent to the
  reference's plain `mps[0] @ nodej.mps[0]`.

### 2.2 `cut_bondim!` — `src/tensor_network.jl` (~lines 363, 379)
SVD-truncates the shared physical bond. The merge and the V-split must respect the
non-conjugated downstream contraction.
- **Now:** `merged = mati * matj'`; later `matj = V * Diagonal(sqS)`.
- **Fix:** `merged = mati * transpose(matj)`; `matj = conj(V) * Diagonal(sqS)`.
- **Derivation:** the joint matrix is `M[x,y] = Σ_d mati[x,d]·matj[y,d] = mati·transpose(matj)`
  (non-conj). SVD `M = U·S·V'`. The truncated bond is contracted non-conjugated downstream, so
  we need `Σ_k mati'[x,k]·matj'[y,k] = M[x,y] = Σ_k U[x,k]·S[k]·conj(V[y,k])`, giving
  `mati' = U·√S` and `matj' = conj(V)·√S`. (`mati = U*Diagonal(sqS)` is unchanged.)

### 2.3 `cut_bondim_opt!` — `src/tensor_network.jl` (~lines 448, 464)
Same operation with a QR pre-reduction.
- **Fix:** `merged = ri * transpose(rj)` (was `ri * rj'`); `matj = conj(V) * Diagonal(sqS)`
  (was `V * Diagonal(sqS)`).
- The QR (`qr(mati)`, `qr(matj)`), `_thin_q`, and the `qi*mati`/`qj*matj` re-application are
  already complex-correct (standard `A = Q·R` reconstruction) — unchanged.

### 2.4 What is already correct (must NOT change)
Every `Diagonal(S)*V'` / `copy(V')` in `raw2mps`, `cano_to!`, `swap!`, `compress!`,
`compress_opt!` materializes the genuine SVD identity `U·S·V'`; one side carries the full
`S·V'` (or `V'` with `S` on the other side), so non-conjugated downstream contraction
reconstructs `U·S·V'` exactly. These are correct for complex and stay. `tsvd`/`rsvd` already
handle complex (`svd` returns complex `U`,`V`, real `S`; `count`/`sum` operate on real `S`).

## 3. Behavior notes

- A complex contraction's value can be complex; `eat!` case (a) returns `(log|r|, 0, r/|r|)`
  with a complex unit `phase`, and `contraction!` accumulates `psi *= phase`. The final value
  `exp(lnZ) * psi` is complex and is what tests compare to `exact_contract`.
- `lnZ` accumulates `log(norm)` where `norm` is a real magnitude (Frobenius/L2 or max-abs), so
  `lnZ` stays real; the phase lives in `psi`.

## 4. Testing

`test/test_complex.jl` (CPU, runs always) + complex cases added to `test/test_gpu.jl` (GPU,
guarded by `CUDA.functional()`):

- **Unit:** `eat!` of two complex leaf vectors returns `sum(vi .* vj)` (not `dot`); a
  `cut_bondim!` (and `cut_bondim_opt!`) on a genuinely low-rank `ComplexF64` bond preserves the
  contracted value (lossless).
- **Integration (exact mode, `Dmax<0`):** random `ComplexF64` generic networks (chain, loop,
  small loopy) — `exp(lnZ)*psi ≈ exact_contract(tensors, ixs)` to `rtol~1e-10`. This exercises
  `eat!` and the SVD-split paths.
- **Integration (finite Dmax):** a complex network whose shared bonds are low-rank (so
  truncation to a sufficient `Dmax` is lossless) — CATN result `≈` oracle, exercising
  `cut_bondim!`/`cut_bondim_opt!` (both `svdopt` settings).
- **Regression:** the existing real-valued suite (146 tests) stays green — every fix is a no-op
  for real input (`transpose==adjoint`, `conj(V)==V`, `sum(vi.*vj)==dot(vi,vj)`).
- **GPU smoke:** a complex `CuArray` network contraction `==` its CPU counterpart (and the
  oracle), `rtol` loosened for the device.

## 5. Risks

- The `cut_bondim` derivation is the subtle part; the low-rank-complex-bond value-preservation
  test is the gate that proves it (a wrong `conj`/`transpose` placement fails it).
- Confirm no OTHER bond contraction conjugates: the audit found only these three (all `ein"…"`
  contractions are non-conjugating; the remaining `'` are SVD reconstructions). The exact-mode
  complex integration test would expose any missed spot.

## 6. Deliverables

- Conjugation fixes in `eat!`, `cut_bondim!`, `cut_bondim_opt!`.
- `test/test_complex.jl` + complex GPU cases; full suite green (real + complex, CPU + GPU).
- README/spec "complex unsupported" limitation removed.
