# CATN.jl Complex-Number Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CATN correct for complex-valued tensors by fixing the three tensor-network bond contractions that wrongly conjugate (`eat!` case (a); `cut_bondim!`; `cut_bondim_opt!`), validated against the OMEinsum exact oracle.

**Architecture:** Tensor-network bonds are contracted WITHOUT conjugation. Replace the three conjugating adjoints with non-conjugating `transpose`/`sum`, and conjugate the SVD V-factor in the cut-bond split so the truncated bond reconstructs `U·S·V'` under non-conjugated downstream contraction. Every other `'` in the codebase is a genuine SVD reconstruction and stays.

**Tech Stack:** Julia 1.12 (juliaup at `/mnt/home/xgao1/.juliaup/bin/julia`), OMEinsum, LinearAlgebra, Test; CUDA (test-only) for the GPU smoke test.

**Reference:** spec `docs/superpowers/specs/2026-06-28-catn-complex-support-design.md`. The Python reference (`/mnt/home/xgao1/project/tnmp/catn/`) uses plain `@`/`.T` (non-conjugating) for these contractions.

## Global Constraints

- Julia via the juliaup binary `/mnt/home/xgao1/.juliaup/bin/julia` (PATH `julia`); never `module load julia`. Full suite: `julia --project=. -e 'using Pkg; Pkg.test()'`; faster: `julia --project=. -e 'include("test/runtests.jl")'`.
- Tensor-network BOND contractions must be non-conjugating. The three fixes: `dot(vi,vj)`→`sum(vi .* vj)`; merge `* matj'`/`* rj'`→`* transpose(matj)`/`* transpose(rj)`; cut-bond split `matj = V*Diagonal(sqS)`→`matj = conj(V)*Diagonal(sqS)`.
- Do NOT touch the SVD-reconstruction adjoints (`Diagonal(S)*V'`, `copy(V')`) in `raw2mps`/`cano_to!`/`swap!`/`compress!`/`compress_opt!` — they are correct for complex.
- Every fix is a no-op for real input (`transpose==adjoint`, `conj(V)==V`, `sum(vi.*vj)==dot(vi,vj)` for real). The existing 146-test suite (real, CPU+GPU) MUST stay green.
- The final contracted value for complex is `exp(lnZ) * psi` (psi carries the phase). Tests compare this to `exact_contract(tensors, ixs)[]`.
- GPU tests guarded by `CUDA.functional()`.
- Commit after each task with trailer `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

## File Structure

```
src/
  mps_node.jl        # eat! case (a): dot -> sum(.*)   (Task 1)
  tensor_network.jl  # cut_bondim! + cut_bondim_opt!: transpose merge + conj(V) split  (Task 2)
test/
  test_complex.jl    # NEW: complex unit + integration tests (Tasks 1-2)
  test_gpu.jl        # add complex GPU smoke (Task 3)
  runtests.jl        # include test_complex.jl (Task 1)
README.md            # remove "complex unsupported" limitation (Task 3)
```

---

### Task 1: Fix `eat!` case (a) + complex exact-mode validation

**Files:**
- Modify: `src/mps_node.jl` (`eat!` case (a), line ~475)
- Create: `test/test_complex.jl`
- Modify: `test/runtests.jl` (include it)

**Interfaces:**
- Consumes: `TensorNetwork`, `contraction!`, `MPSNode`, `eat!`, `find_neighbor`, `mps2raw` (existing); `exact_contract` (test/exact.jl, included earlier in runtests.jl).
- Produces: complex-correct `eat!`; a complex test file other tasks extend.

- [ ] **Step 1: Write the failing tests** — create `test/test_complex.jl`

```julia
using CATN
using CATN: MPSNode, TensorNetwork, contraction!, eat!, find_neighbor, mps2raw
using LinearAlgebra, Test
# exact_contract is available from exact.jl (included earlier in runtests.jl)

@testset "complex" begin
    @testset "eat! both-leaves uses non-conjugating product" begin
        u = randn(ComplexF64, 4); v = randn(ComplexF64, 4)
        ni = MPSNode(u, [2]; norm_method=0)
        nj = MPSNode(v, [1]; norm_method=0)
        lognorm, err, phase = eat!(ni, nj, 1, 1)
        @test isempty(ni.mps)
        @test exp(lognorm) * phase ≈ sum(u .* v)        # NON-conjugating; dot(u,v) would conjugate u
        @test !(exp(lognorm) * phase ≈ dot(u, v))        # guard: must differ from the conjugating dot
    end

    @testset "complex generic contraction matches oracle (exact mode)" begin
        networks = [
            ([randn(ComplexF64,3,4), randn(ComplexF64,4,5), randn(ComplexF64,5,3)],
             [[:a,:b],[:b,:c],[:c,:a]]),                                   # chain/loop
            ([randn(ComplexF64,2,3,4), randn(ComplexF64,2), randn(ComplexF64,3), randn(ComplexF64,4)],
             [[:a,:b,:c],[:a],[:b],[:c]]),                                 # star/tree
            ([randn(ComplexF64,2,3),randn(ComplexF64,3,2),randn(ComplexF64,2,3),randn(ComplexF64,3,2)],
             [[:a,:b],[:b,:c],[:c,:d],[:d,:a]]),                           # loop of 4
        ]
        for (ts, ixs) in networks
            ref = exact_contract(ts, ixs)[]
            for sel in 0:2
                tn = TensorNetwork(ts, ixs; Dmax=-1, chi=10_000, select=sel,
                                   reverse=true, compress=true, norm_method=1)
                lnZ, err, psi = contraction!(tn)
                @test exp(lnZ) * psi ≈ ref rtol=1e-10
            end
        end
    end
end
```

- [ ] **Step 2: Run, expect failure**

Run: `julia --project=. -e 'include("test/runtests.jl")'` (after wiring in Step 4) — or temporarily `julia --project=. -e 'using Pkg; Pkg.activate("."); include("test/exact.jl"); include("test/test_complex.jl")'`.
Expected: FAIL — `eat!` uses `dot` (conjugates), so the both-leaves test's `≈ sum(u.*v)` fails and `!(≈ dot(u,v))` fails; some exact-mode complex contractions mismatch the oracle.

- [ ] **Step 3: Fix `eat!` case (a)** — `src/mps_node.jl` line ~475, change:
```julia
        r  = dot(vi, vj)
```
to:
```julia
        r  = sum(vi .* vj)   # non-conjugating bond contraction (dot would conjugate vi)
```

- [ ] **Step 4: Wire the test** — `test/runtests.jl`: add `include("test_complex.jl")` inside the `@testset`.

- [ ] **Step 5: Run, expect pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS — the both-leaves test and all exact-mode complex networks match the oracle; the existing real suite stays green. (If an exact-mode complex network still mismatches, a non-conjugation bug exists OUTSIDE `eat!`/cut-bond — STOP and report, do not loosen tolerance.)

- [ ] **Step 6: Commit**

```bash
git add src/mps_node.jl test/test_complex.jl test/runtests.jl
git commit -m "fix: eat! contracts the both-leaves bond without conjugation (complex)"
```

---

### Task 2: Fix `cut_bondim!` / `cut_bondim_opt!` + complex finite-Dmax validation

**Files:**
- Modify: `src/tensor_network.jl` (`cut_bondim!` ~lines 363,379; `cut_bondim_opt!` ~lines 448,464)
- Modify: `test/test_complex.jl` (add cut-bond unit + finite-Dmax integration tests)

**Interfaces:**
- Consumes: `cut_bondim!`, `cut_bondim_opt!`, `TensorNetwork`, `contraction!`, `mps2raw` (existing); `exact_contract`.
- Produces: complex-correct bond truncation.

**Derivation (for the implementer):** the shared physical bond is contracted non-conjugated, so the joint matrix is `merged = mati * transpose(matj)`. SVD `merged = U*Diagonal(S)*V'`. The truncated bond is contracted non-conjugated downstream, so the split must satisfy `Σ_k mati'[x,k]·matj'[y,k] = merged[x,y] = Σ_k U[x,k]·S[k]·conj(V[y,k])` ⇒ `mati' = U*Diagonal(√S)` (unchanged), `matj' = conj(V)*Diagonal(√S)`. The QR pre-reduction in `cut_bondim_opt!` (`qr`, `_thin_q`, `qi*mati`, `qj*matj`) is already complex-correct and stays.

- [ ] **Step 1: Write the failing tests** — append to `test/test_complex.jl` (inside the `@testset "complex"`)

```julia
    @testset "cut_bondim preserves complex low-rank bond" begin
        # two order-2 complex nodes whose shared bond is rank 2, stored as dim 4
        P = randn(ComplexF64,3,2) * randn(ComplexF64,2,4)   # 3x4, rank 2 on the :m leg
        Qm = randn(ComplexF64,4,2) * randn(ComplexF64,2,3)  # 4x3, rank 2
        ixs = [[:a,:m],[:m,:b]]
        for cutter! in (CATN.cut_bondim!, CATN.cut_bondim_opt!)
            tn = TensorNetwork([P,Qm], ixs; chi=1000, Dmax=2)
            cutter!(tn, 1, 2)                                  # :m is leg 2 of node 1
            @test size(tn.tensors[1].mps[2], 2) == 2          # truncated to Dmax=2
            got = ein"am,mb->ab"(mps2raw(tn.tensors[1]), mps2raw(tn.tensors[2]))
            @test got ≈ P * Qm atol=1e-8                      # value preserved (lossless: bond was rank 2)
        end
    end

    @testset "complex finite-Dmax contraction matches oracle" begin
        # loop of 4 with genuinely low-rank complex bonds (rank 2), Dmax=2 lossless
        mats = [randn(ComplexF64,4,2)*randn(ComplexF64,2,4) for _ in 1:4]   # each 4x4, rank 2
        ts = [reshape(mats[1], 4,4), reshape(mats[2],4,4), reshape(mats[3],4,4), reshape(mats[4],4,4)]
        ixs = [[:a,:b],[:b,:c],[:c,:d],[:d,:a]]
        ref = exact_contract(ts, ixs)[]
        for opt in (true, false)
            tn = TensorNetwork(ts, ixs; Dmax=2, chi=64, select=1, compress=true, svdopt=opt)
            lnZ, err, psi = contraction!(tn)
            @test exp(lnZ) * psi ≈ ref rtol=1e-6
        end
    end
```
(Add `using CATN: cut_bondim!, cut_bondim_opt!` and `using OMEinsum: @ein_str` / `using OMEinsum` at the top of `test_complex.jl` if `ein"..."`/the cutters aren't already in scope. `cut_bondim!`/`cut_bondim_opt!` are exported; `ein` comes from OMEinsum — add `using OMEinsum`.)

- [ ] **Step 2: Run, expect failure**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: FAIL — the complex low-rank cut tests mismatch (`got ≈ P*Qm` fails) because the merge conjugates and the V-split lacks `conj`.

- [ ] **Step 3: Fix `cut_bondim!`** — `src/tensor_network.jl`:
  - line ~363: `merged = mati * matj'` → `merged = mati * transpose(matj)`
  - line ~379: `matj = V * Diagonal(sqS)` → `matj = conj(V) * Diagonal(sqS)`

- [ ] **Step 4: Fix `cut_bondim_opt!`** — `src/tensor_network.jl`:
  - line ~448: `merged = ri * rj'` → `merged = ri * transpose(rj)`
  - line ~464: `matj = V * Diagonal(sqS)` → `matj = conj(V) * Diagonal(sqS)`

- [ ] **Step 5: Run, expect pass**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS — complex low-rank cut value-preservation (both `cut_bondim!` and `cut_bondim_opt!`) and complex finite-Dmax contraction match the oracle; the real suite stays green. (If a complex cut test still fails, re-check the `conj`/`transpose` placement against the derivation — do NOT loosen tolerances.)

- [ ] **Step 6: Commit**

```bash
git add src/tensor_network.jl test/test_complex.jl
git commit -m "fix: cut_bondim non-conjugating merge + conj(V) split (complex)"
```

---

### Task 3: Complex GPU smoke test + remove the limitation docs

**Files:**
- Modify: `test/test_gpu.jl` (add a complex GPU case inside the `CUDA.functional()` block)
- Modify: `README.md` (remove the "complex unsupported" limitation)

**Interfaces:**
- Consumes: `TensorNetwork`, `contraction!` (now complex-correct), `exact_contract`, CUDA.

- [ ] **Step 1: Add the complex GPU smoke test** — in `test/test_gpu.jl`, inside the existing `@testset "GPU"` block (after the Float32 test), add:

```julia
        @testset "complex GPU contraction matches CPU + oracle" begin
            ts = [randn(ComplexF64,3,4), randn(ComplexF64,4,5), randn(ComplexF64,5,3)]
            ixs = [[:a,:b],[:b,:c],[:c,:a]]
            ref = exact_contract(ts, ixs)[]
            gtn = TensorNetwork([CuArray(t) for t in ts], ixs; Dmax=-1, chi=10_000)
            lnZ, err, psi = contraction!(gtn)
            @test exp(lnZ) * psi ≈ ref rtol=1e-7
        end
```

- [ ] **Step 2: Run the full suite (CPU + GPU)**

Run: `julia --project=. -e 'using Pkg; Pkg.test()'`
Expected: PASS — the complex GPU contraction matches the oracle on the RTX 6000; everything else stays green. (If complex `svd` on the GPU errors, report it — cuSOLVER supports complex SVD, but flag any issue rather than skipping.)

- [ ] **Step 3: Remove the "complex unsupported" limitation** — in `README.md`, delete/replace the sentence in the Limitations section stating complex tensors are not supported/validated. Replace with a positive note, e.g.:
```markdown
Complex-valued tensors are supported (contractions are non-conjugating tensor contractions);
the result `exp(lnZ) * psi` carries the complex phase in `psi`.
```
If the GPU/Limitations note also lives in the spec docs, leave the historical spec files unchanged (they are dated design records) — only update README (user-facing).

- [ ] **Step 4: Commit**

```bash
git add test/test_gpu.jl README.md
git commit -m "test: complex GPU smoke; docs: complex tensors now supported"
```

---

## Self-Review

**Spec coverage:** §2.1 eat! → Task 1; §2.2 cut_bondim! → Task 2; §2.3 cut_bondim_opt! → Task 2; §2.4 (don't touch SVD reconstructions) → enforced via Global Constraints + tests; §4 testing (eat! unit + exact-mode integration → Task 1; cut-bond unit + finite-Dmax integration → Task 2; regression = full suite each task; GPU smoke → Task 3); §6 remove limitation docs → Task 3. All covered.

**Placeholder scan:** every step has concrete code/commands and exact before→after edits with line hints. No "TBD"/"add error handling"/"similar to".

**Type/name consistency:** fixes reference real function/var names confirmed from the source (`dot(vi,vj)` at mps_node.jl:475; `merged = mati * matj'` / `matj = V * Diagonal(sqS)` in cut_bondim!; `merged = ri * rj'` / `matj = V * Diagonal(sqS)` in cut_bondim_opt!). `exact_contract`, `cut_bondim!`, `cut_bondim_opt!`, `contraction!` are all existing/exported. Tests compare `exp(lnZ)*psi ≈ exact_contract(...)[]`, consistent with the established pattern.
