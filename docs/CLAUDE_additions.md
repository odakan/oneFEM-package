# CLAUDE.md — Additions
#
# INSERT these sections into your existing CLAUDE.md.
# Section order: paste "Architecture Reference" right after "Project Overview",
# paste the rest at the end.
# ─────────────────────────────────────────────────────────────────────────────

## Architecture Reference

**Read `docs/oneFEM_blueprint.md` before writing any code in this repository.
This is mandatory, not optional.** It contains:
- The research vision and why this codebase exists (Section 0)
- Every interface contract (Sections 5–8)
- The OpenSees C++ → Python mapping (Section 17)
- All known pitfalls (Section 14)
- The WP → paper → grant chain (Section 16)

If you have not read it, stop and read it now.

---

## Current Implementation Status

This table is the ground truth for what exists vs. what is planned.
Check it before starting any task. If you finish a WP, update this table.
✅ = implemented AND benchmark passes
⚠️ = implemented but not fully validated  
🔲 = planned, not started
⚠️ Stub = file exists but raises NotImplementedError — do NOT implement without WP assignment

### Domain Layer

| Component | Status | Location | Notes |
|---|---|---|---|
| `Domain` | ✅ Working | `model/main.py` | Static + dynamic + eigen |
| `Node` base | ✅ Working | `model/node/main.py` | |
| `Node22`, `Node23`, `Node36` | ✅ Working | `model/node/` | |
| `Node37` (warping DOF) | ⚠️ Partial | `model/node/` | Exists, not benchmarked |
| `SP_Constraint` | ✅ Working | `model/constraint/` | Homogeneous only |
| `MP_Constraint` | 🔲 Planned | — | Needed for rigid diaphragm |
| `LoadPattern` | ✅ Working | `model/pattern/` | |
| `TimeSeries` (Constant, Linear, Path) | ✅ Working | `model/tseries/` | |
| `TimeSeries` (Trig) | 🔲 Planned | — | Needed for harmonic loading |

### Element Layer

| Component | Status | Location | Notes |
|---|---|---|---|
| `Truss` (2D/3D) | ✅ Working | `model/element/truss/` | Benchmarked in `src/truss.py` |
| `ElasticBeamColumn` | ✅ Exists | `model/element/beam/` | Needs CrdTransf |
| `ZeroLength` | ✅ Exists | `model/element/zerolength/` | |
| `Shell` | 🔲 Planned | — | |
| `Solid` | 🔲 Planned | — | |
| `forceBeamColumn` (fiber) | 🔲 Planned | — | Needs FiberSection first |
| `CrdTransf` (Linear2d/3d) | 🔲 Planned | — | Required before beam is reliable |
| `CrdTransf` (Corotational) | 🔲 Planned | — | Geometric nonlinearity |
| `Section` (Rectangular) | ✅ Working | `model/element/section/` | |
| `SectionForceDeformation` | 🔲 Planned | — | For beam-column elements |
| `FiberSection` | 🔲 Planned | — | Needs SectionForceDeformation |

### Material Layer

| Component | Status | Location | Notes |
|---|---|---|---|
| `Elastic` | ✅ Working | `model/material/uniaxial/` | Benchmarked |
| `ElasticPerfectlyPlastic` | ✅ Working | `model/material/uniaxial/` | Benchmarked in `src/epp_truss.py` |
| `Steel01` | 🔲 Planned | — | Priority for earthquake engineering |
| `Concrete01` | 🔲 Planned | — | Priority for earthquake engineering |
| `Hardening` (kinematic+isotropic) | 🔲 Planned | — | |
| `ElasticIsotropic` (nD) | 🔲 Planned | `model/material/nD/` | |
| `J2Plasticity` (nD) | 🔲 Planned | — | |
| `ParallelMaterial` | 🔲 Planned | — | Needs `getCopy()` on all materials |
| `NeuralMaterialSurrogate` | 🔲 WP19 | — | The Oracle — research frontier |

### Analysis Layer

| Component | Status | Location | Notes |
|---|---|---|---|
| `Linear` algorithm | ✅ Working | `analysis/algorithm/` | |
| `Newton` algorithm | ✅ Working | `analysis/algorithm/` | Benchmarked |
| `KrylovNewton` | ✅ Exists | `analysis/algorithm/` | Not benchmarked |
| `LoadControl` | ✅ Working | `analysis/integrator/` | |
| `DisplacementControl` | ✅ Exists | `analysis/integrator/` | |
| `Newmark` | ✅ Working | `analysis/integrator/` | Benchmarked in `src/dynamic_truss.py` |
| `CentralDifference` | ✅ Exists | `analysis/integrator/` | |
| `HHT / GeneralizedAlpha` | 🔲 Planned | — | Numerical damping |
| `ArcLength` | 🔲 Planned | — | Snap-through problems |
| `FullGeneral` solver | ✅ Working | `analysis/system/` | scipy dense |
| `UMFPACK` solver | ✅ Working | `analysis/system/` | Linux only |
| `CUSPARSESolver` | 🔲 WP13 | — | `cupyx.scipy.sparse.linalg` backend |
| Parallel element assembly | 🔲 WP11 | — | `concurrent.futures.ThreadPoolExecutor` |
| MPI domain decomp | 🔲 WP10 | — | `mpi4py` — distributed ranks |
| `NormUnbalance` test | ✅ Working | `analysis/test/` | |
| `NormDispIncr` test | ✅ Working | `analysis/test/` | |
| `EnergyIncr` test | 🔲 Planned | — | |
| Rayleigh damping | ⚠️ Partial | `model/main.py` | Interface exists, not fully wired |
| Eigen solver | ✅ Working | `analysis/eigen/` | Benchmarked in `src/eigen_truss.py` |

### Output / Infrastructure

| Component | Status | Location | Notes |
|---|---|---|---|
| `NodeRecorder` | ✅ Working | `output/recorder/` | |
| `ElementRecorder` | ✅ Working | `output/recorder/` | |
| `ModeShapeRecorder` | ✅ Working | `output/recorder/` | |
| `HDF5Archive` | 🔲 WP8 | — | Checkpoint/restart |
| `UnifiedAllocator` | 🔲 WP23-24 | — | SoC zero-copy — hardware thesis |
| `benchmark_unified_memory.py` | ✅ Done | `benchmarks/` | Grant deliverable — do not break |

---

## Stop and Ask Rules

**Do not proceed autonomously on these decisions. Stop and ask the PI first.**

1. **Any change to the `Material` interface** (adding, removing, or renaming
   methods on the abstract base or any method that `Section` calls).
   Reason: breaks the neural surrogate injection point (WP19) and FE² nesting.

2. **Any change to the commit/revert call order in `Domain._commit()`.**
   Reason: the pipeline order is load-bearing — nodes before elements.
   Getting this wrong produces silent wrong answers, not crashes.

3. **Adding a new top-level directory** under `src/oneFEM/`.
   Reason: new directories = new WPs. This is a research planning decision,
   not a code organization decision.

4. **Implementing anything in Circle C** (FE², Oracle-guided stepping,
   parallel MPI) without an explicit WP assignment.
   Reason: Circle C work has dependencies on Circle B that aren't complete yet.

5. **Modifying `benchmark_unified_memory.py`**.
   Reason: this is a grant deliverable. Any change to methodology must be
   intentional and documented.

6. **Changing the sparse matrix format** (COO/CSR/LIL) in the SOE assembly.
   Reason: this choice maps directly to how CuPy and mpi4py consume the matrix.
   Changing it breaks the parallel and GPU assembly paths.

7. **Suggesting or implementing a C++ extension, Cython wrapper, or pybind11
   binding for performance.**
   Reason: Python backends (CuPy, mpi4py, PyTorch) are the planned performance
   path for all WPs. A C++ extension is only justified if a Python backend
   genuinely cannot meet the requirement. If you think one is needed, stop
   and explain why to the PI — do not implement it autonomously.

---

## Task Cookbook

Exact sequence for the most common tasks. Follow this order precisely.

### Adding a New Uniaxial Material

```
1. Read the existing material you're closest to (Elastic or EPP)
2. Create model/material/uniaxial/your_material.py
3. Implement in this order:
   a. __init__(self, mat_id, ...)  ← super().__init__(mat_id) FIRST
   b. _setTrialStrain(strain, strain_rate=0.0)
   c. getStress()
   d. getTangent()
   e. getInitialTangent()
   f. getStrain()
   g. _commitState()       ← deep copy every state variable
   h. _revertToLastCommit() ← deep copy committed → trial
   i. revertToStart()      ← zero everything
   j. getCopy()            ← return self.__class__(self._tag, *params) with fresh state
4. Write a benchmark script in src/ that:
   a. Applies monotonic loading to failure (or yield)
   b. Applies cyclic loading (tension → compression → tension)
   c. Tests commit/revert: load → commit → load more → revert → verify back to committed
   d. Verifies against analytical solution with rel error < 1e-6
5. Run the benchmark. All sub-tests must PASS before opening a PR.
6. Update the Status table above (⚠️ or ✅)
```

### Adding a New Element

```
1. Identify which Node specializations this element connects
   (e.g., Truss connects Node22 or Node36)
2. Identify which Section/Material it uses
3. Create model/element/{type}/main.py
4. Implement in this order:
   a. __init__(self, tag, node_tags, section/material, ...)
   b. _domain(self)        ← geometry setup, store node refs
   c. getTangentStiff()    ← local K via material.getTangent()
   d. getInitialStiff()    ← local K via material.getInitialTangent()
   e. getResistingForce()  ← local F via material.getStress()
   f. _update(self)        ← extract local disps, call material._setTrialStrain()
   g. _commit(self)        ← commit material, store committed force
   h. _revert(self)        ← revert material
   i. revertToStart()
   j. getMass()            ← return zero Matrix if no mass
   k. getDamp()            ← return zero Matrix initially
   l. zeroLoad() / addLoad() / getResistingForceIncInertia()
   If beam: wrap all local↔global transforms through a CrdTransf object
5. Write benchmark in src/ verifying against analytical stiffness matrix
6. Run all existing benchmarks — ensure nothing breaks
7. Update Status table
```

### Adding a New Benchmark Script

```
1. Name: src/{description}.py
2. Header comment block:
   - What is being tested
   - Analytical reference (equation or paper citation)
   - Expected pass criterion (rel error < X)
3. Structure:
   def test_case_name():
       # build model
       # run analysis
       # compute error vs. analytical
       result = "PASS" if err < tol else "FAIL"
       print(f"  {test_case_name}: {result}  (err={err:.2e}, tol={tol:.2e})")
       return result == "PASS"

   if __name__ == "__main__":
       results = [test_1(), test_2(), ...]
       print(f"\n{sum(results)}/{len(results)} PASS")
4. Add to the benchmark table in CLAUDE.md
```

### Debugging a Newton Convergence Failure

```
Step 1: Check tangent first — getTangent() wrong is the #1 cause
  - Add print(element.getTangentStiff()) before the Newton loop
  - Is it positive definite? Is the diagonal dominant?
  - Check section: is self._C updated from getTangent() not getStress()?

Step 2: Check residual assembly
  - Print F_ext and F_int separately
  - Are units consistent? (N vs kN, m vs mm)
  - Is the load being applied to the right DOF?

Step 3: Check pipeline order
  - Is domain._commit() called BEFORE element._update()/_commit()?
  - Is element._update() pushing strain to the material?

Step 4: Check commit/revert
  - Add assert self._eps_commit != self._eps_trial after a revert
  - If they're equal, you have an aliasing bug
```
