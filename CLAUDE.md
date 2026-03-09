# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**oneFEM** is a Python finite element modeling and analysis package (v0.0.1, alpha). It supports 2D/3D simulations with various element types, materials, and analysis methods. Licensed under GPLv3.

## Architecture Reference

**Read `docs/oneFEM_blueprint.md` before writing any code in this repository.
This is mandatory, not optional.** It contains:
- The research vision and why this codebase exists (Section 0)
- Every interface contract (Sections 5–8)
- The OpenSees C++ → Python mapping (Section 17)
- All known pitfalls (Section 14)
- The WP → paper → grant chain (Section 16)

If you have not read it, stop and read it now.

**Read `docs/oneFEM_v2_architecture.md` before any refactoring work.** It is the governing
design document for the v2 architecture. It defines:
- The four-pillar model: node, element, kinematics, material
- `physics_family` string compatibility system
- Full element API contract (get_B, get_H, get_F, get_dN_dX, etc.)
- Branching rules (what you may and may NOT branch on)
- Audit table of 16 current violations and 6-phase refactor plan
- Self-contained element files (no shared `isoparametric.py`)

## Backend Strategy (Hardware Abstraction)

oneFEM is **pure Python with swappable array backends**. All numerical computation uses a numpy-compatible API. The abstraction point is `_systools/backend.py` (planned) — one import swap targets different hardware:

| Backend | Hardware | WP | Role |
|---|---|---|---|
| **numpy** | CPU | current | Default, always works |
| **CuPy** | NVIDIA GPU | WP13 | Drop-in numpy for GPU assembly/solve |
| **JAX** | GPU, TPU, XLA | — | `jax.grad` → auto-tangents for any material; eliminates hand-coded C=dσ/dε |
| **PyTorch** | NVIDIA, Apple MPS, Intel XPU, TPU | WP19 | Neural material surrogate bridge |
| **MLX** | Apple Silicon (unified memory) | WP23-24 | Zero-copy CPU↔GPU, SoC thesis target |
| **dpnp** (Intel oneAPI) | Intel GPU, FPGA | — | Intel SoC path |

**Design rules:** (1) numpy API standard (NEP 47) everywhere. (2) `_systools/backend.py` owns `import numpy as xp` — one-line swap to retarget. (3) Integration loops must be batchable for GPU. (4) UnifiedAllocator (WP23-24) manages array placement. (5) No C++/Cython/pybind11 (ADR-008).

**`_systools/backend.py` exists.** Data wrappers import `np` from it. Code outside `_systools/data/` still uses `import numpy as np` directly — migrate module-by-module during WP13.

## Build & Install

```bash
# Install in development mode
pip install -e .

# Install dependencies only
pip install -r requirements.txt   # numpy, scipy, matplotlib, ipython

# Optional: UMFPACK sparse solver (Linux only)
sudo apt install ninja-build swig libopenblas-dev liblapack-dev libsuitesparse-dev
pip install numpy==1.26.4
pip install --force-reinstall --no-deps scipy scikit-umfpack
```

Package source lives in `src/oneFEM/` (setuptools with `package_dir={"": "src"}`).

## Testing

No automated test framework exists yet. Validation is done via benchmark scripts:

```bash
cd src && python truss.py
```

**Benchmarks (run from `src/`):**

| Script | Tests | What |
|---|---|---|
| `truss.py` | ALL PASS | 3D triangle truss vs analytical stiffness |
| `dynamic_truss.py` | ALL PASS | SDOF Newmark vs closed-form, 3 configs (err < 1e-4) |
| `eigen_truss.py` | ALL PASS | Eigenvalue/modal analysis (err < 1e-10) |
| `epp_truss.py` | ALL PASS | EPP Newton-Raphson, elastic regression |
| `examples/beam.py` | 10 PASS | BeamColumn 2D/3D: stiffness, cantilever, inclined (err < 1e-10) |
| `examples/column_buckling.py` | 8 PASS | PDelta/Corot CrdTransf, P_cr eigenvalue (err < 3%) |
| `examples/quad4_patch_test.py` | 6 PASS | MacNeal-Harder patch test (err < 1e-15) |
| `examples/quad4_cooks_membrane.py` | 2 PASS | Cook's membrane convergence (5 meshes) |
| `examples/corot_benchmarks.py` | 5 PASS | Snap-through arch (Crisfield ref), Lee's frame |
| `examples/hex8_benchmarks.py` | 39 PASS | Patch (12), cylinder (2), cantilever (1), Cook's 3D (1), incomp patch (6), locking (3), TL-UL (4), Corot (3), Corot large (7) |
| `examples/hex8_curved_cantilever.py` | Visualization | Bathe & Bolourchi 1979 curved cantilever, TL/UL/Corot/Linear, locking study (not a gate test) |
| `examples/corot_continuum_benchmarks.py` | 5 PASS | EICR Quad4: convergence, Corot==TL==UL, patch |
| `examples/quad4_benchmarks.py` | 8 PASS | B1-B8: patch, Cook's, shear, cantilever NL, snap-through, Lee's, buckling, cylinder |

- `examples/test_model/` — model definition files (`.txt` format)

## Current Implementation Status

<!-- # verified 2026-03-09 -->
This table is the ground truth for what exists vs. what is planned.
Check it before starting any task. If you finish a WP, update this table.

### Domain Layer

| Component | Status | Location | Notes |
|---|---|---|---|
| `Domain` | ✅ Working | `model/main.py` | Static + dynamic + eigen |
| `Node` base | ✅ Working | `model/node/main.py` | |
| `Node22`, `Node23`, `Node36` | ✅ Working | `model/node/` | |
| `Node24`, `Node33`, `Node34` | ⚠️ Partial | `model/node/` | Exist, not benchmarked; Node33 used by Hex8 (working) |
| `Node37` (warping DOF) | ⚠️ Partial | `model/node/` | Exists, not benchmarked |
| `SP_Constraint` | ✅ Working | `model/constraint/` | Homogeneous only |
| `MP_Constraint` (EqualDOF) | ⚠️ Stub | `model/constraint/multipoint/` | Exists as empty stub |
| `LoadPattern` | ✅ Working | `model/pattern/` | |
| `TimeSeries` (Constant, Linear, Path) | ✅ Working | `model/tseries/` | |
| `TimeSeries` (Trig) | ✅ Working | `model/tseries/trig_series.py` | Implemented |

### Element Layer

| Component | Status | Location | Notes |
|---|---|---|---|
| `Truss` (2D/3D) | ✅ Working | `model/element/truss/` | Benchmarked in `src/truss.py` |
| `ElasticBeamColumn2d` | ✅ Working | `model/element/beam/` | Benchmarked in `src/examples/beam.py` |
| `ElasticBeamColumn3d` | ✅ Working | `model/element/beam/` | Benchmarked in `src/examples/beam.py` |
| `ZeroLength` | ✅ Exists | `model/element/zerolength/` | |
| `ShellQ4` | ⚠️ Stub | `model/element/shell/` | Empty stub |
| `Quad4` (continuum) | ✅ Working | `model/element/continuum/quad4.py` | Patch test 6/6 PASS, Cook's membrane convergence PASS |
| `Hex8` (continuum, B-bar) | ✅ Working | `model/element/continuum/hex8.py` | B-bar + incompatible modes, 32 benchmarks PASS (patch, cylinder, cantilever, Cook's 3D, locking, TL-UL, Corot) |
| `Solid` (Tri, Brick) | ⚠️ Stub | `model/element/solid/` | Empty stubs |
| `forceBeamColumn` (fiber) | 🔲 Planned | — | Needs FiberSection first |
| `CrdTransf` (Linear2d) | ✅ Working | `model/element/coordTransformation/` | Benchmarked |
| `CrdTransf` (Linear3d) | ✅ Working | `model/element/coordTransformation/` | Benchmarked |
| `CrdTransf` (PDelta2d/3d) | ✅ Working | `model/element/coordTransformation/` | Benchmarked in `src/examples/column_buckling.py` |
| `CrdTransf` (Corot2d/3d) | ✅ Working | `model/element/coordTransformation/` | Benchmarked in `src/examples/column_buckling.py` |
| `Section` (Rectangular) | ✅ Working | `model/element/section/` | |
| `SectionForceDeformation` | 🔲 Planned | — | For beam-column elements |
| `FiberSection` | ⚠️ Stub | `model/element/section/fiber_section.py` | Empty stub; needs SectionForceDeformation |

### Material Layer

| Component | Status | Location | Notes |
|---|---|---|---|
| `Elastic` | ✅ Working | `model/material/uniaxial/` | Benchmarked |
| `ElasticPerfectlyPlastic` | ✅ Working | `model/material/uniaxial/` | Benchmarked in `src/epp_truss.py` |
| `Steel01` | 🔲 Planned | — | Priority for earthquake engineering |
| `Concrete01` | 🔲 Planned | — | Priority for earthquake engineering |
| `Hardening` (kinematic+isotropic) | 🔲 Planned | — | |
| `ElasticIsotropic` (nD) | ✅ Working | `model/material/nD/elastic_isotropic.py` | PlaneStress/PlaneStrain/3D, CTensor-based, benchmarked via Quad4 patch test |
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
| `DisplacementControl` | ✅ Working | `analysis/integrator/` | Benchmarked in `src/examples/column_buckling.py` |
| `Newmark` | ✅ Working | `analysis/integrator/` | Benchmarked in `src/dynamic_truss.py` |
| `CentralDifference` | ✅ Exists | `analysis/integrator/` | |
| `HHT / GeneralizedAlpha` | 🔲 Planned | — | Numerical damping |
| `ArcLength` | 🔲 Planned | — | Snap-through problems |
| `SparseGeneral` (LinearSOE) | ✅ Working | `analysis/system/sparse_general.py` | SparseAssembler + SpSolve, default SOE |
| `UmfPackSOE` (LinearSOE) | ✅ Working | `analysis/system/umfpack.py` | SparseAssembler + UmfPackSolver, Linux only |
| `FullGeneral` solver | ✅ Working | `analysis/system/full_general.py` | scipy dense, solve-only (legacy System) |
| `CUSPARSESolver` | 🔲 WP13 | — | `cupyx.scipy.sparse.linalg` backend |
| Parallel element assembly | 🔲 WP11 | — | `concurrent.futures.ThreadPoolExecutor` |
| MPI domain decomp | 🔲 WP10 | — | `mpi4py` — distributed ranks |
| `NormUnbalance` test | ✅ Working | `analysis/test/` | |
| `NormDispIncr` test | ✅ Working | `analysis/test/` | |
| `EnergyIncr` test | ⚠️ Stub | `analysis/test/energy_increment.py` | Empty stub |
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

## Architecture

### Module Organization

```
src/oneFEM/
├── _systools/          # Internal utilities
│   ├── data/           # Vector, Matrix, Tensor wrappers around NumPy
│   ├── math_tools.py   # Precision comparisons (RTOL=1e-6, ATOL=1e-12)
│   ├── simulation_manager.py  # Top-level orchestrator
│   ├── source.py       # .txt model file parser
│   └── cli.py          # CLI entry point (oneFEM command)
├── model/              # FEM model definition
│   ├── main.py         # Domain — central model container
│   ├── node/           # Node base + specializations by (nDim, nDOF)
│   ├── element/        # Element base + truss, beam, shell, solid, zerolength
│   ├── material/       # Material base + uniaxial, nD
│   ├── pattern/        # Load patterns (Plain + time series)
│   ├── tseries/        # Time series (constant, linear, path)
│   └── constraint/     # Boundary constraints
├── analysis/           # Solver components
│   ├── main.py         # Analysis — orchestrates solution pipeline
│   ├── algorithm/      # Linear, Newton, Krylov-Newton
│   ├── integrator/     # Static (LoadControl, DisplacementControl) / Dynamic (Newmark, CentralDifference)
│   ├── system/         # LinearSOE (assembly+solve), solvers (SpSolve, UmfPack)
│   ├── numberer/       # DOF numbering (Plain, ReverseCuthillMcKee)
│   ├── constraints/    # Constraint handlers (Plain, Penalty)
│   ├── eigen/          # Eigenvalue solvers (Eigen: fullGenLapack, genBandArpack)
│   └── test/           # Convergence tests
├── input/              # Mesh import (CUBIT, GID)
└── output/             # NodeRecorder, ElementRecorder, ModeShapeRecorder
```

### Key Classes and Flow

**SimulationManager** → connects **Domain** (model) with **Analysis** (solver) and **Recorders** (output).

**Analysis pipeline** (working for static linear and nonlinear):
1. `SimulationManager.analyze(nSteps, dt)` → `Analysis._analyze(model, nSteps, dt)`
2. `Domain._domain()` — sequential DOF numbering, element `_domain()` init
3. `Analysis._organize(model)` — separate free (`uu`) / fixed (`pp`) DOF indices
4. Per step: `Domain._assemble()` → `Algorithm.solve(model, uu, pp)` → `Domain._commit(F, u)` → element `_update()`/`_commit()` → `Domain._record(time)`
5. **Newton-Raphson (static nonlinear)**: starts from committed displacements, uses residual R = F_ext - F_int (from `Domain.getInternalForce()`), reassembles tangent K per iteration if `tangent='current'`

**Eigen pipeline** (standalone, no Analysis needed):
1. `model.eigen(numModes, solver)` — calls `_domain()`, `_assemble()`, partitions free/fixed DOFs, solves K·φ=λ·M·φ via `Eigen.solve()`, stores results, triggers `_record_eigen()`
2. `model.modalProperties()` — computes ω, f, T, participation factors, effective modal masses; prints table; returns dict
3. `model.getEigenvalue(mode)` / `model.getEigenvector(mode)` — 1-based accessors

**Core classes:**
- **Domain** (`model/main.py`): Central container holding nodes, elements, patterns, constraints, recorders, and global K/F/u (Matrix/Vector objects). Maintains `__node_map` dict for O(1) node lookup. Key methods: `add()`, `remove()`, `_domain()`, `_assemble()`, `_commit()`, `_record(time)`, `getInternalForce()`, `getCommittedDisp()`, `eigen(numModes)`, `modalProperties()`, `getEigenvalue(mode)`, `getEigenvector(mode)`. Properties: `nodes`, `elements`, `patterns`, `K`, `F`, `u`, `nDOF`.
- **Analysis** (`analysis/main.py`): Orchestrates solution via pluggable Strategy components: Algorithm, Integrator, LinearSOE, Numberer, ConstraintHandler, Test.
- **LinearSOE** (`analysis/system/linear_soe.py`): Owns assembly (SparseAssembler) + solving (LinearSOESolver). Concrete: `SparseGeneral` (SpSolve, default), `UmfPackSOE` (UmfPackSolver, Linux). Legacy solve-only: `FullGeneral`, `ProfileSPD` (System base).
- **Node** (`model/node/main.py`): Base class. Specializations named `Node{nDim}_{nDOF}` (e.g., `Node36` = 3D, 6-DOF). Maintains trial and committed states for displacement, velocity, acceleration, force.
- **Element** (`model/element/main.py`): Abstract base. Subclasses must implement `_domain()`, `_commit()`, `_revert()`, `_update()`. Holds nodes, section, local k (Matrix) and f (Vector).
- **Material** (`model/material/main.py`): Abstract base with `_setTrialStrain()`, `_commitState()`, `_revertToLastCommit()`. Uniaxial materials: `Elastic` (linear), `ElasticPerfectlyPlastic` (bilinear with zero post-yield hardening, tracks plastic strain).
- **Section** (`model/element/section/main.py`): Holds material, computes section-level tangent (EA) and stress. `Rectangular` computes `EA = E * h * w` and updates tangent from material on each `_setTrialStrain` call.
- **Recorders** (`output/recorder/`): `NodeRecorder`, `ElementRecorder`, `ModeShapeRecorder`. Support single or list of nodes/elements. Data in `rec.data[key]`; time in `rec.time`. Single item → `data[key] = [val_per_step, ...]`; multi → `data[key] = [{ID: val, ...}, ...]`. Optional `file=` for incremental output; `save(path)` for batch. ModeShapeRecorder auto-triggered by `Domain._record_eigen()`.
- **Eigen** (`analysis/eigen/main.py`): `Eigen(solver).solve(K, M, numModes)` — K·φ = λ·M·φ. Backends: `'fullGenLapack'` (dense eigh), `'genBandArpack'` (sparse eigsh, shift-invert). Returns sorted eigenvalues (ω²) and mass-normalized eigenvectors.

### Continuum Element Enrichment API Contract

Any continuum element implementing incompatible modes or other displacement enrichment must override the following six methods in the element class. The base class (`ContinuumElement`) provides default `None` returns so standard elements (Quad4, Tet4, etc.) require no changes.

```
get_H(xi, u_e)              — full enriched displacement gradient H = H_std + H_enrich
get_F(xi, u_e)              — enriched deformation gradient F = I + H_enriched
get_B_NL(xi, u_e)           — enriched nonlinear B matrix built from enriched F
get_H_enrichment(xi)        — enrichment delta ONLY: returns H_enrich = get_H() - super().get_H()
get_strain_enrichment(xi)   — Voigt strain enrichment (e.g. G @ alpha); for strain-level injection
get_G(xi)                   — incompatible mode strain matrix (6×nAlpha) for static condensation
```

**Consistency requirement**: `get_H_enrichment(xi)` must return exactly `get_H(xi, u_e) - super().get_H(xi, u_e)`. `get_strain_enrichment(xi)` must return `get_G(xi) @ alpha`. All six methods must be mutually consistent.

**Which kinematics consume which method:**

| Kinematics | Injection point | Reason |
|---|---|---|
| Linear | `get_strain_enrichment(xi)` in `update()` | strain-level enrichment (G@alpha) |
| TotalLagrangian | `get_H(xi, u_e)` | computes H_total directly from nodal positions |
| UpdatedLagrangian | `get_H_enrichment(xi)` | H_total built incrementally via `F_incr @ F_commit`; enrichment added as delta |
| CorotContinuum | `get_F(xi, u_e)` for R extraction; `get_strain_enrichment(xi)` for corotated strain | polar decomp needs enriched F; strain needs enrichment in corotated frame |

**Element accessor API**: Kinematics use `element.get_thickness()` and `element.get_gp_weight(gp)` — not private attributes `_thickness` or `_gp_data`.

**Wilson-Taylor J₀ rule (ADR-009 amendment)**: J₀ and G matrices for incompatible mode enrichment are computed once from the original undeformed element geometry and **never refreshed**, regardless of kinematics formulation. This is distinct from `dN_dX` in the kinematics layer which follows the kinematics reference (UL advances at commit per Bathe FEP §6.2). J₀ is a Taylor et al. center-point patch-test correction — an element geometric property, not a kinematics reference quantity. Refreshing J₀ in UL produces non-monotonic TL-UL convergence and is wrong.

**Reference implementations**: Hex8 — see `get_strain_enrichment()`, `get_H_enrichment()`, `get_H()`, `get_F()`, `get_B_NL()`, `get_G()`. Mirror this pattern for Hex20, Tet10, or any future enriched element.

### Data Structures (`_systools/data/`)

Custom `Vector`, `Matrix`, `Tensor` classes wrap NumPy arrays. Key features:
- `@property` accessors: `data`, `length`/`size`, `dtype`
- Indexing: `__getitem__`/`__setitem__` delegate to numpy
- Numpy interop: `__array__()` so `np.asarray(vec)` works
- Linear algebra: `Vector.dot()`, `Vector.outer()`, `Matrix.dot()`, `Matrix.solve()`
- Assembly: `Matrix.__iadd__` and element-wise `K[i,j] += ke[a,b]` scatter

Global K, F, u are Matrix/Vector objects. The Linear solver extracts numpy arrays internally via `np.asarray()`.

### State Management Pattern

All stateful objects (Node, Material, Element) use a **trial/committed** two-state pattern:
- `_update()` — set trial state (current iteration)
- `_commitState()` — accept trial as committed (converged step) — **must deep-copy**, not alias
- `_revertToLastCommit()` — roll back trial to last committed state — **must deep-copy**, not alias

### Multiscale Vision (FE²)

Recursive `SimulationManager` nesting: a micro-scale SM solving an RVE acts as a `Material` inside a macro element. The Material interface (`_setTrialStrain` → run inner solve → return homogenized stress/tangent; `_commitState`/`_revertToLastCommit` → propagate) enables this directly. Trial/committed pattern at every level (Node, Material, Element, Domain) makes each scale independently convergent.

## Code Conventions

- **Private attributes**: double underscore `self.__attr` (name mangling) for Domain internals; single underscore `self._attr` for protected member access in subclasses
- **Internal methods**: `_method()` (protected) or `__method()` (mangled)
- **Public API**: `getX()` / `setX()` style (Java-like getters/setters), plus `@property` for key accessors on Vector/Matrix/Domain
- **Node naming**: `Node{nDim}_{nDOF}` — e.g., `Node22` (2D, 2-DOF), `Node36` (3D, 6-DOF), `Node37` (3D, 7-DOF)
- **File headers**: Every source file has a copyright header block
- **Error messages**: Prefixed with class path, e.g., `"oneFEM.Node._setDOF() - ..."`
- **No type hints** in the current codebase
- **Relative imports** within the package
- Python >= 3.7, < 4.0

## Known Element Limitations

- **Wilson incompatible modes + curved geometry**: Wilson modes halve uz (shear locking) error but cannot cure uy (membrane locking) on curved elements. G matrix built from J₀ at center assumes flat geometry; curvature coupling defeats this. Arc-only refinement outperforms cross-section refinement. EAS (Simo & Rifai 1990) required for full membrane locking cure. See `hex8_curved_cantilever.py` header for complete data.
- **Corot (EICR) + large combined rotation**: Diverges at ~70-75% load on Bathe curved cantilever (combined bending + torsion + large rotation), independent of step count. Formulation limit, not Newton convergence issue.

## Common Pitfalls

- Node coordinate attribute is `_coord` (not `_coords`)
- Vector/Matrix use name-mangled `__data` internally — always access via `@property` `.data` from outside
- Element subclasses must NOT call `super()._commit()` / `super()._update()` — base raises NotImplementedError
- `isinstance(x, array)` doesn't work — use `isinstance(x, ndarray)` (numpy)
- `_is_parallel` checks cross product ≈ 0 (not dot product ≈ 0)
- Analysis `__add_analysis` must use independent `if` statements (not `elif` chain) to store all components
- **Pipeline order**: `Domain._commit()` (update nodes) must come BEFORE `element._update()/_commit()` so elements read current-step displacements
- **Node commit/revert must deep-copy**: Use `Vector(self._u_trial)` not `self._u_commit = self._u_trial` (aliasing breaks revert)
- **Mutable default args**: Use `None` defaults in constructors, not `[]` or `Material()` — mutable defaults are shared across instances
- **Node subclasses must use single-underscore `_attr`** (not `__attr` name-mangling) for attributes accessed by the base class interface (`getNDOF()`, `getDOFs()`, `_fix`, etc.)
- **nDMaterial and other Material subclasses** must call `super().__init__(mat_id)` to initialize the parent
- **Rectangular section `_setTrialStrain`** must update `self._C` from material tangent (not just stress) for nonlinear materials
- **Kinematics reference rule (ADR-009)**: Kinematics dN/dX follows formulation (UL refreshes at commit only — Bathe FEP §6.2; others never). Wilson-Taylor J₀/G always original config, never refreshed. See Enrichment API Contract section and `memory/kinematics_reference_rule.md`.

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
