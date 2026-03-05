# oneFEM — Python Architecture Reference

**Purpose:** Authoritative reference for Claude Code when implementing, extending,
or reviewing the oneFEM codebase. Read this file before writing any code.
It documents what exists, what the conventions are, where things go, and
what the strategic design targets are.

Cross-reference with `CLAUDE.md` for build/install and known pitfalls.

---

## 0. Research Vision — Read This First

**oneFEM is not just a FEM library. It is the software core of a multi-year
PI-led research programme in computational mechanics and earthquake engineering.**
Every line of code either proves a scientific claim, enables a paper, or
unlocks a grant. Before implementing anything, understand what it is *for*.

### The Central Scientific Thesis

> Earthquake-scale nonlinear structural simulations can be made simultaneously
> *scalable*, *portable*, and *scientifically reproducible* by combining three
> innovations: (1) a hardware-topology-aware parallel assembly kernel,
> (2) a neural material surrogate that replaces expensive constitutive evaluation,
> and (3) a unified-memory execution model that eliminates host↔device data
> movement on SoC hardware.

Every work package (WP) is a step toward proving one part of this thesis.

### The Hardware Architecture This Code Is Targeting

This is critical context that affects every design decision:

**The problem with conventional GPU-accelerated FEM:**
```
CPU assembles K  ──[PCIe copy, ~10ms]──►  GPU solves K·u=f  ──[PCIe copy]──►  CPU
```
At moderate problem size (50K–500K DOF), the copy overhead dominates the
solve time. The GPU is idle during assembly; the CPU is idle during solve.

**The unified memory SoC architecture this codebase targets:**
```
CPU assembles K  ──►  GPU solves K·u=f       (same physical DRAM, no copy)
         └── shared LPDDR5/HBM pool, coherent address space ──┘
```
On SoC hardware (NVIDIA GH200, AMD MI300A, Jetson AGX Orin), `cudaMallocManaged`
allocates memory that is *natively visible* to both CPU and GPU. There is no
PCIe bus. The copy cost is structurally zero. This is not an optimization — it
is a fundamentally different execution model.

**The architectural claim we are building toward (WP23-24):**
- A `UnifiedAllocator` that allocates into the unified address space
- A `SparseMatrixSOE` that assembles on CPU and solves on GPU *without any
  explicit data movement* — no `cudaMemcpy`, no `hipMemcpy`
- The Tier-1 acceptance gate for WP23-24 is binary: *does the code run
  correctly with zero explicit copies on GH200/MI300A?*

**Why the Python layer matters for this:**
The Python benchmarks (`benchmark_unified_memory.py`) provide hardware-portable
*evidence* of the copy-overhead problem on discrete GPUs and its elimination on
SoC, using identical Python/CuPy code on both platforms. This evidence feeds
directly into grant proposals (NSF, NVIDIA Academic Grant) and paper P1.

### Python IS the Project — Not a Mockup

**This is the most important framing to understand before writing any code.**

oneFEM is a pure Python research platform. It is not a prototype for a future
C++ port. Performance comes entirely from hardware-native Python backends:

```
oneFEM Python stack
│
├── Assembly (CPU, parallel):
│     concurrent.futures.ThreadPoolExecutor  — shared memory element loop
│     mpi4py                                 — distributed domain decomposition
│
├── Linear solve (GPU):
│     cupyx.scipy.sparse.linalg.spsolve      — cuSPARSE under the hood
│     cupyx.scipy.sparse.linalg.gmres        — iterative for large systems
│
├── Unified memory (SoC, WP23-24):
│     cupy.cuda.UnifiedMemory                — cudaMallocManaged in Python
│     cupy.asarray() on SoC = pointer remap, not copy
│
├── Neural surrogate (WP19):
│     PyTorch / JAX material model           — drops into Material interface
│     torch.jit.script for consistent tangent via autograd
│
└── FE² multiscale:
      nested SimulationManager               — pure Python, works today
```

**C++ appears in this project in exactly one role: OpenSees is the reference
implementation whose interfaces we mirror.** We read its source to understand
the correct method signatures and assembly patterns (Section 17). We do not
plan to rewrite oneFEM in C++. If a future contributor wants a C++ port, the
Python interfaces define the contracts they must implement.

The scientific contribution is the *Python abstraction* — proving that a
unified-memory, hardware-portable, research-grade FEM platform can be built
entirely in Python with production-scale performance via CuPy, mpi4py, and
PyTorch.

### The Three Research Circles

```
Circle A — Foundation (existing + near-term):
  Working Python FEM → validated static + dynamic + eigen  ✅ done
  Steel01, Concrete01, Hardening materials
  ElasticBeamColumn with CrdTransf
  HDF5 checkpoint/restart (WP8-9)

Circle B — Novel Capabilities (medium-term):
  CuPy sparse solver (WP13)          — cupyx replaces scipy FullGeneral
  Parallel element assembly (WP10-11) — mpi4py + concurrent.futures
  UnifiedAllocator (WP23-24)          — zero-copy on SoC ← hardware thesis
  Neural material surrogate (WP19)    — PyTorch/JAX Oracle

Circle C — Frontier (long-term):
  FE² multiscale → MicroSimulationManager IS a Material
  Oracle-guided adaptive load stepping
  Provably reproducible parallel simulation on GH200/MI300A
```

### Work Package → Scientific Output Mapping

Every WP you are asked to implement maps to at least one paper or grant.
Do not treat WPs as arbitrary code organization — they are research deliverables.

| WP | Scientific claim | Paper / Grant |
|---|---|---|
| WP3-6 | Correct nonlinear FEM baseline in Python | P0: software paper |
| WP8-9 | Bitwise-reproducible checkpoint/restart | P1: reproducibility |
| WP10-11 | Parallel assembly scales with core count | P1, NSF OAC |
| WP13 | cuSPARSE solver outperforms scipy at 100K+ DOF | P1, NVIDIA grant |
| WP19 | Neural surrogate ×10 speedup vs. constitutive eval | P2: surrogate paper |
| WP23-24 | Zero-copy on GH200/MI300A vs. discrete GPU | P1, NVIDIA grant |
| FE² | Multiscale convergence via nested SimulationManager | P3: FE² paper |

### What This Means When You Write Code

1. **Every acceptance criterion is binary.** "The test passes" or "it does not."
   No partial credit. No "mostly works." This is how research claims are verified.

2. **The Material interface is the most important abstraction in the codebase.**
   It is the injection point for the neural surrogate (WP19) *and* the FE²
   micro-model. If you break its trial/commit contract, you break both.

3. **The `UnifiedAllocator` is not a performance optimization — it is a
   scientific hypothesis.** Implement it to be provably correct first,
   measurably fast second.

4. **When you reach for a performance tool, reach for a Python backend first:**
   - Need GPU solve? → `cupyx.scipy.sparse.linalg`
   - Need parallel assembly? → `concurrent.futures.ThreadPoolExecutor`
   - Need distributed? → `mpi4py`
   - Need neural material? → `torch.nn.Module` with autograd tangent
   - Never suggest a C++ extension unless a Python backend genuinely cannot
     meet the requirement.

5. **OpenSees is a reference, not a target.** Read its source to understand
   correct interfaces (Section 17). Do not propose porting oneFEM to C++.

6. **The benchmark script is a grant deliverable.** `benchmark_unified_memory.py`
   produces the preliminary data figure for the NVIDIA Academic Grant and NSF
   OAC proposal. Keep it runnable and reproducible.

---

## 1. Non-Negotiable Coding Rules



Violating these breaks existing tests or the commit/revert contract.

1. **Trial/committed two-state pattern on every stateful object.**
   `_update()` sets trial state only. `_commitState()` makes it permanent.
   `_revertToLastCommit()` discards trial. These three must deep-copy state
   — never alias (`self._u_commit = Vector(self._u_trial)`, not `= self._u_trial`).

2. **Return codes, not exceptions, for numerical failures.**
   `_commitState()`, `_revertToLastCommit()`, `_update()` return `int`
   (0 = OK). Exceptions are for programming errors only (wrong type, None).

3. **No type hints** — current codebase has none; do not add them without
   a separate decision to migrate the whole codebase.

4. **Private naming convention:**
   - `self.__attr` (double underscore, name-mangled): Domain internals only
   - `self._attr` (single underscore): protected members in all other classes
   - Node subclasses **must** use `_attr` (not `__attr`) for anything the
     base class interface touches (`getNDOF()`, `getDOFs()`, `_fix`, etc.)

5. **Public API style: `getX()` / `setX()`** (Java-like getters) plus
   `@property` for key accessors. No bare attribute access across module
   boundaries.

6. **Relative imports** everywhere inside the package.

7. **Pipeline order is fixed:**
   `Domain._commit()` (update nodes) → `element._update()/_commit()`
   Elements read current-step displacements from nodes, so nodes must be
   committed first.

8. **`super().__init__(mat_id)` required** in all Material subclasses.
   Omitting it silently breaks the material registry.

9. **Mutable default arguments:** always `None`, never `[]` or `Material()`.

10. **Element subclasses must NOT call `super()._commit()` or
    `super()._update()`** — the base raises `NotImplementedError`.

---

## 2. Repository Layout

```
src/oneFEM/
│
├── _systools/                   # Internal infrastructure
│   ├── data/
│   │   ├── vector.py            # Vector — wraps np.ndarray, 1D
│   │   ├── matrix.py            # Matrix — wraps np.ndarray, 2D
│   │   └── tensor.py            # Tensor — wraps np.ndarray, adaptable (3D or 4D tensors in compressed tensor notation)
│   ├── math_tools.py            # RTOL=1e-6, ATOL=1e-12 comparison helpers
│   ├── simulation_manager.py    # SimulationManager — top-level orchestrator
│   ├── source.py                # .txt model file parser
│   └── cli.py                   # CLI: `oneFEM` command
│
├── model/                       # FEM domain objects
│   ├── main.py                  # Domain — central model container ★
│   ├── node/
│   │   └── main.py              # Node base + Node{nDim}_{nDOF} specializations
│   ├── element/
│   │   ├── main.py              # Element abstract base
│   │   ├── section/main.py      # Section (holds material, computes EA/EI)
│   │   ├── truss/               # Truss element
│   │   ├── beam/                # Beam-column element
│   │   ├── shell/               # Shell element
│   │   ├── solid/               # Solid element
│   │   └── zerolength/          # ZeroLength element
│   ├── material/
│   │   ├── main.py              # Material abstract base
│   │   ├── uniaxial/            # Elastic, ElasticPerfectlyPlastic, ...
│   │   └── nD/                  # nD materials
│   ├── pattern/                 # Load patterns (Plain + time series)
│   ├── tseries/                 # TimeSeries (Constant, Linear, Path)
│   └── constraint/              # Boundary constraints
│
├── analysis/                    # Solver pipeline
│   ├── main.py                  # Analysis — orchestrates Strategy components ★
│   ├── algorithm/               # Linear, Newton, Krylov-Newton
│   ├── integrator/              # LoadControl, DisplacementControl, Newmark, CentralDiff
│   ├── system/                  # FullGeneral, UMFPACK
│   ├── numberer/                # Plain, ReverseCuthillMcKee
│   ├── constraints/             # Plain, Penalty constraint handlers
│   ├── eigen/                   # Eigen (fullGenLapack, genBandArpack)
│   └── test/                    # Convergence tests
│
├── input/                       # Mesh import (CUBIT, GID)
└── output/
    └── recorder/
        ├── node_recorder.py     # NodeRecorder
        ├── element_recorder.py  # ElementRecorder
        └── mode_shape_recorder.py # ModeShapeRecorder
```

★ = most important files; start here when understanding the system.

---

## 3. The Three-Object Pipeline

Everything flows through three top-level objects:

```
SimulationManager
      │
      ├── Domain          ← owns all model objects (nodes, elements, materials,
      │                      patterns, constraints, recorders, global K/F/u)
      │
      ├── Analysis        ← owns strategy components (Algorithm, Integrator,
      │                      System, Numberer, ConstraintHandler, Test)
      │
      └── Recorders       ← observe committed state, write to file or memory
```

`SimulationManager.analyze(nSteps, dt)` is the entry point for a run.
It delegates to `Analysis._analyze(model, nSteps, dt)`.

The eigen pipeline is **standalone** — it does not use Analysis:
```python
model.eigen(numModes, solver)      # triggers _domain(), _assemble(), solves
model.modalProperties()            # computes ω, f, T, participation, eff. mass
```

---

## 4. Domain (`model/main.py`)

The central container. **Owns** all domain objects. Nothing else owns them.

### Key Internals

```python
class Domain:
    # Storage (name-mangled — access only via public interface)
    __node_map    # dict[int, Node]   — O(1) lookup by tag
    __elements    # list[Element]
    __patterns    # list[Pattern]
    __constraints # list[Constraint]
    __recorders   # list[Recorder]

    # Global system (Matrix/Vector objects, not raw numpy)
    K     # global stiffness   — Matrix
    F     # global force        — Vector
    u     # global displacement — Vector
    nDOF  # int, set by _domain()
```

### Key Methods

| Method | When called | What it does |
|---|---|---|
| `add(obj)` | Model build | Registers node/element/material/pattern/recorder |
| `remove(obj)` | Model build | Deregisters |
| `_domain()` | Before first solve | Sequential DOF numbering; element `_domain()` init |
| `applyLoad(pseudo_time)` | Start of each step | Scales all patterns by `TimeSeries.getFactor(t)`; applies nodal loads |
| `setLoadConstant()` | Mid-analysis | Locks current load level; `getFactor()` returns 1.0 from here on |
| `_assemble()` | Each step/iteration | Assembles global K and F from all elements |
| `update()` | During Newton | Calls `element._update()` for all elements (after trial disp set on nodes) |
| `_commit(F, u)` | After convergence | Updates node committed state; triggers element `_update()`/`_commit()` |
| `revertToLastCommit()` | On Newton failure | Propagates to all nodes, elements, materials |
| `_record(time)` | After `_commit` | Calls all recorders |
| `getInternalForce()` | Newton loop | Returns assembled F_int from element resisting forces |
| `getCommittedDisp()` | Newton loop | Returns committed displacement vector |
| `eigen(numModes, solver)` | Standalone | Full eigen solve: `_domain()`, `_assemble()`, partition, solve, `_record_eigen()` |
| `modalProperties()` | After `eigen()` | Computes and prints ω, f, T, participation, eff. mass; returns dict |
| `getEigenvalue(mode)` | Post-eigen | 1-based accessor |
| `getEigenvector(mode)` | Post-eigen | 1-based accessor |
| `setRayleighDampingFactors(aM,bK,bK0,bKc)` | Dynamic setup | Sets αM, βK for C = αM·M + βK·K_current + βK0·K_initial |

### DOF Partition (inside Analysis._organize)

```
Free DOFs  → indices uu   (what the solver sees)
Fixed DOFs → indices pp   (prescribed, typically zero)
```

This partition happens once at analysis setup and is passed to the
Algorithm at each step.

### Load Patterns and TimeSeries

```python
class LoadPattern:
    """
    Container for loads scaled by a TimeSeries.
    OpenSees: LoadPattern + TimeSeries.
    """
    _series         # TimeSeries — f(pseudoTime) scaling function
    _nodal_loads    # dict[node_tag, Vector] — static reference loads
    _elem_loads     # list[ElementalLoad] — distributed loads on elements
    _sp_constraints # list[SP_Constraint] — time-varying prescribed DOFs

    def applyLoad(self, pseudo_time: float):
        factor = self._series.getFactor(pseudo_time)
        # Apply each nodal load * factor to node unbalanced load
        # Apply each elemental load * factor via element.addLoad()
        # Apply each SP constraint * factor

class TimeSeries:
    """OpenSees: TimeSeries — f(t) load scaling."""
    def getFactor(self, pseudo_time: float) -> float: ...

# Concrete implementations:
# ConstantSeries  — getFactor() always returns 1.0
# LinearSeries    — getFactor(t) = t
# PathTimeSeries  — getFactor(t) interpolated from data array
# TrigSeries      — getFactor(t) = sin(2π/T * (t - t_start) + shift)
```

### Constraints

**SP_Constraint** (single-point — prescribed DOF value):
```python
class SP_Constraint:
    _node_tag    # int — which node
    _dof_number  # int — which DOF (0-based)
    _value_r     # float — reference value
    _is_constant # bool — if True, value doesn't scale with load factor

    def getValue(self) -> float:
        return self._value_r  # or self._value_r * load_factor if not constant

    def applyConstraint(self, load_factor: float):
        """Apply prescribed value to node DOF."""
```

**MP_Constraint** (multi-point — ties DOFs between nodes, future):
```python
class MP_Constraint:
    """
    {u_constrained} = [C] * {u_retained}
    Used for rigid diaphragms, rigid links, equal-DOF constraints.
    Requires LagrangeConstraintHandler or TransformationConstraintHandler.
    OpenSees: MP_Constraint with constraint matrix C.
    """
    _node_retained    # int
    _node_constrained # int
    _constraint       # Matrix [C]
    _constr_dofs      # list[int] — constrained DOF indices
    _retain_dofs      # list[int] — retained DOF indices
```

---

## 5. Node (`model/node/main.py`)

### Base Class

```python
class Node:
    # Coordinates — single underscore (NOT double)
    _coord   # Vector — note: _coord not _coords (common mistake)

    # Trial state
    _u_trial   # Vector — displacement
    _v_trial   # Vector — velocity
    _a_trial   # Vector — acceleration
    _f_trial   # Vector — force (reactions)

    # Committed state — MUST be deep copies, never aliases
    _u_commit  # Vector(self._u_trial) — correct
    _v_commit  # Vector(self._v_trial) — correct
    _a_commit  # Vector(self._a_trial) — correct
    _f_commit  # Vector(self._f_trial) — correct

    # DOF management
    _fix       # list[bool] — which DOFs are constrained
    _dofs      # list[int]  — global equation indices, set by _domain()

    # Dynamic analysis additions
    _mass      # Matrix — nodal mass (nDOF × nDOF); None = no mass
    _alphaM    # float  — Rayleigh mass damping factor
    _unbal     # Vector — unbalanced (external) force at this node
```

### Node Public Interface (complete)

```python
# Coordinates
node.getCrds()                    # returns _coord Vector

# Trial state setters (called by integrator during Newton)
node.setTrialDisp(v: Vector)
node.setTrialVel(v: Vector)
node.setTrialAccel(v: Vector)
node.incrTrialDisp(dv: Vector)    # incremental update — preferred in Newton loop

# Trial state getters
node.getTrialDisp()
node.getTrialVel()
node.getTrialAccel()

# Committed state getters
node.getDisp()                    # alias: getCommittedDisp()
node.getVel()
node.getAccel()

# Loads and mass
node.getUnbalancedLoad()          # external force vector at this node
node.getUnbalancedLoadIncInertia()  # external force + M*a (for transient)
node.setMass(m: Matrix)
node.getMass()

# State management (all three required)
node.commitState()
node.revertToLastCommit()
node.revertToStart()              # reset to zero displacement (initial state)

# Eigenvectors (set after eigen solve)
node.setEigenvector(mode: int, phi: Vector)
node.getEigenvector(mode: int)
```

### Node Specializations

Named `Node{nDim}_{nDOF}`, e.g.:
- `Node22` — 2D, 2 DOF (ux, uy)
- `Node23` — 2D, 3 DOF (ux, uy, rz)
- `Node36` — 3D, 6 DOF (ux, uy, uz, rx, ry, rz)
- `Node37` — 3D, 7 DOF (ux, uy, uz, rx, ry, rz, warping)

**Adding a new specialization:** Create `Node{nDim}_{nDOF}` in `model/node/`,
inherit from `Node`, set `nDim` and `nDOF`, initialize `_coord` as a
`Vector` of length `nDim`, initialize all state vectors as `Vector(np.zeros(nDOF))`.
Use `_attr` (not `__attr`) for everything the base class touches.

### Commit/Revert Pattern (CRITICAL — see common pitfalls)

```python
def _commitState(self):
    self._u_commit = Vector(self._u_trial)   # deep copy via Vector constructor
    self._v_commit = Vector(self._v_trial)
    self._a_commit = Vector(self._a_trial)
    return 0

def _revertToLastCommit(self):
    self._u_trial = Vector(self._u_commit)   # deep copy, not alias
    self._v_trial = Vector(self._v_commit)
    self._a_trial = Vector(self._a_commit)
    return 0
```

---

## 6. Element (`model/element/main.py`)

### Abstract Interface

Every concrete element must implement these methods (mirroring OpenSees C++):

```python
class MyElement(Element):

    # ── Setup ─────────────────────────────────────────────────────────────────

    def _domain(self):
        """
        Called once before analysis. Resolve node references, compute local
        geometry (length, direction cosines), initialize local k (Matrix)
        and f (Vector). OpenSees equivalent: setDomain(Domain*).
        """

    # ── State management (all three required) ─────────────────────────────────

    def _update(self):
        """
        Called during Newton iteration after trial displacements are updated.
        Read current nodal trial displacements, compute strain, push to
        section/material via _setTrialStrain(). Updates material trial state.
        OpenSees equivalent: update().
        NOTE: Called BEFORE _commit(), during iteration.
        """

    def _commit(self):
        """
        Called after Newton convergence. Commit section/material state.
        Store committed local force vector for next step's F_int.
        DO NOT call super()._commit() — base raises NotImplementedError.
        OpenSees equivalent: commitState().
        """

    def _revert(self):
        """
        Called when Newton fails to converge. Revert section/material to
        last committed state. OpenSees equivalent: revertToLastCommit().
        """

    def revertToStart(self):
        """
        Reset element to initial (virgin) state. Called during model reset
        or restart from step 0. OpenSees equivalent: revertToStart().
        """

    # ── Stiffness and force ────────────────────────────────────────────────────

    def getTangentStiff(self) -> Matrix:
        """
        Current tangent stiffness K_T (nDOF_e × nDOF_e).
        Uses material's getTangent() — consistent with current trial state.
        OpenSees equivalent: getTangentStiff().
        """

    def getInitialStiff(self) -> Matrix:
        """
        Initial (elastic) stiffness K_0. Uses material's getInitialTangent().
        Used by Modified Newton and initial stiffness Rayleigh damping.
        OpenSees equivalent: getInitialStiff().
        """

    def getResistingForce(self) -> Vector:
        """
        Internal resisting force vector (nDOF_e,) in global coords.
        F_int = integral of stress over element. Used to form residual.
        OpenSees equivalent: getResistingForce().
        """

    # ── Mass and damping (required for transient analysis) ────────────────────

    def getMass(self) -> Matrix:
        """
        Element mass matrix (nDOF_e × nDOF_e).
        Lumped or consistent — element-type dependent.
        For truss: lumped = rho*L/2 on each diagonal translational DOF.
        OpenSees equivalent: getMass().
        Return zeros matrix if element has no mass.
        """

    def getDamp(self) -> Matrix:
        """
        Element damping matrix. Default implementation: Rayleigh damping
        C = alphaM * M + betaK * K_current + betaK0 * K_initial.
        OpenSees equivalent: getDamp().
        """

    def getResistingForceIncInertia(self) -> Vector:
        """
        F_int + M*a + C*v — full dynamic resisting force.
        Required for transient analysis residual.
        OpenSees equivalent: getResistingForceIncInertia().
        """

    # ── Elemental loads ────────────────────────────────────────────────────────

    def zeroLoad(self):
        """
        Zero any applied elemental load (gravity, distributed loads).
        Called at start of each load step before applyLoad().
        OpenSees equivalent: zeroLoad().
        """

    def addLoad(self, load, load_factor: float) -> int:
        """
        Add elemental load contribution scaled by load_factor.
        Relevant for beam elements with distributed loads.
        OpenSees equivalent: addLoad(ElementalLoad*, double).
        """

    def addInertiaLoadToUnbalance(self, accel: Vector) -> int:
        """
        Add -M * accel_committed to nodal unbalanced loads.
        Called during dynamic analysis for non-zero initial conditions.
        OpenSees equivalent: addInertiaLoadToUnbalance(const Vector&).
        """
```

### OpenSees Name Mapping for Elements

| OpenSees C++ | oneFEM Python | Notes |
|---|---|---|
| `setDomain(Domain*)` | `_domain()` | Setup phase |
| `commitState()` | `_commit()` | After convergence |
| `revertToLastCommit()` | `_revert()` | On Newton failure |
| `revertToStart()` | `revertToStart()` | Full reset |
| `update()` | `_update()` | During Newton iteration |
| `getTangentStiff()` | `getTangentStiff()` | Current tangent K |
| `getInitialStiff()` | `getInitialStiff()` | Initial K_0 |
| `getResistingForce()` | `getResistingForce()` | F_int |
| `getResistingForceIncInertia()` | `getResistingForceIncInertia()` | F_int + Ma + Cv |
| `getMass()` | `getMass()` | Mass matrix |
| `getDamp()` | `getDamp()` | Damping matrix |
| `zeroLoad()` | `zeroLoad()` | Clear elemental loads |

### Section (`model/element/section/main.py`)

Sits between Element and Material. Two distinct levels:

**1. `Rectangular` (current — simple cross-section)**
Computes section-level EA (axial stiffness) and stress resultant for truss/simple beam.

```python
class Rectangular:
    def _setTrialStrain(self, strain):
        self._material._setTrialStrain(strain)
        # CRITICAL: update self._C from material TANGENT, not stress
        self._C = self._material.getTangent() * self._h * self._w
        self._stress = self._material.getStress() * self._h * self._w
```

**2. `SectionForceDeformation` (OpenSees pattern — for beam-column elements)**
Section-level force-deformation interface. Operates on section resultants
(N, M_z, M_y, V_y, V_z, T) rather than scalar stress/strain.

```python
class SectionForceDeformation:
    """
    OpenSees: SectionForceDeformation.
    Used by forceBeamColumn, dispBeamColumn, and fiber sections.
    """

    def setTrialSectionDeformation(self, deforms: Vector) -> int:
        """
        deforms: section deformation vector — content depends on section type.
        For 2D beam: [epsilon_axial, kappa_z]  (axial strain + curvature)
        For 3D beam: [epsilon, kappa_z, kappa_y, gamma_y, gamma_z, phi]
        """

    def getSectionDeformation(self) -> Vector:
        """Return current trial deformation vector."""

    def getStressResultant(self) -> Vector:
        """
        Return section force resultant vector.
        For 2D beam: [N, M_z]
        For 3D beam: [N, M_z, M_y, V_y, V_z, T]
        """

    def getSectionTangent(self) -> Matrix:
        """Return consistent section tangent stiffness matrix (dF/de)."""

    def getInitialTangent(self) -> Matrix:
        """Return initial section tangent."""

    def getType(self) -> list:
        """
        Return list of response type flags, e.g.:
        [SECTION_RESPONSE_P, SECTION_RESPONSE_MZ]
        Defines which resultants this section computes.
        """

    def getOrder(self) -> int:
        """Number of response quantities (length of resultant vector)."""

    def getCopy(self) -> "SectionForceDeformation":
        """Deep clone. Required when section is shared across integration points."""

    def commitState(self) -> int: ...
    def revertToLastCommit(self) -> int: ...
    def revertToStart(self) -> int: ...
```

**FiberSection** (future WP): Discretizes cross-section into fibers, each with a
`UniaxialMaterial`. Section stress resultants computed by integrating fiber
stresses × area over the section. Enables inelastic beam-column behavior.

### Coordinate Transformation (`coordTransformation/`) — Required for Beams

**Every beam or frame element requires a `CrdTransf` object.** This is not
optional — without it, local↔global conversion is hardcoded and geometric
nonlinearity is impossible.

```python
class CrdTransf:
    """
    Maps element-local frame to global frame.
    OpenSees: CrdTransf (Linear2d/3d, Corotational2d/3d, PDelta2d/3d).
    """

    def initialize(self, node1: Node, node2: Node) -> int:
        """Compute initial local axes from nodal coordinates."""

    def update(self) -> int:
        """Recompute axes after deformation (needed for Corotational)."""

    def getInitialLength(self) -> float: ...
    def getDeformedLength(self) -> float: ...

    def getBasicTrialDisp(self) -> Vector:
        """Return displacements in basic (local, rigid-body-removed) frame."""

    def getGlobalResistingForce(self, basic_force: Vector, p0: Vector) -> Vector:
        """Transform local resisting force to global frame."""

    def getGlobalStiffMatrix(self, basic_stiff: Matrix, basic_force: Vector) -> Matrix:
        """Transform local tangent stiffness to global frame.
        Includes geometric stiffness for Corotational/PDelta."""

# Implementations (in order of complexity):
# LinearCrdTransf2d/3d   — small deformation, fixed local axes
# PDeltaCrdTransf2d/3d   — P-delta geometric nonlinearity
# CorotCrdTransf2d/3d    — large rotation, corotational formulation
```

**When adding any beam element:** pass `CrdTransf` as a constructor argument.
The element calls `transf.getBasicTrialDisp()` in `_update()` and
`transf.getGlobalStiffMatrix()` / `transf.getGlobalResistingForce()` in
`getTangentStiff()` / `getResistingForce()`.

---

## 7. Material (`model/material/main.py`)

### Abstract Interface

```python
class Material:
    def __init__(self, mat_id):
        super().__init__(mat_id)   # REQUIRED in all subclasses

    def _setTrialStrain(self, strain, strain_rate=0.0):
        """Update trial state. Return 0 (OK)."""

    def _commitState(self):
        """Make trial state permanent. Return 0 (OK)."""

    def _revertToLastCommit(self):
        """Restore committed state. Return 0 (OK)."""

    def revertToStart(self):
        """
        Reset to initial (virgin) state — zero strain, zero stress.
        Required for restart from step 0 and for material reuse in FiberSection.
        OpenSees equivalent: revertToStart().
        """

    def getStrain(self):
        """
        Return current TRIAL strain.
        OpenSees equivalent: getStrain().
        Needed by recorders and FiberSection integration loops.
        """

    def getStress(self):
        """Return stress at current TRIAL strain."""

    def getTangent(self):
        """Return consistent tangent at current TRIAL strain."""

    def getInitialTangent(self):
        """Return initial (elastic) tangent. Constant."""

    def getCopy(self):
        """
        Return a deep clone of this material with the same parameters
        but independent state. Required for:
        - FiberSection (one copy per fiber)
        - Parallel/Series material composition
        - Any element that needs multiple independent material instances
        OpenSees equivalent: getCopy().
        """
```

### OpenSees Name Mapping for Materials

| OpenSees C++ | oneFEM Python | Notes |
|---|---|---|
| `setTrialStrain(e, de)` | `_setTrialStrain(strain, strain_rate)` | Update trial |
| `commitState()` | `_commitState()` | Accept trial |
| `revertToLastCommit()` | `_revertToLastCommit()` | Discard trial |
| `revertToStart()` | `revertToStart()` | Full reset |
| `getStrain()` | `getStrain()` | Trial strain |
| `getStress()` | `getStress()` | Trial stress |
| `getTangent()` | `getTangent()` | Consistent tangent |
| `getInitialTangent()` | `getInitialTangent()` | Elastic stiffness |
| `getCopy()` | `getCopy()` | Deep clone |

### Existing Uniaxial Materials

| Class | Description |
|---|---|
| `Elastic` | Linear elastic, σ = E·ε |
| `ElasticPerfectlyPlastic` | Bilinear, zero post-yield hardening; tracks `_eps_p` |

### Material Composition Pattern (OpenSees: Parallel / Series)

OpenSees supports combining materials algebraically without subclassing.
These are planned additions following the same `getCopy()` + trial/commit contract:

```python
class ParallelMaterial(Material):
    """
    Materials in parallel: σ_total = Σ σ_i(ε)
    K_total = Σ K_i
    All receive the same strain; stresses and tangents are additive.
    """
    def __init__(self, tag, materials: list):
        self._mats = [m.getCopy() for m in materials]   # independent copies

class SeriesMaterial(Material):
    """
    Materials in series: ε_total = Σ ε_i
    1/K_total = Σ 1/K_i
    All carry the same stress; strains are additive.
    """
```

This pattern is why `getCopy()` is non-negotiable: `ParallelMaterial` must
own independent copies of each sub-material so their states don't alias.

### Adding a New Uniaxial Material

1. Create `model/material/uniaxial/your_material.py`
2. `super().__init__(mat_id)` as first line of `__init__`
3. Trial state: `_eps_trial`, `_sig_trial`, `_tangent_trial`
4. Committed state: `_eps_commit`, `_sig_commit`, `_tangent_commit`
5. `_commitState()` deep-copies all state variables
6. `_revertToLastCommit()` deep-copies committed back to trial
7. `getStress()` returns trial stress
8. `getTangent()` returns trial tangent (consistent, not initial)
9. Error messages prefixed: `"oneFEM.ClassName.method() - description"`
10. Validated via a benchmark script in `src/`

### Strategic Target: Neural Material Surrogate (WP19)

The `Material` interface is the injection point for the Oracle.
A `NeuralMaterialSurrogate` that implements the same interface
(`_setTrialStrain`, `_commitState`, `_revertToLastCommit`, `getStress`,
`getTangent`) drops in anywhere a conventional material is used — no
changes to Section, Element, or Domain required.

The surrogate manages its own internal state (hidden state for RNNs, or
history window for feedforward) using the same trial/committed two-phase
protocol. The `_commitState` / `_revertToLastCommit` contract is what
makes this safe and Newton-compatible.

---

## 8. Analysis Pipeline (`analysis/main.py`)

### Strategy Components

`Analysis` holds pluggable components, each independently swappable:

| Component | Interface | Current implementations |
|---|---|---|
| `Algorithm` | `solve(model, uu, pp)` | `Linear`, `Newton`, `KrylovNewton` |
| `Integrator` | `newStep()`, `update()`, `commit()` | `LoadControl`, `DisplacementControl`, `Newmark`, `CentralDifference` |
| `System` | `solve(K, F)` | `FullGeneral` (scipy dense), `UMFPACK` (sparse) |
| `Numberer` | `number(domain)` | `Plain`, `ReverseCuthillMcKee` |
| `ConstraintHandler` | `handle(domain)` | `Plain`, `Penalty` |
| `Test` | `test()` → int | convergence tests |

**Important:** `Analysis.__add_analysis` uses independent `if` statements
(not `elif` chain) to register all components. Do not refactor to `elif`.

### formTangent Flags

```python
CURRENT_TANGENT = 0   # Re-assemble K from current trial state (full Newton)
INITIAL_TANGENT = 1   # Use K_0 from start of analysis (initial stiffness Newton)
HALL_TANGENT    = 2   # Hall's method (rarely used)

# In Algorithm.solve():
self._integrator.formTangent(CURRENT_TANGENT)   # full Newton — re-forms every iter
self._integrator.formTangent(INITIAL_TANGENT)   # modified Newton — forms once per step
```

### Convergence Test Return Codes (OpenSees standard)

```python
# test() must return:
CONVERGED  = 0    # iteration converged — exit Newton loop
CONTINUE   = -1   # not yet converged — iterate again
FAILED     = -2   # exceeded max iterations — abort step

# In Newton algorithm:
result = self._test.test()
if result == CONVERGED: break
if result == FAILED:    return FAIL
# result == CONTINUE:   keep iterating
```

### Static Analysis Sequence (per step)

```
1. domain.applyLoad(pseudo_time)         — scale all patterns by TimeSeries
2. integrator.newStep()                  — increment load factor / time
3. algorithm.solve(model, uu, pp):
   a. integrator.formTangent(flag)       — SOE.zeroA(); scatter K_e via addA()
   b. integrator.formUnbalance()         — SOE.zeroB(); b = F_ext - F_int
   c. soe.solve()                        — K_uu * Δu = b_u
   d. model.incrTrialDisp(Δu)           — update trial displacements at nodes
   e. domain.update()                    — element._update() reads new trial disp
   f. convergence_test.test()            — 0=done, -1=continue, -2=failed
4. domain._commit(F, u)                  — nodes commit (BEFORE element commit)
5. element._commit()                     — elements commit materials
6. domain._record(time)                  — recorders observe committed state
```

### Newton-Raphson Details

```python
# Newton starts from committed displacements (not zero)
# Residual:  b = F_ext - F_int
# F_int:     domain.getInternalForce()  — assembles from element.getResistingForce()
# For dynamic: F_int = element.getResistingForceIncInertia() includes M*a + C*v
# Tangent:   K re-assembled if CURRENT_TANGENT; reused if INITIAL_TANGENT
# Update:    Δu applied to free DOFs only (uu indices)
```

### Newmark Dynamic Integration Formulas

```python
# Parameters: beta (β), gamma (γ)
# Average acceleration:  β=0.25, γ=0.5  (unconditionally stable)
# Linear acceleration:   β=1/6,  γ=0.5  (conditionally stable)

# Effective stiffness (formed once per step in newStep()):
K_eff = K + (gamma/(beta*dt)) * C + (1/(beta*dt**2)) * M

# Effective force (from committed state u_n, v_n, a_n):
F_eff = F_ext - F_int
      + M * (1/(beta*dt**2)*u_n + 1/(beta*dt)*v_n + (1/(2*beta)-1)*a_n)
      + C * (gamma/(beta*dt)*u_n + (gamma/beta-1)*v_n + dt*(gamma/(2*beta)-1)*a_n)

# After solving K_eff * Δu = F_eff:
u_new = u_n + Δu
v_new = gamma/(beta*dt) * (u_new - u_n) + (1 - gamma/beta)*v_n + dt*(1 - gamma/(2*beta))*a_n
a_new = 1/(beta*dt**2) * (u_new - u_n) - 1/(beta*dt)*v_n - (1/(2*beta) - 1)*a_n
```

### Eigen Pipeline (standalone)

```python
model.eigen(numModes=3, solver='genBandArpack')
# Internally:
#   1. domain._domain()          — DOF numbering
#   2. domain._assemble()        — build K and M
#   3. partition K, M to free DOFs
#   4. Eigen(solver).solve(K_ff, M_ff, numModes)
#      → sorted eigenvalues (ω²) and mass-normalized eigenvectors
#   5. domain stores results
#   6. domain._record_eigen()    → triggers ModeShapeRecorder

model.modalProperties()
# Returns dict: {omega, freq, period, participation, eff_mass}

phi = model.getEigenvector(mode=1)   # 1-based
lam = model.getEigenvalue(mode=1)    # 1-based
```

---

## 9. Data Structures (`_systools/data/`)

Custom wrappers over numpy. **Always use these for structural data** —
never raw numpy arrays at the model level.

### Vector

```python
v = Vector(np.zeros(3))      # from array
v = Vector([1.0, 2.0, 3.0])  # from list

v.data         # @property — returns underlying np.ndarray
v.length       # @property — number of elements
v[i]           # __getitem__ delegates to numpy
v[i] = x       # __setitem__ delegates to numpy
np.asarray(v)  # works via __array__()
v.dot(other)   # dot product
v.outer(other) # outer product

# Deep copy pattern:
v_copy = Vector(v)   # correct — constructs new Vector from v's data
v_copy = v           # WRONG — alias, breaks revert
```

### Matrix

```python
m = Matrix(np.zeros((3,3)))

m.data         # @property
m.size         # @property — (rows, cols)
m[i,j]         # __getitem__
m[i,j] = x    # __setitem__
m.dot(v)       # matrix-vector product
m.solve(v)     # solve m*x = v
m += ke_scatter  # __iadd__ with element stiffness scatter

# CRITICAL: access internals via .data @property
# DO NOT access __data directly — it is name-mangled
```

**Key gotcha:** `Vector.__data` and `Matrix.__data` use name mangling.
Always access via `.data` property. `np.asarray(vec)` works via `__array__()`.

---

## 10. Recorders (`output/recorder/`)

### NodeRecorder

```python
rec = NodeRecorder(
    recID=1,
    nd=[1, 2, 3],        # node tags — list or single int (backward compat)
    dofs=[1, 2],          # DOF indices to record
    results='disp',       # 'disp' | 'vel' | 'accel' | 'reaction'
    file='output.txt'     # optional — incremental write each step
)

# Data formats:
# Single node:  rec.data['disp'] = [val_step0, val_step1, ...]
# Multi-node:   rec.data['disp'] = [{1: val, 2: val, 3: val}, ...]

rec.time           # list of time values
rec.save('path/')  # batch write
```

### ElementRecorder

```python
rec = ElementRecorder(recID=2, ele=[10,11],
                       results='stress', file='elem.txt')
# results: 'strain' | 'stress' | 'force' | 'tangent'
```

### ModeShapeRecorder

```python
# Auto-triggered by domain._record_eigen() — do not call manually

rec.data['mode_1']        # dict: {nodeID: [phi_vals]}
rec.data['eigenvalue_1']  # float
rec.data['omega_1']       # float
rec.data['freq_1']        # float
phi = rec.getNodeModeShape(mode=1, nodeID=2)
rec.save('modes.txt')
```

---

## 11. Multiscale FE² Architecture (Strategic Target)

The trial/committed state protocol at every level enables recursive nesting.
A `SimulationManager` that solves a micro-scale RVE **is** a `Material`
from the macro scale's perspective:

```
MacroSimulationManager
  └── Domain
        └── Element
              └── Material = MicroSimulationManager
                    └── Domain
                          └── Element
                                └── Material (Elastic, EPP, ...)
```

### Interface Contract for FE²

```python
class MicroSimulationManager(Material):
    """Acts as a Material at the macro scale. Drives a full RVE solve."""

    def _setTrialStrain(self, macro_strain):
        # 1. Apply macro_strain as BCs on the RVE domain
        # 2. Run inner analyze() to convergence
        # 3. Compute homogenized stress and consistent tangent
        self._sig_hom_trial = ...   # volume-averaged stress
        self._C_hom_trial   = ...   # consistent homogenized tangent
        return 0

    def getStress(self):
        return self._sig_hom_trial

    def getTangent(self):
        return self._C_hom_trial

    def _commitState(self):
        self._inner_sim.domain._commitAllMaterials()
        self._sig_hom_commit = self._sig_hom_trial
        self._C_hom_commit   = self._C_hom_trial
        return 0

    def _revertToLastCommit(self):
        self._inner_sim.domain._revertAllMaterials()
        self._sig_hom_trial = self._sig_hom_commit
        self._C_hom_trial   = self._C_hom_commit
        return 0
```

This is why the two-state protocol is non-negotiable: each scale independently
manages its own convergence, and `revert` at any level cascades cleanly.

---

## 12. Error Message Convention

```python
raise ValueError("oneFEM.Node22._setDOF() - DOF index out of range")
raise TypeError("oneFEM.Domain.add() - expected Element, got str")
# Format: "oneFEM.{ClassName}.{methodName}() - {description}"
```

---

## 13. Adding New Components — Checklists

### New Uniaxial Material

- [ ] `model/material/uniaxial/your_material.py`
- [ ] `super().__init__(mat_id)` as first line of `__init__`
- [ ] Trial state: `_eps_trial`, `_sig_trial`, `_tangent_trial`
- [ ] Committed state: `_eps_commit`, `_sig_commit`, `_tangent_commit`
- [ ] `_commitState()` deep-copies all state variables
- [ ] `_revertToLastCommit()` deep-copies committed → trial
- [ ] `revertToStart()` resets all state to zero
- [ ] `getStrain()` returns `_eps_trial`
- [ ] `getStress()` returns trial stress
- [ ] `getTangent()` returns trial tangent (consistent, not initial)
- [ ] `getInitialTangent()` returns constant elastic stiffness
- [ ] `getCopy()` returns a deep clone with same params, fresh state
- [ ] Error messages with full class path prefix
- [ ] Benchmark script in `src/` with quantified pass criterion

### New Element

- [ ] `model/element/{type}/main.py`
- [ ] Inherit from `Element`
- [ ] Implement `_domain()`, `_update()`, `_commit()`, `_revert()`, `revertToStart()`
- [ ] Implement `getTangentStiff()`, `getInitialStiff()`, `getResistingForce()`
- [ ] Implement `getMass()` — return zero matrix if element has no mass
- [ ] Implement `getDamp()` — default Rayleigh is fine initially
- [ ] Implement `zeroLoad()`, `addLoad()` — required for elemental loads
- [ ] Implement `getResistingForceIncInertia()` — for transient analysis
- [ ] For beam elements: accept `CrdTransf` as constructor argument
- [ ] Do NOT call `super()._commit()` or `super()._update()`
- [ ] `_attr` (single underscore) for all instance variables
- [ ] Local stiffness `k` as `Matrix`, local force `f` as `Vector`
- [ ] `_update()` reads nodal displacements AFTER `Domain._commit()` runs
- [ ] Benchmark script validates against analytical or published solution

### New Analysis Algorithm

- [ ] `analysis/algorithm/your_algorithm.py`
- [ ] Implement `solve(model, uu, pp) -> int`
- [ ] Register in `Analysis.__add_analysis` with independent `if` (not `elif`)
- [ ] Convergence via `Test` component

### New Recorder

- [ ] `output/recorder/your_recorder.py`
- [ ] Only reads **committed** state — never trial state
- [ ] `data` dict: single → `[val, ...]`; multi → `[{id: val}, ...]`
- [ ] `time` list mirrors `data` list length
- [ ] `save(path)` for batch output; optional `file=` for incremental

---

## 14. Known Pitfalls (from CLAUDE.md)

| Pitfall | Correct pattern |
|---|---|
| Node coordinate attribute | `_coord` not `_coords` |
| Vector/Matrix internals | `.data` property, never `__data` directly |
| `isinstance` for numpy | `isinstance(x, np.ndarray)` not `isinstance(x, array)` |
| Parallel check | Cross product ≈ 0 (not dot product ≈ 0) |
| Analysis component registration | Independent `if` blocks, not `elif` chain |
| Pipeline order | `Domain._commit()` before `element._update()/_commit()` |
| Commit/revert copy | `Vector(self._u_trial)` not `= self._u_trial` |
| Mutable defaults | `None` in constructor, not `[]` or `Material()` |
| Node subclass attributes | `_attr` single underscore |
| Material subclass init | Always call `super().__init__(mat_id)` |
| Section nonlinear tangent | Update `self._C` from `getTangent()` not `getStress()` |
| Convergence test return codes | `0`=converged, `-1`=continue, `-2`=failed (not `1` for continue) |
| Material `getCopy()` missing | FiberSection and Parallel/Series silently share state without it |
| `getStrain()` missing | Recorders and fiber integration loops call it — must return `_eps_trial` |
| `revertToStart()` missing | Restart from step 0 and material reuse in FiberSection require it |
| Element `getMass()` missing | Transient analysis crashes silently without mass matrix |
| Beam element without `CrdTransf` | Hardcoded local axes — geometric nonlinearity impossible |

---

## 15. Test Coverage and Benchmarks

Run all from `src/`:

| Script | What it tests | Pass criterion |
|---|---|---|
| `src/truss.py` | 3D triangle truss static | Analytical direct stiffness |
| `src/dynamic_truss.py` | SDOF Newmark vs closed-form | rel error < 1e-4, 3 configs |
| `src/eigen_truss.py` | Eigenvalue / modal analysis | rel error < 1e-10, 4 scenarios |
| `src/epp_truss.py` | Nonlinear Newton + EPP material | 4 sub-tests incl. regression |

**When adding a new component, add a benchmark script** with a quantified
pass criterion (e.g., `rel error < 1e-6` vs. analytical or published reference).

---

## 16. WP Mapping — Python Implementation Targets

All WPs are implemented in Python using hardware-native backends.
There is no C++ port planned. Performance comes from CuPy, mpi4py, and PyTorch.

| WP | oneFEM component | Python backend | What to implement |
|---|---|---|---|
| WP1: Serializable | `model/` + `_systools/data/` | `h5py` | `serialize()` / `deserialize()` on Node, Material |
| WP3: Domain | `model/main.py` | pure Python | Existing — extend with HDF5 checkpoint |
| WP4: Materials | `model/material/` | pure Python + numpy | New materials; commit/revert pattern established |
| WP5: Elements | `model/element/` | pure Python + numpy | New elements; CrdTransf for beams |
| WP6: Static/Dynamic | `analysis/` | scipy sparse | Working — Newton + Newmark benchmarked |
| WP7: Python API | `_systools/simulation_manager.py` | pure Python | OpenSees-compatible Layer 0 shim |
| WP8: HDF5 I/O | new `output/archive/hdf5_archive.py` | `h5py` | `HDF5Archive`; lz4 compressed datasets |
| WP9: Checkpoint | `Domain` + `HDF5Archive` | `h5py` + `hashlib` | `serialize/deserialize` + SHA256 Tier-1 gate |
| WP10: Distributed | new `_systools/parallel/mpi_channel.py` | `mpi4py` | `MPIChannel`; Domain partitioning across ranks |
| WP11: Shared parallel | `analysis/integrator/` element loop | `concurrent.futures` | `ThreadPoolExecutor` over element assembly loop |
| WP13: CUDA solver | new `analysis/system/cupy_solver.py` | `cupy` + `cupyx` | `CuPySolver`; same interface as `FullGeneral` |
| WP19: Neural surrogate | new `model/material/neural/surrogate.py` | `torch` / `jax` | `NeuralMaterialSurrogate`; autograd consistent tangent |
| WP23-24: UnifiedAllocator | new `_systools/allocator.py` | `cupy.cuda.UnifiedMemory` | `UnifiedAllocator`; zero-copy SOE on SoC |
| FE² | new `_systools/multiscale.py` | pure Python | `MicroSimulationManager` implements Material interface |

### Backend Install Reference

```bash
# GPU solve (WP13)
pip install cupy-cuda12x          # adjust to your CUDA version
# or for ROCm:
pip install cupy-rocm-5-0

# Distributed (WP10)
pip install mpi4py

# Neural surrogate (WP19)
pip install torch                  # or: pip install jax[cuda]

# HDF5 I/O (WP8-9)
pip install h5py

# Benchmark / unified memory validation
pip install cupy-cuda12x h5py     # already covered above
```

---

## 17. OpenSees C++ → oneFEM Python Mapping

Full cross-reference for when reading OpenSees source to understand oneFEM behavior.

| OpenSees (C++) | oneFEM (Python) | Status | Notes |
|---|---|---|---|
| `Domain` | `Domain` (model/main.py) | ✅ Exists | Central container |
| `Node` | `Node` + `Node{nDim}_{nDOF}` | ✅ Exists | Same specialization pattern |
| `Element` | `Element` (model/element/main.py) | ✅ Exists | Same interface |
| `UniaxialMaterial` | Material base + uniaxial/ | ✅ Exists | Same trial/commit |
| `NDMaterial` | material/nD/ | ✅ Exists | Same interface |
| `SectionForceDeformation` | Section (element/section/) | ⚠️ Partial | Rectangular only; FiberSection future |
| `CrdTransf` | (future) | 🔲 Planned | Required for beam/frame elements |
| `LoadPattern + TimeSeries` | pattern/ + tseries/ | ✅ Exists | Same pattern |
| `SP_Constraint` | model/constraint/ | ✅ Exists | Single-point BCs |
| `MP_Constraint` | (future) | 🔲 Planned | Multi-point constraints |
| `StaticAnalysis` | `Analysis` (analysis/main.py) | ✅ Exists | Unified for static + dynamic |
| `EquiSolnAlgo` | algorithm/ | ✅ Exists | Linear, Newton, KrylovNewton |
| `StaticIntegrator` | integrator/static/ | ✅ Exists | LoadControl, DisplacementControl |
| `TransientIntegrator` | integrator/dynamic/ | ✅ Exists | Newmark, CentralDifference |
| `LinearSOE` | analysis/system/ | ✅ Exists | FullGeneral, UMFPACK |
| `ConvergenceTest` | analysis/test/ | ✅ Exists | NormUnbalance, NormDispIncr |
| `ConstraintHandler` | analysis/constraints/ | ✅ Exists | Plain, Penalty |
| `DOF_Numberer` | analysis/numberer/ | ✅ Exists | Plain, RCM |
| `Recorder` | output/recorder/ | ✅ Exists | Node, Element, ModeShape |
| `Vector` | `_systools/data/vector.py` | ✅ Exists | Wraps numpy |
| `Matrix` | `_systools/data/matrix.py` | ✅ Exists | Wraps numpy |
| `ID` | Python list / numpy int array | ✅ Simplified | No custom class needed |
| `TaggedObjectStorage` | Python dict | ✅ Simplified | O(1) by default |
| `AnalysisModel` | Integrated into Analysis | ✅ Simplified | No FE_Element/DOF_Group wrapper |
| `FE_Element` | Not needed | ✅ Simplified | Python simplification |
| `DOF_Group` | Not needed | ✅ Simplified | Python simplification |
| `MovableObject` / `Channel` | Not needed | ✅ Simplified | Use pickle for serialization |
| `FEM_ObjectBroker` | Not needed | ✅ Simplified | Python dynamic typing |
| `Eigen` | analysis/eigen/ | ✅ Exists | scipy eigh/eigsh |
| `ArcLength integrator` | (future) | 🔲 Planned | For snap-through/snap-back |
| `HHT / GeneralizedAlpha` | (future) | 🔲 Planned | Numerical damping in dynamics |
| `FiberSection` | (future) | 🔲 Planned | Inelastic beam-column |
| `ParallelMaterial` | (future) | 🔲 Planned | Material composition |
| `SeriesMaterial` | (future) | 🔲 Planned | Material composition |
| `Rayleigh damping` | Partially in Domain | ⚠️ Partial | `setRayleighDampingFactors` needed |
| `OPS_Stream` | Python file I/O | ✅ Simplified | Standard file/HDF5 |

---

## 18. Strategic Protocol Layer (Future Direction)

The current codebase uses implicit duck-typing (if it has `_setTrialStrain`,
it's a material). As the codebase grows toward WP19 and FE², formalizing
this with `typing.Protocol` will prevent silent interface mismatches.

**Proposed future addition** (do not implement without a team decision):

```python
# model/protocols.py  (new file — does not replace existing classes)
from typing import Protocol, runtime_checkable

@runtime_checkable
class UniaxialMaterial(Protocol):
    def _setTrialStrain(self, strain, strain_rate=0.0): ...
    def _commitState(self): ...
    def _revertToLastCommit(self): ...
    def revertToStart(self): ...
    def getStrain(self): ...          # current trial strain
    def getStress(self): ...
    def getTangent(self): ...
    def getInitialTangent(self): ...
    def getCopy(self): ...            # deep clone — required for FiberSection

@runtime_checkable
class FEMElement(Protocol):
    def _domain(self): ...
    def _update(self): ...
    def _commit(self): ...
    def _revert(self): ...
    def revertToStart(self): ...
    def getTangentStiff(self): ...
    def getInitialStiff(self): ...
    def getResistingForce(self): ...
    def getMass(self): ...
    def zeroLoad(self): ...
```

`@runtime_checkable` means `isinstance(obj, UniaxialMaterial)` works as
a validation check without any changes to existing classes. `Elastic` and
`ElasticPerfectlyPlastic` satisfy this protocol today without modification.
Add this file when implementing WP19 to validate the neural surrogate at
injection time.

