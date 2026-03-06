# Unified Kinematics System — Design Questionnaire
**Status: COMPLETE — All answers filled. Version 1.0 — March 2026**

---

## Instructions for Claude Code — READ THIS FIRST

You are starting a new implementation session for the oneFEM unified kinematics system.
This document contains every binding design decision made by the PI. Your first task —
before writing a single line of code — is to produce a detailed implementation plan.

### Step 1: Read everything in this document

Compliance rule: Every [PI] answer and every item in the Summary of Decisions (Section X5) is a hard constraint. Before proposing any design or writing any code, check it against these answers. If a proposed approach conflicts with a documented decision, it must be rejected and the documented decision followed — not presented as an alternative option for the PI to rechoose. If a genuine ambiguity or gap exists that the questionnaire does not cover, stop and ask. Do not fill gaps with independent judgment.

Read every section of this document in full before doing anything else:
- All Q&A answers (sections A through T)
- Summary of Decisions (28 rules)
- Math module interface specification (X1)
- Final directory and file tree (X2)
- Element implementor checklist (X3)
- Kinematics implementor checklist (X4)
- Design rules (X5)
- Deferred decisions (X6)

Also read, in this order:
1. `docs/oneFEM_blueprint.md` — full architecture reference
2. `CLAUDE.md` — coding conventions and stop-and-ask rules
3. `docs/oneFEM_Validation_Schedule.md` — complete benchmark specifications

### Step 2: Produce a detailed implementation plan

Divide the work into exactly **3 or 4 phases**. Each phase must:
- Have a clear, single objective stated in one sentence
- List every file to be created or modified, with the exact path
- List every class and method to be implemented, with signatures
- End with a **validation step** that produces plots and confirms correctness
  before the next phase begins
- Be completable in a single focused session without interruption

The phases must follow the implementation order from P1:
1. Node fixes + kinematics hierarchy skeleton + math module extensions + math module API verification (see note below)
2. `ContinuumElement` base + `Quad4` Linear — validated by patch test (Quad4 only) and Cook's membrane
3. `Quad4` TL and UL — validated by simple shear unit test, cantilever rollup, and thick-walled cylinder
4a. Beam refactor only — merge `coordTransformation/` into `kinematics/`, no behavioral change,
    re-run all existing beam benchmarks to confirm zero regression
4b. `CorotContinuumKinematics` interface stub (NotImplementedError body) — validated by
    snap-through arch and Lee's frame using **existing beam/truss Corotational elements only**
    (Quad4 Corot not implemented in this WP)

> **Phase 4 split rationale:** The beam refactor carries regression risk — any broken CrdTransf
> path invalidates all existing beam benchmarks. Isolating it in 4a allows regression
> confirmation before any new corot-continuum stub code is introduced in 4b.

> **Math module API verification (must complete before Phase 2):** Before writing any
> element integration loop code, verify that the math module supports every operation in
> the X1 interface table. Specifically confirm:
> - `CTensor(4th).__matmul__(Matrix)` → Matrix (or explicit `C.to_matrix()` path works)
> - In-place slice assignment on `Matrix` and `Vector` (needed for B-matrix construction)
> - Equivalent of `numpy.einsum` or batched outer products (needed for K_σ construction)
> If any operation is missing, implement it in the math module **before Phase 2** — not
> inline in element code. Write a `tests/math/test_integration_loop_ops.py` that exercises
> every X1 operation on concrete small examples and confirms numerical correctness.
> This test suite is a Phase 1 deliverable.

Each validation step must produce **matplotlib plots** saved to `docs/validation/`:
- Load-displacement curves
- Deformed mesh plots
- Convergence plots (Newton-Raphson iteration count per step)
- Error vs. analytical solution tables printed to console

### Step 3: Write the plan to disk immediately

**Before writing any code**, save the implementation plan as:

```
docs/unified_kinematics_plan.md
```

The plan must be complete enough that if this session is interrupted, a new Claude Code
session can read `docs/unified_kinematics_plan.md` and continue from exactly where
this session stopped — with no information loss.

The plan must include:
- Phase objectives and scope
- File and class inventory per phase
- Method signatures per class
- Validation criteria (pass/fail, reference values, plot descriptions)
- A status table updated at each phase completion:

| Phase | Objective | Status | Validation |
|-------|-----------|--------|------------|
| 1 | Node fixes + hierarchy skeleton + math API verification | ⬜ Pending | — |
| 2 | Quad4 Linear | ⬜ Pending | Patch test (Quad4), Cook's membrane |
| 3 | Quad4 TL + UL | ⬜ Pending | Simple shear, cantilever rollup, thick-walled cylinder |
| 4a | Beam refactor only | ⬜ Pending | Re-run all existing beam benchmarks (zero regression) |
| 4b | CorotContinuumKinematics stub + benchmarks | ⬜ Pending | Snap-through (pre-snap only), Lee's frame |

### Step 4: Implement phase by phase

Only after `docs/unified_kinematics_plan.md` is written and confirmed:
- Implement Phase 1 completely
- Run the Phase 1 validation — produce all plots — confirm pass
- Update the status table in `docs/unified_kinematics_plan.md`
- Then implement Phase 2, and so on

**Stop-and-ask rules from CLAUDE.md still apply throughout.**
**Never start the next phase if the current phase validation has not passed.**

---

## Context for Future Sessions

**Read this section first. It tells you exactly what this document is and how to use it.**

### What is this?

This is a design questionnaire for building a **unified kinematics system** in the oneFEM
finite element package. The PI (Onur Deniz Akan) has a complete design in mind for how
geometric nonlinearity should work across ALL element types. This document captures every
design decision via Q&A so that any Claude Code session can reconstruct the full plan.

### The Goal

Every element in oneFEM (truss, beam, continuum, shell, zerolength, contact — present and
future) will support **four geometric nonlinearity formulations** through an injected
**kinematics strategy object**:

1. **Linear** — small strain, small displacement
2. **Corotational** — large rotation, small strain (element-attached frame rotates)
3. **Total Lagrangian (TL)** — large strain, large displacement, reference = initial config
4. **Updated Lagrangian (UL)** — large strain, large displacement, reference = last converged config

The interface pattern is modeled after the existing **CrdTransf** (coordinate transformation)
used by beam elements. CrdTransf already inherits from a `Kinematics` base class.

### One-to-One Ports from OpenSees C++

Two OpenSees C++ elements will be ported **one-to-one** into oneFEM Python. Their C++
source code defines the specification:

1. **ZeroLengthContactASDimplex** (Akan, Petracca, Camata, Spacone, Lai — 2020)
   - Source: `OPENSEES SRC/element/zeroLength/ZeroLengthContactASDimplex.cpp`
   - Node-to-node contact with Coulomb friction
   - Penalty method (Kn, Kt, mu)
   - Internal constitutive model (no separate Material object)
   - IMPL-EX and implicit integration modes
   - Orientation vector defines local coordinate system
   - Supports mixed DOF counts between nodes (2D: 2-3 DOF, 3D: 3-4-6 DOF)
   - Rotation matrix T (3x3 and 6x6) for local↔global transformation
   - B matrix = trivial [-I, I] displacement jump

2. **ASDShellQ4** (Petracca, Camata — ASDEA)
   - Source: `OPENSEES SRC/element/shell/ASDShellQ4.cpp`
   - 4-node general shell, NOT assumed flat (handles warped geometries)
   - AGQ formulation for in-plane (membrane) behavior
   - MITC4 formulation for out-of-plane (shear) behavior
   - Linear and corotational kinematics via transformation object strategy pattern:
     `ASDShellQ4Transformation` (linear) / `ASDShellQ4CorotationalTransformation` (corot)
   - Uses `SectionForceDeformation` (8 generalized strains: 3 membrane + 3 bending + 2 shear)
   - Hughes-Brezzi drilling DOF penalty (elastic and nonlinear modes)
   - Section orientation angle for orthotropic materials
   - AGQI enhanced membrane with internal DOFs (EAS)
   - 4 Gauss points (2x2), one section copy per Gauss point
   - `calculateAll()` with bitflags for efficient combined update/LHS/RHS

### What Exists Already (as of 2026-03-05, branch: `python_module`)

**Kinematics hierarchy (new, in `model/element/kinematics/`):**
- `Kinematics` — base class with `formulation` tag, `commitState()`, `revertToLastCommit()`, `copy()`
- `ContinuumKinematics(Kinematics)` — intermediate base for continuum elements, interface:
  `initialize()`, `update()`, `getStrain()`, `getBMatrix()`, `getGeometricStiffness()`
- `LinearContinuumKinematics(ContinuumKinematics)` — infinitesimal strain B matrix (2D/3D),
  returns CTensor strain, no geometric stiffness

**CrdTransf hierarchy (existing, in `model/element/coordTransformation/`):**
- `CrdTransf(Kinematics)` — beam coordinate transformation base
- `LinearCrdTransf2d/3d` — small deformation, fixed axes (benchmarked)
- `PDeltaCrdTransf2d/3d` — P-delta geometric nonlinearity (benchmarked)
- `CorotCrdTransf2d/3d` — corotational large rotation (benchmarked)

**nD Material (modified):**
- `nDMaterial(Material)` — base using CTensor for strain (COV) / stress (CONTR) / tangent (CONTR)
- `ElasticIsotropic(nDMaterial)` — PlaneStress, PlaneStrain, 3D; fully implemented

**CTensor (`_systools/data/ctensor.py`):**
- Full Python translation of CTensor.h/.cpp; Voigt notation; 5 matrix representations
  (Helnwein 2001); `__xor__` for 4th:2nd contraction (C:eps -> sigma)

**Elements:**
- `Truss` — working, does its own n-outer-n transform inline (no kinematics object)
- `ElasticBeamColumn2d/3d` — working, uses CrdTransf (benchmarked)
- `ZeroLength`, `ZLContact`, `ShellQ4` — empty stubs

**Continuum directory (`model/element/continuum/`):**
- `__init__.py` imports `ContinuumElement` from `main.py` (which doesn't exist yet)

### How to Use This Document

1. Read the Q&A section below — every answer is a binding design decision from the PI
2. Read `docs/oneFEM_blueprint.md` — the full architecture reference
3. Read `CLAUDE.md` — coding conventions, stop-and-ask rules, status tables
4. Use the answers to build an implementation plan, then implement step by step
5. **Write the plan to disk before writing any code** — save it as `docs/unified_kinematics_plan.md`

### Refactoring Freedom

**Backwards compatibility is NOT required.** Existing element code (Truss, ElasticBeamColumn,
etc.) can be freely refactored to use the unified kinematics system. Breaking existing APIs
is fine as long as:
1. The refactored code **works** — all existing benchmarks still pass
2. The math and physics are **accurate** — verified against analytical solutions
3. The code runs **fast** — no unnecessary overhead from abstraction layers

This means you can change constructors, rename methods, restructure class hierarchies,
and rewrite element internals. The only thing that matters is correctness and performance.

### Key Constraints (from CLAUDE.md and blueprint)

- No new top-level directories under `src/oneFEM/` without PI approval (Stop-and-Ask rule 3)
- Material interface is sacred — do not change it (Stop-and-Ask rule 1)
- Pipeline order is fixed: `Domain._commit()` before `element._update()/_commit()` (rule 2)
- No C++ extensions — Python with backends is the project (blueprint Section 0)
- Trial/committed two-state pattern on every stateful object (blueprint rule 1)
- Deep copy in commit/revert, never alias (blueprint rule 1)
- Single underscore `_attr` for all non-Domain classes (blueprint rule 4)
- `super().__init__(mat_id)` required in all Material subclasses (blueprint rule 8)
- Element subclasses must NOT call `super()._commit()` or `super()._update()` (blueprint rule 10)

---

## Design Questions & Answers

**Legend:**
- **[PI]** = Answer provided directly by the PI in this session.
- **[PRE-FILLED]** = Answer inferred from PI's statements, blueprint, or C++ source code and confirmed by PI.

---

### A. Overall Architecture

**A1.** The four formulations are Linear, Corotational, Total Lagrangian, and Updated Lagrangian.
Does every element type get ALL four, or do some element types only support a subset?
(e.g., PDelta is currently beam-only — does it survive as a 5th option, get folded into
Corotational, or get dropped?)

**A1 Answer [PI]:** Every element type — truss, beam, continuum, shell, zero-length, and
contact — supports all four geometric nonlinearity formulations: Linear, Corotational, Total
Lagrangian, and Updated Lagrangian. Each element implements a set of internal methods (beyond
the standard OpenSees API) that express the formulation-specific kinematics for that element
type; new elements are designed and coded with these methods from the start. The `CrdTransf`
hierarchy remains the mechanism for beams, and the same strategy-pattern philosophy extends
to all other element branches.

PDelta is retained as a **beam-only, 5th option** within `CrdTransf` for backward
compatibility, but it is not part of the unified four-formulation system and carries low
priority. It will not be extended to other element types.


**A2.** The `Kinematics` base class currently has `commitState()`, `revertToLastCommit()`, `copy()`.
The `CrdTransf(Kinematics)` base adds the beam-specific interface (`initialize`, `update`,
`getBasicTrialDisp`, `getGlobalStiffMatrix`, `getGlobalResistingForce`, etc.).
Should the unified `Kinematics` base contain a richer shared interface that ALL element types
use? Or should it remain minimal, with element-type-specific intermediate bases
(`CrdTransf` for beams, `ContinuumKinematics` for solids, etc.) adding their own methods?

**A2 Answer [PI]:** The `Kinematics` base class should contain every method that is shared
across multiple element-type branches, following standard OOP inheritance. Common behaviour —
such as `initialize()`, `update()`, `commitState()`, `revertToLastCommit()`, and `copy()` —
is implemented once in the appropriate ancestor and inherited by all descendants. Duplication
across sibling classes is avoided. Element-type-specific methods (e.g., `getBasicTrialDisp()`
for beams, `getBMatrix()` for continuum) live in the appropriate intermediate base
(`CrdTransf`, `ContinuumKinematics`, etc.), not repeated in every leaf class. The guiding
principle is: **fix it once in the base, and all children benefit**.


**A3.** Currently `CrdTransf` operates at the **element level** (one object per element,
transforms the whole element stiffness/force from local to global). `ContinuumKinematics`
operates at the **integration point level** (called per Gauss point with `dN_dX` and `u_e`).
Is this split intentional and should it remain? Or should one level be preferred for all?

**A3 Answer [PI]:** One kinematics object per element, always — architecturally consistent
across all element types. For beams, trusses, and contact, this is natural as there is no
per-Gauss-point variation. For continuum and shell elements, the kinematics object internally
manages arrays of per-Gauss-point state (deformation gradients, reference configurations,
B matrices, etc.), pre-allocated in `initialize()`. The element passes the GP index when
calling kinematics methods: `update(gp, ...)`, `getBMatrix(gp)`, `getStrain(gp)`, etc.
This mirrors `ASDShellQ4` where a single transformation object manages all internal GP-level
state. The rule is absolute: **one kinematics object injected at element construction,
regardless of element type.**


**A4.** Should every element's constructor accept a `kinematics` argument (defaulting to
Linear if not provided)? This means changing the Truss constructor, for example, from
`Truss(id, nodes, section, rho, cMass)` to `Truss(id, nodes, section, kinematics=None, rho, cMass)`.

**A4 Answer [PRE-FILLED]:** Yes. Every element takes a `kinematics` argument. The PI stated
"every element now and future will have linear, corotational, total lagrangian and updated
lagrangian formulations." Defaulting to Linear when not provided.


---

### B. Strain Measures and Material Interaction

**B1.** Each formulation uses different strain/stress conjugate pairs:
- Linear: infinitesimal strain eps / Cauchy stress sigma
- Corotational: small strain in corotated frame / Cauchy stress in corotated frame
- TL: Green-Lagrange strain E / 2nd Piola-Kirchhoff stress S
- UL (Bathe): incremental Green-Lagrange E / 2nd Piola-Kirchhoff S (reference = last committed config)

Does the kinematics handle the transformation so the **material always receives the same
strain measure** (e.g., always infinitesimal/engineering strain)? Or does the material need
to be formulation-aware?

This is critical for FE-squared — the micro-scale SimulationManager (acting as a Material)
shouldn't need to know what formulation the macro element uses.

**B1 Answer [PRE-FILLED]:** The kinematics handles the transformation so the material is
formulation-agnostic. The material always receives strain in its native measure and returns
stress and tangent without knowing the formulation. This is required by the FE-squared
architecture (blueprint Section 11): the micro-scale SimulationManager implements the
Material interface and cannot know what formulation the macro element uses. The kinematics
is responsible for mapping between the formulation's strain/stress measures and the
material's expected input/output.


**B2.** `CTensor` is currently used at the kinematics-material boundary:
`LinearContinuumKinematics.getStrain()` returns CTensor (2nd order, COV), and
`nDMaterial._setTrialStrain()` accepts CTensor. For the element integration loop
(B^T * C * B * detJ * w), we need numpy arrays for matrix multiplication.

What should the data flow look like?
- Option A: Kinematics returns CTensor -> Material accepts CTensor -> Element extracts numpy for integration
- Option B: Kinematics returns numpy -> Element wraps in CTensor only when calling material
- Option C: Everything stays as CTensor throughout (CTensor needs toArray() and fromArray())
- Option D: Something else?

**B2 Answer [PI]:** All element and kinematics code uses oneFEM native math objects
exclusively — `CTensor`, `Matrix`, `Vector`, and any future math objects — with no raw numpy
in element or kinematics files. The math module wraps numpy (or any future backend) internally
and exposes a clean, typed API. This means a backend change or method rename requires editing
only the math module, not every element file.

The integration loop operates entirely on native objects: `B` is a `Matrix`, stress and strain
are `CTensor`, and operations like `B.T @ C @ B` are implemented as math object methods.
Performance is preserved because all heavy computation delegates to numpy internally within
the math objects.

Cross-type operations required in the math module:
- `CTensor(4th) ^ CTensor(2nd)` → `CTensor(2nd)` — already exists via `__xor__`
- `CTensor(2nd).to_vector()` → `Vector` (Voigt column)
- `CTensor(4th).to_matrix()` → `Matrix` (Voigt nxn)
- `Matrix @ CTensor(4th) @ Matrix` → `Matrix` (for B^T C B)
- `Matrix @ CTensor(2nd)` → `Vector` (for B^T sigma)

The rule: **operations between native objects always return native objects. Raw numpy never
surfaces in element or kinematics code.**


**B3.** For nonlinear formulations (TL, UL, Corot), the B matrix depends on current
displacements. For TL, B = B_L + B_NL(u). For Corot, B is evaluated in the rotated frame.
Does the kinematics object own the current displacement state, or does the element pass
displacements to the kinematics at each call?

**B3 Answer [PI]:** The element owns the nodal displacement array `u_e` and passes it
explicitly to the kinematics at each call. Allocation is formulation-aware: for Linear
kinematics, B is constant and no displacement array is needed or allocated. For Corotational,
TL, and UL, the element pre-allocates `u_e` once during `initialize()` (when the formulation
is known) and fills it in-place each iteration — no per-iteration memory allocation. The
kinematics never holds node references; it receives what it needs when it needs it.

```python
# In initialize():
if self._kinematics.formulation != 'linear':
    self._u_e = Vector(nDOF_total)  # allocated once, reused forever

# In _update():
if self._kinematics.formulation != 'linear':
    self._extractDisplacements(self._u_e)  # fill in-place, no allocation
    self._kinematics.update(gp, dN_dX, self._u_e)
```


---

### C. Geometric Stiffness

**C1.** Linear formulation has no geometric stiffness. TL and UL have K_sigma (initial stress
stiffness). Corotational has geometric stiffness from the rotation. For beams, the CrdTransf
handles geometric stiffness inside `getGlobalStiffMatrix(kb, q)` — it receives the basic
force `q` and adds the geometric terms.

For continuum elements, who computes the geometric stiffness?
- Option A: The kinematics computes it (given stress at the integration point)
- Option B: The element computes it (using information from kinematics)
- Option C: The kinematics returns a modified B matrix that implicitly includes geometric effects

**C1 Answer [PRE-FILLED]:** Option A — the kinematics computes geometric stiffness, given
the current stress state at the integration point. This follows the CrdTransf pattern where
`getGlobalStiffMatrix(kb, q)` receives the basic force and internally adds geometric
stiffness terms. The element does not need to know whether geometric stiffness exists — it
is transparent, handled entirely by the kinematics strategy.


**C2.** For beam CrdTransf, the geometric stiffness depends on the basic forces (axial force N,
moments M). For continuum elements, it depends on the stress tensor at each Gauss point.
Are these fundamentally different enough to justify different interfaces, or should there
be a unified way to pass "stress state" to the kinematics for geometric stiffness computation?

**C2 Answer [PRE-FILLED]:** Different interfaces are justified. Beam CrdTransf receives basic
forces (Vector), while continuum kinematics receives the stress tensor (CTensor) at each
Gauss point. These are fundamentally different quantities at different abstraction levels.
The `Kinematics` base class does not need to unify them — the element-type-specific
intermediate bases (`CrdTransf`, `ContinuumKinematics`, etc.) define the appropriate
interface for their element type.


---

### D. Volume Integration and Reference Configuration

**D1.** Linear and TL integrate over the initial (reference) volume using `detJ_0`.
UL integrates over the current (deformed) volume using `detj`.
Corotational integrates over the initial volume in the corotated frame.

Does the kinematics return the appropriate Jacobian determinant for the current formulation?
Or does the element compute the Jacobian and the kinematics only tells it which configuration
to use?

**D1 Answer [PI]:** The element computes the Jacobian determinant, consistent with the
principle that element-specific geometric computations belong in the element. The kinematics
object exposes its `formulation` tag so the element knows which configuration to use: initial
(`detJ_0`) for Linear and TL, deformed (`detJ`) for UL, and corotated initial for
Corotational. For each new element type, the implementor is responsible for coding the
correct Jacobian computation for each formulation — the kinematics makes no assumptions
about element geometry.


**D2.** For Updated Lagrangian, the reference configuration updates at each committed step.
This means at `commitState()`, the reference coordinates change to the current deformed
coordinates. Does the kinematics store the reference coordinates internally, or does it
get them from the element/nodes each time?

**D2 Answer [PRE-FILLED]:** The kinematics stores the reference coordinates internally and
updates them at `commitState()`. This follows the trial/committed two-state pattern: the
UL kinematics maintains its own reference configuration state. At `commitState()`, the
current deformed coordinates become the new reference. At `revertToLastCommit()`, the
reference reverts to the last committed configuration. This is analogous to how
`ASDShellQ4CorotationalTransformation` in the C++ source has `commit()` and
`revertToLastCommit()` managing its internal state.


---

### E. Nodes and DOFs

**E1.** Current node types: Node22 (2D,2-DOF), Node23 (2D,3-DOF), Node24 (2D,4-DOF),
Node33 (3D,3-DOF), Node34 (3D,4-DOF), Node36 (3D,6-DOF), Node37 (3D,7-DOF).
Which node types are used by which element types?

**E1 Answer [PRE-FILLED]:**
- Truss 2D: Node22 or Node23. Truss 3D: Node33 or Node36 (uses translational DOFs only)
- Beam 2D: Node23. Beam 3D: Node36
- Quad4: Node22. Tri3: Node22
- Brick8: Node33. Tet4: Node33
- ShellQ4: Node36 (3 translations + 3 rotations, 24 DOFs total for 4 nodes — matches ASDShellQ4)
- ZeroLength: any pair of same-dimension nodes (Node22-Node22, Node23-Node23, Node36-Node36, etc.)
- Contact: any pair of same-dimension nodes, supports mixed DOF counts
  (2D: Node22 or Node23; 3D: Node33, Node34, or Node36 — matches C++ source)


**E2.** Node33 is flagged as broken (double-underscore mangling, wrong `_update` signature,
no `_commitState`). It's needed for 3D continuum elements (Brick8). Should it be fixed
now, or is 3D continuum not in immediate scope?

**E2 Answer [PI]:** All broken node types — Node22, Node33, and any others flagged as
broken or partial — are fixed in one pass at the start of this work package, before any
element implementation begins. This clears all node debt once and for all and removes it
as a recurring blocker for new element development.


**E3.** Are any new node types needed for the unified kinematics? For example:
- Nodes with enhanced strain DOFs?
- Nodes with pressure DOFs (mixed u-p formulations)?
- Nodes with bubble functions?

**E3 Answer [PRE-FILLED]:** No new node types needed for the unified kinematics itself.
Enhanced strain (EAS/AGQI) uses element-internal DOFs, not nodal DOFs (as seen in
ASDShellQ4 where `m_eas` manages internal DOFs separately from nodes). Mixed u-p
and bubble functions are future concerns, not part of this work package.


---

### F. Truss Elements

**F1.** Currently the truss computes its own direction cosines, `n (x) n` outer product, and
local-global transformation inline (no kinematics object). Should the truss be refactored
to use a kinematics strategy? If yes, what would `TrussKinematics` look like?

For a truss, "basic" = axial strain. The kinematics would:
- Linear: eps = (u_j - u_i) dot n / L_0 (engineering strain)
- TL: eps = (L^2 - L_0^2) / (2 * L_0^2) (Green-Lagrange)
- UL (Bathe): eps = (L^2 - L_ref^2) / (2 * L_ref^2) (incremental Green-Lagrange, L_ref updated at commit)
- Corot: eps = (L - L_0) / L_0 (engineering in corotated frame)

Is this the right decomposition?

**F1 Answer [PI]:** Truss kinematics formulation is deferred. The truss will be refactored
to the unified kinematics system in a later work package once the formulation decisions are
finalized. The existing truss implementation remains as-is for now. The strain decomposition
above is mechanically correct and will serve as the starting specification when truss porting
begins.


**F2.** The truss currently uses a `Section` (Rectangular) that wraps a uniaxial Material.
Should it keep this, or should it work directly with a uniaxial Material (with area as
an element property)?

**F2 Answer [PI]:** Deferred with F1. Decision will be made at truss porting time.


**F3.** For large-deformation truss (TL/UL/Corot), the direction cosines change with
deformation. The kinematics would need to recompute them. Does the kinematics own
the direction cosine vector `n`, or does the element own it and the kinematics updates it?

**F3 Answer [PI]:** Deferred with F1. Decision will be made at truss porting time.


---

### G. Beam Elements

**G1.** ElasticBeamColumn2d/3d currently takes E, A, I directly (no material object) and
uses CrdTransf. With the unified kinematics:
- Does the constructor signature change?
- Does CrdTransf get renamed or restructured?
- Do the existing Linear/PDelta/Corot CrdTransf implementations get refactored?

**G1 Answer [PI]:** `ElasticBeamColumn2d/3d` is not refactored. Its constructor signature
and `CrdTransf` interface remain unchanged. The beam kinematics story is completed when
`forceBeamColumn`, `dispBeamColumn`, and `FiberSection` are ported one-to-one from OpenSees
in a future work package — those elements will use the existing `CrdTransf` hierarchy
directly, consistent with how OpenSees uses it.


**G2.** For beams, CrdTransf currently handles: `initialize(node_i, node_j)`, `update()`,
`getBasicTrialDisp()`, `getGlobalStiffMatrix(kb, q)`, `getGlobalResistingForce(q, p0)`.
Is this interface complete for all 4 formulations, or does TL/UL for beams require
additional methods?

**G2 Answer [PRE-FILLED]:** The CrdTransf interface is complete for Linear, PDelta, and
Corotational. TL and UL for beams are uncommon (most beam large-deformation uses
corotational), but if implemented, the same interface should suffice — the kinematics
internally changes how it computes basic deformations and transforms stiffness/force,
without exposing new methods to the element.


**G3.** PDelta is currently a separate CrdTransf. In the unified system with only 4
formulations (Linear, Corot, TL, UL), what happens to PDelta?
- Keep as a 5th beam-only option?
- Fold into Corotational as a simplified mode?
- Deprecate it?

**G3 Answer [PI]:** PDelta is retained as a 5th beam-only option within `CrdTransf`,
unchanged. It is not extended to other element types and carries low implementation priority.


**G4.** For nonlinear beams (future forceBeamColumn with FiberSection), will the kinematics
strategy be the same CrdTransf, or does the fiber integration require something different?

**G4 Answer [PRE-FILLED]:** Same CrdTransf. The kinematics strategy handles geometry
(local-global transformation, geometric nonlinearity). The fiber integration is a
section-level concern (SectionForceDeformation integrates fiber stresses over the
cross-section). These are orthogonal: CrdTransf transforms between basic and global frames,
while FiberSection computes section force-deformation response. OpenSees uses exactly this
separation — forceBeamColumn uses the same CrdTransf hierarchy as elasticBeamColumn.


**G5.** Beam mass matrix computation currently lives in the element (ElasticBeamColumn2d/3d).
Should it stay in the element, or does the kinematics affect mass (e.g., for UL where
the reference config changes)?

**G5 Answer [PRE-FILLED]:** Mass stays in the element. Mass is conserved regardless of
formulation. Even for UL where the reference configuration updates, the total mass of the
element doesn't change — only the geometry over which it is distributed changes. The mass
matrix may need to be recomputed with updated geometry for UL, but this is still the
element's responsibility using updated nodal coordinates from the kinematics.


---

### H. Continuum Elements (Quad4, Tri3, Brick8, Tet4)

**H1.** Which continuum elements are in scope for the unified kinematics implementation?
All at once, or in a specific order?

**H1 Answer [PI — CORRECTED]:** **Quad4 is fully in scope for this work package** with all
four formulations (Linear, TL, UL, Corotational) — it is the primary proof-of-concept
element that validates the entire architecture. The minimum viable deliverable (P2) requires
Quad4 passing all 8 benchmarks.

The following are deferred to future work packages, with source code provided by the PI at
porting time: Tri3, Brick8, Tet4, SSPQuad4UP, SSPBrick8UP. These will plug into the
framework established in this WP without requiring any architectural changes.


**H2.** For Quad4 — what's the default integration scheme?
- 2x2 full integration
- 1-point reduced integration with hourglass control
- Both as options?

**H2 Answer [PRE-FILLED]:** 2x2 full integration as default. This matches ASDShellQ4 (which
uses 4 Gauss points at +/-0.577) and is the standard for bilinear Quad4 elements. Reduced
integration with hourglass control can be added as an option later.


**H3.** Does each Gauss point get its own independent material copy (via `getCopy()`)?
This is the standard approach and is required for path-dependent materials.

**H3 Answer [PRE-FILLED]:** Yes. Each Gauss point gets its own independent material copy
via `getCopy()`. This is the standard approach, required for path-dependent (nonlinear)
materials where each integration point can have different stress/strain history. This
matches ASDShellQ4 where `m_sections[i] = section->getCopy()` creates 4 independent
section copies.


**H4.** PlaneStress vs PlaneStrain — currently this is a property of `ElasticIsotropic`
(the material). Should it be:
- A material property (current — material knows its formulation type)?
- An element property (element tells material what type to use)?
- A kinematics property?

**H4 Answer [PRE-FILLED]:** Material property (keep current). `ElasticIsotropic` already
takes `type='PlaneStress'|'PlaneStrain'|'3D'` in its constructor and builds the
appropriate tangent. This is the correct level — the material defines the constitutive
response, and different materials for the same element can have different plane assumptions.
This also matches OpenSees where NDMaterial subclasses define their own plane type.


**H5.** Who owns the shape functions and their derivatives?
- The element (and passes `dN_dX` to kinematics)?
- The kinematics (internally computes shape functions)?
- A separate isoparametric utility module (pure functions, no state)?

**H5 Answer [PRE-FILLED]:** Separate isoparametric utility module (pure functions, no state).
This matches ASDShellQ4 where `shapeFunctions()` and `shapeFunctionsNaturalDerivatives()`
are standalone functions in an anonymous namespace. Shape functions are element-topology-
specific (Quad4, Tri3, Brick8), not formulation-specific, so they don't belong in the
kinematics. A utility module keeps them reusable across element types and kinematics
formulations. References: Hughes (2000), Bathe (1996).


**H6.** For continuum elements, what DOFs do nodes contribute? Purely translational
(ux, uy for 2D; ux, uy, uz for 3D)? Or can continuum elements also connect to nodes
with rotational DOFs (drilling DOFs)?

**H6 Answer [PRE-FILLED]:** Purely translational for continuum (solid) elements. Drilling
DOFs are a shell concern (handled via Hughes-Brezzi penalty in ASDShellQ4), not a solid
element concern. Quad4 uses Node22 (2 translational DOFs), Brick8 uses Node33 (3
translational DOFs).


**H7.** Should the continuum element have a `ContinuumElement` base class (like `CrdTransf`
is a base for beam kinematics), or should each continuum element (Quad4, Tri3, Brick8)
directly inherit from `Element`?

**H7 Answer [PI]:** A `ContinuumElement` base class is introduced, from which all continuum
elements (Quad4, Tri3, Brick8, Tet4, SSPQuad4UP, SSPBrick8UP) inherit. Common continuum
logic — Gauss point setup, displacement extraction, integration loop structure, material copy
management, pre-allocated arrays — is implemented once in `ContinuumElement` and inherited
by all subclasses. This follows the same OOP inheritance principle as A2: shared behaviour
lives in the nearest common ancestor.


**H8.** For large-deformation continuum (TL/UL), the Jacobian changes. TL uses the initial
Jacobian but B_NL depends on displacements. UL re-maps everything to the deformed config.
How much of this complexity lives in the kinematics vs the element?

For example, does the element call:
```python
# Option A: Element does the loop, kinematics does point-level work
for gp in gauss_points:
    B = self._kinematics.getBMatrix(dN_dX, u_e, nDim)
    strain = self._kinematics.getStrain(dN_dX, u_e, nDim)
    K_geo = self._kinematics.getGeometricStiffness(dN_dX, stress, u_e, nDim)
    detJ = self._kinematics.getJacobian(dN_dxi, X_nodes, u_e)  # new method?

# Option B: Kinematics does the entire integration
K, f = self._kinematics.integrate(gauss_points, materials, X_nodes, u_e)
```

**H8 Answer [PI]:** The element drives the integration loop (Option A). The kinematics
object is a pure geometry servant — given a Gauss point index, it returns B matrix, strain,
and geometric stiffness on demand. It has no knowledge of materials, sections, or integration
weights. This preserves strict separation of concerns: geometry in kinematics, constitutive
response in materials, assembly logic in the element. It is also consistent with the beam
pattern where `CrdTransf` serves the element and never owns the integration loop. The
`ContinuumElement` base class owns the loop structure, and subclasses extend it as needed.


---

### I. Shell Elements

**I1.** What shell formulation is planned?
- Mindlin-Reissner (thick shell, includes transverse shear)?
- Kirchhoff (thin shell, no transverse shear)?
- Degenerated solid approach?
- MITC (Mixed Interpolation of Tensorial Components)?
- Multiple options?

**I1 Answer [PRE-FILLED]:** Mindlin-Reissner with MITC4 for shear locking elimination and
AGQI for enhanced membrane behavior. This is a one-to-one port of ASDShellQ4. The element
handles warped (non-flat) geometries. Not assumed flat.


**I2.** Shell element topology — which types?
- ShellQ4 (4-node quad)?
- ShellT3 (3-node tri)?
- ShellQ9 (9-node quad)?
- Others?

**I2 Answer [PRE-FILLED]:** ShellQ4 (4-node quad) first, as a one-to-one port of ASDShellQ4.
Other topologies (ShellT3, etc.) may follow later.


**I3.** Shell nodes — Node36 (3D, 6-DOF: ux, uy, uz, rx, ry, rz)?
Or a shell-specific node type with drilling rotation handling?

**I3 Answer [PRE-FILLED]:** Node36 (3D, 6-DOF). The ASDShellQ4 uses 6 DOFs per node (24
total for 4 nodes). Drilling rotation (rz normal to shell surface) is handled via the
Hughes-Brezzi penalty approach within the element, not by a special node type.


**I4.** Shell corotational is quite specialized (EICR — Element-Independent Corotational).
Does the unified kinematics handle this, or does the shell need its own kinematics branch
(like `ShellKinematics(Kinematics)` parallel to `CrdTransf` and `ContinuumKinematics`)?

**I4 Answer [PRE-FILLED]:** Shell gets its own kinematics branch under the `Kinematics` base.
In ASDShellQ4, the transformation is a shell-specific class hierarchy:
`ASDShellQ4Transformation` (linear) and `ASDShellQ4CorotationalTransformation` (corot).
The shell transformation handles local coordinate system creation, global-local displacement
mapping, and global transformation of LHS/RHS — these are shell-specific operations that
don't map to the beam CrdTransf or continuum kinematics interfaces.


**I5.** Shell elements need through-thickness integration (layers/plies for composite,
or Simpson points through thickness). Who owns this — the section, the material, or the
element?

**I5 Answer [PRE-FILLED]:** The section owns through-thickness integration. ASDShellQ4 uses
`SectionForceDeformation` which handles this internally. The section receives 8 generalized
strains (3 membrane + 3 bending + 2 shear) and returns 8 generalized stresses and an 8x8
tangent. Through-thickness integration (e.g., layered composite, fiber) is encapsulated
within the section implementation.


**I6.** Is shell implementation a near-term priority or planned for later? If later,
should the kinematics architecture still account for it now (design the interfaces so
shells fit in later without refactoring)?

**I6 Answer [PI]:** Shell implementation is not part of this work package. The kinematics
architecture is designed now to accommodate `ShellKinematics` as a first-class branch
alongside `CrdTransf` and `ContinuumKinematics`, so no refactoring is needed when porting
time comes. `ASDShellQ4` source code is available as the definitive reference for the shell
kinematics interface design.


---

### J. ZeroLength Elements

**J1.** What is the ZeroLength element used for in oneFEM?
- Point springs (translational/rotational)?
- Plastic hinges at beam ends?
- Interface elements between different meshes?
- All of the above?

**J1 Answer [PI]:** `ZeroLength` is a general-purpose element used for: point springs
(translational and rotational), plastic hinges at beam ends, interface elements between
different meshes, and viscous absorbing boundaries (Lysmer-Kuhlemeyer dashpot elements
for wave propagation and soil-structure interaction problems), among other applications.


**J2.** ZeroLength has zero geometric length, so geometric nonlinearity (TL/UL/Corot)
doesn't apply in the traditional sense. Does ZeroLength get a kinematics object at all?
If yes, what does it do? If no, is it an exception to the "every element gets kinematics" rule?

**J2 Answer [PI]:** `ZeroLength` is not an exception — it requires kinematics like every
other element. The orientation vectors supplied at construction define the initial local
coordinate axes, and under large deformation those axes rotate with the nodal motion. The
kinematics object manages this rotation, mapping between the (possibly rotating) local frame
and the global frame. For the Linear formulation the orientation is fixed; for Corotational
the local frame is updated each iteration from current nodal positions; TL and UL track the
full deformation gradient of the zero-length interface. The `ContactKinematics` branch
(orientation vector → rotation matrix T, trivial B = [-I, I]) serves both `ZeroLength` and
`ZeroLengthContactASDimplex` as their shared kinematics base.


**J3.** What materials does ZeroLength use?
- Uniaxial materials (one per DOF direction)?
- nD materials?
- Both?

**J3 Answer [PI]:** `ZeroLength` follows the OpenSees structure exactly: `ZeroLength` takes
a uniaxial material per DOF, `ZeroLengthND` takes an nD material, and `ZeroLengthSection`
takes a section. All three variants are deferred to a future porting work package — only
`ZeroLengthContactASDimplex` is ported in the current work package.


**J4.** What node types does ZeroLength connect? Two nodes at the same location?
Can they have different DOF counts (e.g., Node23 to Node36)?

**J4 Answer [PRE-FILLED]:** Two nodes at the same location. Yes, they can have different
DOF counts. The C++ ZeroLengthContactASDimplex explicitly supports this:
- 2D: nodes can have 2 or 3 DOFs each (Node22 or Node23)
- 3D: nodes can have 3, 4, or 6 DOFs each (Node33, Node34, or Node36)
The element scatters into the global DOFset using index offsets (`numDOF[0]`, `numDOF[1]`).


---

### K. Contact Elements

**K1.** What kind of contact is planned?
- Node-to-node (simplest)?
- Node-to-surface?
- Surface-to-surface (mortar)?
- All as progressive steps?

**K1 Answer [PRE-FILLED]:** Node-to-node first (one-to-one port of ZeroLengthContactASDimplex).
Node-to-surface and surface-to-surface may follow as future WPs.


**K2.** Is contact implemented as an Element subclass (ContactElement with nodes on each
surface), or as a Constraint (like MP_Constraint)?

**K2 Answer [PRE-FILLED]:** Element subclass. The C++ `ZeroLengthContactASDimplex` inherits
from `Element`. It participates in assembly like any other element — it returns tangent
stiffness, resisting force, and DOF mappings. Not a constraint.


**K3.** What is the contact constitutive model?
- Penalty method (spring-like, easy to implement)?
- Lagrange multiplier (exact, needs additional DOFs)?
- Augmented Lagrange?
- Nitsche's method?

**K3 Answer [PRE-FILLED]:** Penalty method with Coulomb friction. Parameters: Kn (normal
penalty stiffness), Kt (tangential penalty stiffness), mu (friction coefficient). This
matches the C++ source exactly.


**K4.** For contact in large deformation, the gap function and contact normal change with
deformation. Does the kinematics handle this (computing updated normals, gap, slip), or
is this the contact element's own responsibility?

**K4 Answer [PRE-FILLED]:** In the C++ source, the orientation is fixed (user-supplied
`Xorient`), not updated during deformation. The initial gap is computed once in `setDomain()`
and added to the displacement jump. Since the initial port is one-to-one, the orientation
stays fixed. For large-deformation contact where normals update, the `ContactKinematics`
interface is designed to accommodate this extension in a future dedicated WP without
refactoring.


**K5.** Is contact a near-term priority, or is it planned for future WPs? Should the
kinematics architecture account for contact now, or is it OK to add contact support later?

**K5 Answer [PI]:** `ZeroLengthContactASDimplex` is near-term but gets its own dedicated
work package. The initial one-to-one port from OpenSees C++ comes first, followed by a
separate WP that extends it to handle geometric nonlinearity (updating contact normal and
gap function with deformation). The current work package designs the `ContactKinematics`
interface to accommodate this extension without future refactoring.


---

### L. State Management

**L1.** For Updated Lagrangian, `commitState()` must update the reference configuration
(current deformed becomes the new reference). This is a non-trivial state change.
Does the kinematics `commitState()` handle this, or does the element trigger it?

**L1 Answer [PRE-FILLED]:** The kinematics `commitState()` handles it. This follows the
CrdTransf pattern where `commit()` is on the transformation object (see ASDShellQ4:
`m_transformation->commit()` is called in `ASDShellQ4::commitState()`). The element calls
`self._kinematics.commitState()` inside its own `_commit()` method, and the kinematics
internally updates its reference configuration.


**L2.** For Corotational, the element-attached frame rotates each iteration. Is this
rotation stored as trial state (reverted on Newton failure) and committed state (accepted
on convergence)? Or is the rotation recomputed from nodal positions each time (stateless)?

**L2 Answer [PRE-FILLED]:** The corotational transformation stores trial/committed state.
In ASDShellQ4, `m_transformation->commit()` and `m_transformation->revertToLastCommit()`
are called explicitly, confirming it manages internal state. The transformation updates
its deformed configuration via `update(UG)` during each iteration and commits/reverts
that state appropriately.


**L3.** Should the kinematics `copy()` method produce a fully independent deep copy
(like Material `getCopy()`)? This matters if elements are ever cloned (e.g., for
FiberSection or parallel material patterns).

**L3 Answer [PRE-FILLED]:** Yes. `copy()` must produce a fully independent deep copy with
the same parameters but independent state. This follows the same pattern as Material
`getCopy()` and is required for any scenario where elements or kinematics objects are
cloned.


---

### M. Naming and File Organization

**M1.** The kinematics files currently live in `model/element/kinematics/`. The CrdTransf
files live in `model/element/coordTransformation/`. In the unified system:
- Keep both directories?
- Merge CrdTransf into kinematics?
- Rename kinematics to coordTransformation?
- Something else?

**M1 Answer [PI]:** The `coordTransformation/` directory is merged into `kinematics/`. All
kinematics classes — `CrdTransf` and its subclasses, `ContinuumKinematics`,
`ShellKinematics`, `ContactKinematics` and their subclasses — live together under
`model/element/kinematics/`. Separating them would imply two independent systems; merging
reflects that they are all branches of the same `Kinematics` hierarchy.


**M2.** Naming convention for kinematics classes:
- Current: `LinearContinuumKinematics`, `LinearCrdTransf2d`
- Should beam kinematics be renamed (e.g., `LinearBeamKinematics2d`)?
- Or keep `CrdTransf` naming for beams?

**M2 Answer [PI]:** Beam kinematics retain the `CrdTransf` naming convention
(`LinearCrdTransf2d`, `CorotCrdTransf3d`, etc.) for consistency with OpenSees and the
future beam element ports.


**M3.** For continuum kinematics, should formulation-specific classes be named:
- `LinearContinuumKinematics`, `TLContinuumKinematics`, `ULContinuumKinematics`, `CorotContinuumKinematics`?
- Or shorter: `LinearKin`, `TotalLagrangianKin`, etc.?
- Or follow OpenSees-style naming?

**M3 Answer [PI]:** Continuum kinematics classes are named by formulation prefix + element
family + `Kinematics`: `LinearContinuumKinematics`, `TotalLagrangianContinuumKinematics`,
`UpdatedLagrangianContinuumKinematics`, `CorotContinuumKinematics`. The same prefix
convention (`Linear`, `TotalLagrangian`, `UpdatedLagrangian`, `Corot`) applies across all
kinematics branches for consistency.


---

### N. Benchmarks and Validation

**N1.** What benchmarks validate each formulation?
- Linear continuum: patch test (constant stress), Cook's membrane?
- TL: large-deformation cantilever, snap-through arch?
- UL: same problems as TL (should give identical results)?
- Corotational: Lee's frame, Williams' toggle frame?
- Others?

**N2.** What's the priority order for benchmarks? Which problems first?

**N1 & N2 Answer [PI]:** See `docs/oneFEM_Validation_Schedule.md` for the full benchmark
specification — complete model definitions, reference values, pass/fail criteria, modeling
tips, and pitfalls for all 8 benchmarks.

Summary of benchmarks in priority order:

| Priority | Benchmark | Formulation(s) | Gate |
|----------|-----------|----------------|------|
| 1 | Patch Test | Linear | G1 |
| 2 | Cook's Membrane | Linear | G2 |
| 3 | Simple Shear (kinematic unit test) | TL, UL | G3/G4 |
| 4 | Large-Deformation Cantilever Rollup | TL, UL | G5 |
| 5 | Snap-Through Arch | TL, UL, Corot | G6 |
| 6 | Lee's Frame | Corot (beams) | G7 |
| 7 | Williams Toggle Frame | Corot (beams) | G7 |
| 8 | Thick-Walled Cylinder | TL, UL | — |


**N3.** For truss with large deformation — is there a standard benchmark?
(e.g., two-bar truss snap-through, or cable under self-weight)

**N3 Answer [PI]:** Deferred with truss kinematics (F1). Williams toggle frame (Benchmark 7)
doubles as a truss snap-through benchmark once the truss kinematics refactor is complete.


---

### O. Integration with Existing Analysis Pipeline

**O1.** The current analysis pipeline (`Domain._assemble()`) scatters element stiffness
into global K. For large deformation, the element may return K_material + K_geometric.
Does the element combine these before returning from `getTangentStiff()`, or does the
Domain need to know about geometric stiffness?

**O1 Answer [PRE-FILLED]:** The element combines K_material + K_geometric and returns the
total tangent from `getTangentStiff()`. The Domain and analysis pipeline are completely
unaware of geometric stiffness — it is transparent. This already works this way for beam
elements: CrdTransf's `getGlobalStiffMatrix(kb, q)` returns the combined tangent including
geometric stiffness terms from PDelta or Corotational formulations.


**O2.** For Updated Lagrangian, the mesh effectively changes at each committed step
(reference config updates). Does `Domain._domain()` need to be re-called, or does the
element handle the remapping internally via the kinematics?

**O2 Answer [PRE-FILLED]:** No re-call of `Domain._domain()`. The element handles the
remapping internally via the kinematics `commitState()`, which updates the reference
configuration. `Domain._domain()` is called once before analysis for DOF numbering and
initial setup. The kinematics maintains its own reference configuration state.


**O3.** The Integrator forms the tangent via `formTangent()` which calls
`Domain._assemble()`. For geometric nonlinearity, does the integrator need to know
about the formulation, or is it completely transparent (the element returns the correct
tangent regardless of formulation)?

**O3 Answer [PRE-FILLED]:** Completely transparent. The integrator calls `Domain._assemble()`,
which calls `element.getTangentStiff()` on each element. The element delegates to its
kinematics strategy, which returns the appropriate tangent (including geometric stiffness
if applicable). The integrator, system solver, algorithm, and domain never need to know
what geometric nonlinearity formulation is in use. This is the entire point of the
strategy pattern.


---

### P. Priority and Phasing

**P1.** What is the implementation order across element types?
(e.g., continuum Linear first -> continuum TL -> beam refactor -> truss refactor -> shell -> contact)

**P1 Answer [PI]:** Implementation order:
1. Fix all broken nodes in one pass (Node22, Node33, and any others flagged broken or partial)
2. `ContinuumElement` base class + `Quad4` with `LinearContinuumKinematics`
3. `TotalLagrangianContinuumKinematics` on Quad4
4. `UpdatedLagrangianContinuumKinematics` on Quad4, validated together with TL
5. Beam refactor — merge `coordTransformation/` into `kinematics/`, no behavioral change;
   add unified kinematics API on top of existing OpenSees API; re-validate all existing beam benchmarks
6. `CorotContinuumKinematics` interface stub (NotImplementedError); snap-through and Lee's frame using beam/truss corot
7. `ZeroLengthContactASDimplex` in its own dedicated WP
8. `ASDShellQ4` in a later WP
9. Tri3 (deferred — plugs into framework established here)
10. 3D continuum (Brick8, Tet4) after 2D is solid
11. All remaining OpenSees ports as future WPs with source code provided at porting time


**P2.** What is the minimum viable deliverable? (e.g., "Quad4 with Linear kinematics
passing patch test" or "all 4 formulations on Quad4" or something else)

**P2 Answer [PI]:** The WP is complete when the unified kinematics framework is fully
implemented and validated, not when all elements are ported. Minimum viable deliverable:
all broken nodes fixed; the full `Kinematics` class hierarchy in place with all branches
designed (`CrdTransf`, `ContinuumKinematics`, `ShellKinematics`, `ContactKinematics`);
`ContinuumElement` base class implemented; `Quad4` working with all four formulations
(Linear, TL, UL, Corotational); all 8 validation benchmarks passing for Quad4; and
`coordTransformation/` merged into `kinematics/`. Any subsequent element (Tri3, Brick8,
shell, contact) simply plugs into the established framework in a future WP.


**P3.** Are there any dependencies on other WPs that affect the order?
(e.g., does FiberSection need to land before nonlinear beam kinematics?)

**P3 Answer [PI]:** This WP is self-contained. No other WP needs to complete before it,
and no other WP is blocked by it. The one soft dependency is arc-length control in the
analysis pipeline, needed for full post-snap-through path tracing. Without it, the
snap-through benchmark (Benchmark 5) is considered **passed** when: (a) the limit load
P_cr is within 5% of the reference value under displacement control, and (b) the solver
diverges gracefully beyond the limit point rather than producing a false equilibrium.
Recovery of the full post-snap unstable branch requires arc-length control and is deferred
to the analysis pipeline WP. FiberSection,
forceBeamColumn, shell, and contact all depend on interfaces designed in this WP but
implemented in later WPs; they impose no constraints on execution order here.


**P4.** Should the plan include refactoring existing working elements (Truss,
ElasticBeamColumn) to use the unified kinematics, or should existing elements be left
as-is and only new elements use the system?

**P4 Answer [PI]:** Existing elements (Truss, ElasticBeamColumn) are refactored in this WP
to add the unified kinematics API on top of their existing OpenSees-compatible API. The
OpenSees API is preserved — existing method signatures, behavior, and benchmarks remain
intact. The new kinematics methods (formulation-specific internal functions for strain,
B matrix, geometric stiffness, Jacobian) are added as an extension layer. Existing tests
still pass, and the elements simultaneously gain full four-formulation support.


---

### Q. Lessons from OpenSees Source Code (ZeroLengthContactASDimplex & ASDShellQ4)

Both source files have been read in full. The following questions arise from concrete
patterns observed in those implementations.

**From ZeroLengthContactASDimplex.cpp (your own element):**

The contact element uses an orientation vector (`Xorient`) to build a local coordinate
system (rotation matrix T), then computes strain as displacement jump in local coordinates:
`eps = B * T * u_global`, where B is the trivial `[-I, I]` matrix. Stress is computed
internally (not via a separate Material object) using a Coulomb friction model with
IMPL-EX integration. The tangent is either analytical (implex) or numerical perturbation
(implicit).

**Q1.** In the oneFEM Python version, should the contact element follow the same pattern
(internal constitutive model, no separate Material), or should the friction law be
abstracted as a Material (or a new ContactMaterial) so it can be swapped?

**Q1 Answer [PRE-FILLED]:** Internal constitutive model, matching the C++ source (one-to-one
port). The Coulomb friction law with IMPL-EX is tightly coupled to the element's
numerical integration scheme. Abstracting it as a Material would add complexity without
clear benefit, since the friction model is not interchangeable in the same way structural
materials are.


**Q2.** The C++ contact element supports both implicit and IMPL-EX integration
(`-intType` flag). IMPL-EX extrapolates internal variables from committed history,
giving a symmetric tangent. Should the oneFEM version support IMPL-EX from day one,
or start implicit-only?

**Q2 Answer [PRE-FILLED]:** Both from day one (one-to-one port). IMPL-EX is a core feature
of this element — it provides a symmetric tangent that improves convergence. The C++ source
already has both paths cleanly separated in `updateInternal(do_implex, do_tangent)`.


**Q3.** The contact element uses `Xorient` (user-supplied normal direction) to build its
local coordinate system, not node geometry (since nodes are coincident). In the unified
kinematics, does this orientation vector belong to the kinematics object or the element?

**Q3 Answer [PI]:** The orientation vector belongs to the kinematics object. It is a
geometric property that defines the local coordinate system — exactly what kinematics
objects are responsible for. The `ContactKinematics` base class holds `Xorient`, builds
the rotation matrix T from it in `initialize()`, and manages its update (or non-update
for the fixed-orientation initial port) across formulations. The element receives the
rotation matrix from the kinematics and never owns the orientation directly.


**From ASDShellQ4.cpp:**

The shell element uses a **transformation object** (`m_transformation`) that can be
either `ASDShellQ4Transformation` (linear) or `ASDShellQ4CorotationalTransformation`
(corotational), selected by a boolean flag in the constructor. The transformation handles:
- `createReferenceCoordinateSystem()` — initial local axes
- `createLocalCoordinateSystem(UG)` — current local axes (deformed for corot)
- `calculateLocalDisplacements(local_cs, UG, UL)` — global -> local displacement mapping
- `transformToGlobal(local_cs, UG, UL, LHS, RHS, do_lhs)` — local K/F -> global K/F
- `commit()`, `revertToLastCommit()`, `revertToStart()` — state management
- `update(UG)` — update deformed configuration

The element itself doesn't know if it's linear or corotational — it calls the same
transformation interface either way. The B matrix computation (membrane + bending + MITC4
shear) is done in the reference coordinate system, and the transformation handles the
rest.

**Q5.** This is essentially the same strategy pattern as CrdTransf for beams, but for
shells. In your unified kinematics, should the shell transformation follow the same
`ASDShellQ4Transformation` / `ASDShellQ4CorotationalTransformation` split? Or should
it use the generic `Kinematics` base with shell-specific methods?

**Q5 Answer [PRE-FILLED]:** Shell-specific kinematics branch under the `Kinematics` base,
following the ASDShellQ4 transformation pattern. A `ShellKinematics(Kinematics)` base
with `LinearShellKinematics`, `CorotShellKinematics`, `TotalLagrangianShellKinematics`,
and `UpdatedLagrangianShellKinematics` concrete implementations. This mirrors how
`CrdTransf(Kinematics)` works for beams. The shell transformation has a fundamentally
different interface than beam CrdTransf (4 nodes vs 2, local coordinate system vs basic
frame, etc.).


**Q6.** ASDShellQ4 uses `SectionForceDeformation` (8 generalized strain components:
3 membrane + 3 bending + 2 shear). The section handles through-thickness integration.
Should oneFEM shells also use `SectionForceDeformation`, or should they use nDMaterial
directly with explicit through-thickness integration in the element?

**Q6 Answer [PRE-FILLED]:** `SectionForceDeformation` (one-to-one port). The shell section
receives 8 generalized strains and returns 8 generalized stresses + 8x8 tangent. Through-
thickness integration is encapsulated in the section.


**Q7.** The shell element computes everything in one method (`calculateAll`) with
bitflags (OPT_UPDATE, OPT_LHS, OPT_RHS) to avoid redundant computation. This is
efficient but monolithic. Should oneFEM follow this pattern, or keep separate
`_update()`, `getTangentStiff()`, `getResistingForce()` methods as in the current
Element interface?

**Q7 Answer [PRE-FILLED]:** Keep separate `_update()`, `getTangentStiff()`,
`getResistingForce()` methods. This maintains consistency with the existing Element
interface that all other elements use, and that the Domain/Analysis pipeline expects.
Internally, the shell element can cache computed results to avoid redundant work (e.g.,
cache the B matrix and section tangent from `_update()` for use in `getTangentStiff()`).


**Q8.** ASDShellQ4 handles drilling DOF with the Hughes-Brezzi penalty approach
(`Bd` matrix linking in-plane displacement gradient to drilling rotation). It supports
elastic and nonlinear drilling modes. What's your plan for drilling DOFs?

**Q8 Answer [PRE-FILLED]:** Hughes-Brezzi drilling DOF penalty, matching ASDShellQ4 (one-to-
one port). Both elastic and nonlinear drilling modes. The drilling stiffness is computed
from the average in-plane shear modulus scaled by 0.233 (roto-distortional modulus).
Drilling B matrix uses weighted sum of reduced-integrated (center) and fully-integrated
versions to suppress spurious zero-energy modes while avoiding over-stiffening.


**Q9.** The shell uses AGQI (Area-coordinate based Generalized Quadrilateral with Internal
DOFs) for enhanced membrane behavior and MITC4 for shear locking elimination. These are
quite specific formulations. Should oneFEM's shell element replicate these exactly, use
simpler alternatives (e.g., standard Q4 membrane + assumed strain shear), or match the
ASDShellQ4 formulation?

**Q9 Answer [PRE-FILLED]:** Match ASDShellQ4 exactly (one-to-one port). AGQI for enhanced
membrane with internal DOFs (EAS, with static condensation). MITC4 for transverse shear
locking elimination. These are proven formulations that handle distorted and warped
geometries well.


**Q10.** The shell uses a section orientation angle (`m_angle`) to rotate strains/stresses
between the element local system and the material/section system. This handles fiber-
reinforced or orthotropic materials where the material axes differ from element axes.
Should oneFEM support this from the start?

**Q10 Answer [PRE-FILLED]:** Yes, from the start (one-to-one port). The section orientation
angle is implemented in ASDShellQ4 with rotation matrices for generalized strains (`Re`)
and generalized stresses (`Rs`). It is computed once in `setDomain()` from the user-
supplied local x-axis direction (`-local` flag) relative to the reference coordinate
system's x-axis.


---

### R. Node-DOF Map

The following table maps node types to their DOF definitions and which elements use them.

| Node Type | nDim | nDOF | DOF Definitions | Used By (Elements) | Status |
|-----------|------|------|-----------------|---------------------|--------|
| Node22 | 2 | 2 | ux, uy | Truss2D, Quad4, Tri3 | ⚠️ Fix in this WP |
| Node23 | 2 | 3 | ux, uy, θz | Beam2D, ZeroLength2D, Contact2D | ✅ Working |
| Node24 | 2 | 4 | ux, uy, θz, p | 2D Cosserat u-r-p formulation | ⚠️ Fix in this WP |
| Node33 | 3 | 3 | ux, uy, uz | Truss3D, Brick8, Tet4, Contact3D | ⚠️ Fix in this WP |
| Node34 | 3 | 4 | ux, uy, uz, p | 3D Cauchy u-p mixed formulation | ⚠️ Fix in this WP |
| Node36 | 3 | 6 | ux, uy, uz, θx, θy, θz | Beam3D, ShellQ4, ZeroLength3D, Contact3D | ✅ Working |
| Node37 | 3 | 7 | ux, uy, uz, θx, θy, θz, p | 3D Cosserat u-r-p formulation | ⚠️ Fix in this WP |

**R1.** What is the 4th DOF of Node24?

**R1 Answer [PI]:** Node24 (2D, 4-DOF): ux, uy, θz, p — 2D Cosserat with pressure
(u-r-p formulation).


**R2.** What is the 4th DOF of Node34?

**R2 Answer [PI]:** Node34 (3D, 4-DOF): ux, uy, uz, p — 3D Cauchy with pressure
(u-p mixed formulation). Node37 (3D, 7-DOF): ux, uy, uz, θx, θy, θz, p — 3D Cosserat
with pressure (u-r-p formulation). Node24, Node34, and Node37 are future-use node types
for mixed formulations and are not part of the current work package scope beyond the
initial fix pass.


**R3.** Is the DOF ordering above correct for all nodes? In particular, does Node36 use
(ux, uy, uz, θx, θy, θz) or some other ordering?

**R3 Answer [PRE-FILLED]:** Yes. This is the standard OpenSees ordering: 3 translational DOFs
first, then 3 rotational DOFs. The existing ElasticBeamColumn3d and CrdTransf3d confirm this.


**R4.** Node22 and Node33 are flagged as broken. They are required for continuum elements
(Quad4 needs Node22, Brick8 needs Node33). Fix priority?

**R4 Answer [PI]:** All broken nodes (Node22, Node33, Node24, Node34, Node37) are fixed
in one pass at the start of this work package. See E2 answer.


---

### S. Detailed Technical Questions — Flow, Coding, Performance

**S1. Element Integration Loop Flow:**
For a continuum element (e.g., Quad4) with the unified kinematics, what should the
`_update()` / `getTangentStiff()` / `getResistingForce()` flow look like?

**S1 Answer [PI]:** The correct flow uses native oneFEM math objects throughout. The element
drives the loop; the kinematics is a pure geometry servant. The kinematics caches B, detJ,
and strain internally when `update(gp, dN_dX, u_e)` is called — subsequent calls to
`getBMatrix(gp)`, `getStrain(gp)`, `getDetJ(gp)` are free cache lookups. The element uses
the isoparametric utility module to map `dN_dxi → dN_dX` and owns Jacobian computation.

```python
# In ContinuumElement._update():
if self._kinematics.formulation != 'linear':
    self._extractDisplacements(self._u_e)  # fill pre-allocated Vector in-place
for gp in range(self._nGP):
    dN_dX = self._iso.computePhysicalDerivatives(gp, self._X_nodes, self._u_e)
    self._kinematics.update(gp, dN_dX, self._u_e)
    strain = self._kinematics.getStrain(gp)           # CTensor (2nd, COV)
    self._materials[gp]._setTrialStrain(strain)

# In ContinuumElement.getTangentStiff():
K = Matrix(self._nDOF, self._nDOF)
for gp in range(self._nGP):
    B     = self._kinematics.getBMatrix(gp)            # Matrix
    C     = self._materials[gp].getTangent()           # CTensor (4th, CONTR)
    detJ  = self._kinematics.getDetJ(gp)               # scalar
    K    += B.T @ C @ B * detJ * self._w[gp]          # Matrix @ CTensor @ Matrix
    K    += self._kinematics.getGeometricStiffness(
                gp, self._materials[gp].getStress())   # Matrix
return K

# In ContinuumElement.getResistingForce():
f = Vector(self._nDOF)
for gp in range(self._nGP):
    B    = self._kinematics.getBMatrix(gp)             # Matrix (cached)
    sig  = self._materials[gp].getStress()             # CTensor (2nd, CONTR)
    detJ = self._kinematics.getDetJ(gp)                # scalar (cached)
    f   += B.T @ sig * detJ * self._w[gp]             # Matrix @ CTensor → Vector
return f
```


**S2. Kinematics Per Gauss Point vs Per Element:**
In the proposed flow above, there's one kinematics object per Gauss point (`self._kinematics[gp]`).
Alternative: one kinematics per element that takes the Gauss point index as argument.

- Option A: Array of kinematics objects (one per GP) — each has independent state
- Option B: Single kinematics per element, called with GP index

**S2 Answer [PI]:** One kinematics object per element (Option B), architecturally consistent
with all other element types. The kinematics internally manages arrays of per-GP state
pre-allocated in `initialize()`. The element passes the GP index when calling kinematics
methods. See A3 answer.


**S3. Argument Passing to Kinematics:**
What arguments does `kinematics.update()` need for each element type?

**S3 Answer [PI]:** The element owns `u_e` and passes it explicitly. The kinematics never
holds node references. For continuum: `kinematics.update(gp, dN_dX, u_e)`. For beams:
`crdTransf.update()` reads node positions internally (from node references set in
`initialize(node_i, node_j)`) — this existing pattern is preserved. Allocation is
formulation-aware: `u_e` is only allocated for nonlinear formulations. See B3 answer.


**S4. CTensor Representation at the B^T C B Boundary:**

**S4 Answer [PRE-FILLED + PI]:** The B matrix produces engineering strain components in
standard Voigt ordering: [eps_11, eps_22, eps_33, gamma_12, gamma_23, gamma_13] where
gamma = 2*eps (engineering shear). This is the standard FEM convention. When this strain
vector is stored as CTensor(COV), the COV representation factors (1, 1, 1, 0.5, 0.5, 0.5)
convert engineering shear back to tensorial shear for the double-dot product.

All operations between native math objects (`Matrix @ CTensor @ Matrix`) handle Voigt
factor correctness internally — no manual factor management in element code. See B2 answer
for the full math module cross-type operation specification.


**S5. Performance — CTensor vs numpy in Hot Loop:**

**S5 Answer [PI]:** All element and kinematics code uses oneFEM native math objects
exclusively. The math module wraps numpy internally for all heavy computation. Performance
is preserved because native object operations delegate to numpy; the abstraction overhead
is negligible. A backend change (numpy → JAX, GPU, sparse) requires editing only the math
module. See B2 answer for the complete data flow and cross-type operation specification.


**S6. Object Hierarchy — Kinematics Inheritance:**

**S6 Answer [PI]:**

```
Kinematics  (base: formulation tag, initialize, update, commitState,
             revertToLastCommit, copy — shared methods here, never duplicated)
│
├── CrdTransf  (beams: initialize(node_i, node_j), update(),
│              getBasicTrialDisp, getGlobalStiffMatrix, getGlobalResistingForce)
│   ├── LinearCrdTransf2d / LinearCrdTransf3d
│   ├── PDeltaCrdTransf2d / PDeltaCrdTransf3d   ← beam-only, 5th option, low priority
│   ├── CorotCrdTransf2d / CorotCrdTransf3d
│   ├── TotalLagrangianCrdTransf2d / TotalLagrangianCrdTransf3d
│   └── UpdatedLagrangianCrdTransf2d / UpdatedLagrangianCrdTransf3d
│
├── ContinuumKinematics  (solids: update(gp, dN_dX, u_e), getStrain(gp),
│                         getBMatrix(gp), getDetJ(gp), getGeometricStiffness(gp, stress))
│   ├── LinearContinuumKinematics
│   ├── CorotContinuumKinematics               ← EICR, Felippa & Haugen (2005)
│   ├── TotalLagrangianContinuumKinematics
│   └── UpdatedLagrangianContinuumKinematics
│
├── ShellKinematics  (shells — interface designed now, implemented at ASDShellQ4 port time)
│   ├── LinearShellKinematics
│   ├── CorotShellKinematics                   ← EICR, matches ASDShellQ4CorotTransformation
│   ├── TotalLagrangianShellKinematics
│   └── UpdatedLagrangianShellKinematics
│
└── ContactKinematics  (contact + zero-length: orientation vector → rotation matrix T,
                        B = [-I, I], manages local frame rotation across formulations)
    ├── LinearContactKinematics
    ├── CorotContactKinematics
    ├── TotalLagrangianContactKinematics
    └── UpdatedLagrangianContactKinematics
```

All leaf classes inherit shared behaviour from their branch base, which inherits from
`Kinematics`. Shared logic is never duplicated — it lives at the nearest common ancestor.


**S7. Kinematics Factory / Default Construction:**

**S7 Answer [PI]:** Option A — each element creates its own default kinematics internally.
This is the correct choice because different elements in the same domain can use different
formulations simultaneously (some elements Linear, others TL, others UL). Each element
owns its kinematics object independently, injected at construction. When no kinematics
argument is provided, the element instantiates its own Linear default. Mixed-formulation
domains are trivial — the user simply passes different kinematics objects to different
elements at construction time, with no global state or factory coordination required.

```python
class Quad4(ContinuumElement):
    def __init__(self, tag, nodes, material, kinematics=None):
        if kinematics is None:
            kinematics = LinearContinuumKinematics()
        super().__init__(tag, nodes, material, kinematics)
```


**S8. Kinematics Initialization Timing:**

**S8 Answer [PRE-FILLED]:** In `Element._domain()`, matching the CrdTransf pattern. The
kinematics receives node coordinates and computes initial geometry (Jacobian for reference
config, initial element dimensions, etc.). For UL, this is the initial reference
configuration that gets updated at `commitState()`. For continuum elements, pre-allocation
of per-GP state arrays (`_B`, `_strain`, `_detJ`, `_u_e` if nonlinear) also happens here.


**S9. Geometric Stiffness Return Type:**

**S9 Answer [PI]:** `Matrix` — consistent with the math module philosophy (S5/B2). All
kinematics methods return native oneFEM math objects. `getGeometricStiffness(gp, stress)`
returns a `Matrix` of size (nDOF_total × nDOF_total) that the element adds to the material
stiffness `Matrix` in the integration loop.


**S10. Element Stiffness Return Type:**

**S10 Answer [PI]:** `Matrix` — already a native oneFEM math object, consistent with the
math module philosophy. No change needed.


**S11. Memory Layout — Pre-allocation vs Per-Call Allocation:**

**S11 Answer [PRE-FILLED]:** Yes, pre-allocate in `_domain()`. This avoids repeated
allocation in the inner loop. For a mesh with thousands of elements, each with 4 Gauss
points, allocation overhead adds up. The CTensor class already uses this pattern (fixed
81-element flat list, pre-allocated at construction).


---

### T. Reference Library

| Topic | Reference Paper/Book | Example Code | Status |
|-------|---------------------|--------------|--------|
| **Voigt representation system** | Helnwein (2001) CMAME 190(22) | `docs/ctensor.py` | ✅ Have |
| **Linear FEM (continuum)** | Hughes (2000), Bathe (1996) Ch. 4-5 | — | ✅ Standard textbook |
| **Total Lagrangian** | Bathe (1996) Ch. 6; Bonet & Wood (2008) Ch. 7 | OpenSees `TotalLagrangianFD20NodeBrick.cpp` | ✅ Have |
| **Updated Lagrangian** | Bathe (1996) Ch. 6 (primary); Bonet & Wood (2008) | OpenSees UL source | ✅ Have |
| **Corotational (beams)** | Crisfield (1991) Vol. 1 | Already implemented (CrdTransf) | ✅ Have |
| **Corotational (shells)** | EICR — Felippa & Haugen (2005) CMAME 194 | `OPENSEES SRC/.../ASDShellQ4.cpp` | ✅ Have |
| **Corotational (continuum)** | Felippa & Haugen (2005) CMAME 194 | — | ✅ Same EICR framework |
| **MITC4 shell** | Bathe & Dvorkin (1986) | `OPENSEES SRC/.../ASDShellQ4.cpp` | ✅ Have |
| **AGQI membrane** | Chen et al. | `OPENSEES SRC/.../ASDShellQ4.cpp` | ✅ Have |
| **Hughes-Brezzi drilling** | Hughes & Brezzi (1989) | `OPENSEES SRC/.../ASDShellQ4.cpp` | ✅ Have |
| **SectionForceDeformation** | OpenSees architecture | OpenSees `SectionForceDeformation.h` | ✅ Have |
| **Coulomb friction + penalty** | — | `OPENSEES SRC/.../ZeroLengthContactASDimplex.cpp` | ✅ Have |
| **IMPL-EX integration** | Oliver et al. (2008) CMAME | `OPENSEES SRC/.../ZeroLengthContactASDimplex.cpp` | ✅ Have |
| **Patch test** | MacNeal & Harder (1985) Comput. Struct. 20 | `docs/oneFEM_Validation_Schedule.md` | ✅ Have |
| **Cook's membrane** | Cook (1974) J. Struct. Div. ASCE | `docs/oneFEM_Validation_Schedule.md` | ✅ Have |
| **Snap-through benchmarks** | Crisfield (1991) Vol. 1 Ch. 9 | `docs/oneFEM_Validation_Schedule.md` | ✅ Have |
| **Isoparametric shape functions** | Hughes (2000), Bathe (1996) | — | ✅ Standard textbook |
| **Gauss quadrature rules** | Hughes (2000), Bathe (1996) | — | ✅ Standard textbook |

**T1 Answer [PI]:** Primary reference for both TL and UL: Bathe, *Finite Element Procedures*
(1996/2006), Ch. 6 — explicit B_L + B_NL decomposition for TL and incremental UL
updated-reference flow. Secondary reference for clean kinematic notation: Bonet & Wood,
*Nonlinear Continuum Mechanics for Finite Element Analysis* (2nd ed. 2008), Ch. 3 and 7.
Reference implementation: OpenSees C++ source (`TotalLagrangianFD20NodeBrick.cpp`) for
one-to-one portable pseudocode.

**T2 Answer [PI]:** Corotational continuum kinematics follows Felippa's Element-Independent
Corotational Reference (EICR) framework — the same theoretical foundation used for the
shell corotational transformation in `ASDShellQ4CorotationalTransformation`. Primary
reference: Felippa & Haugen (2005), *A unified formulation of small-strain corotational
finite elements*, CMAME 194. This ensures theoretical consistency across shell and
continuum corotational formulations.

**T3 Answer [PI]:** Isoparametric shape functions and Gauss quadrature rules follow Hughes,
*The Finite Element Method* (2000) and Bathe, *Finite Element Procedures* (1996/2006).
Both are already referenced for TL/UL — using the same sources throughout keeps the
implementation consistent.


---

## Summary of Decisions

*(Condensed list of every binding design decision for quick reference during implementation.)*

1. Every element type supports all four formulations: Linear, Corotational, TL, UL.
2. PDelta is beam-only, 5th option, low priority, not extended to other elements.
3. `Kinematics` base contains all shared methods. No duplication across siblings.
4. One kinematics object per element, always. Per-GP state is internal to the kinematics.
5. The element owns `u_e`. Pre-allocated only for nonlinear formulations. Filled in-place.
6. The element drives the integration loop. Kinematics is a pure geometry servant.
7. `ContinuumElement` base class owns the loop structure. Subclasses extend it.
8. The kinematics caches B, detJ, strain after `update(gp, ...)`. Subsequent calls are free.
9. The element computes the Jacobian. Kinematics exposes `formulation` tag to select config.
10. All element and kinematics code uses native math objects (CTensor, Matrix, Vector). No raw numpy. Add this rule to `CLAUDE.md` and enforce with a CI grep: `grep -r "import numpy\|np\." src/oneFEM/model/element/ --include="*.py"` must return nothing.
11. The math module defines all cross-type operations. Backend changes require only math module edits.
12. All broken nodes fixed in one pass at WP start. No deferred node debt.
13. `coordTransformation/` merged into `kinematics/`. One directory, one hierarchy.
14. Beam naming: `CrdTransf` retained. Continuum naming: `TotalLagrangianContinuumKinematics`, etc.
15. UL reference coordinates stored inside kinematics, updated at `commitState()`.
16. Corotational frame stored as trial/committed state. Updated via `update()`, committed/reverted explicitly.
17. `copy()` produces fully independent deep copy (same as Material `getCopy()`).
18. Geometric stiffness computed by kinematics, returned as `Matrix`. Transparent to Domain/Analysis.
19. Orientation vector for contact/zero-length belongs to `ContactKinematics`.
20. Shell interface designed now, implemented at ASDShellQ4 port time.
21. ZeroLength and contact use `ContactKinematics` as shared kinematics base.
22. ZeroLength uses kinematics — not an exception. Local frame rotates with deformation.
23. Existing elements (Truss, ElasticBeamColumn) gain new kinematics API on top of OpenSees API.
24. Minimum viable deliverable: Quad4 with all 4 formulations, all 8 benchmarks passing.
25. No WP dependencies. Soft dependency on arc-length control (displacement control acceptable substitute).
26. Corotational continuum follows Felippa EICR framework, consistent with shell corotational.
27. TL/UL primary reference: Bathe Ch. 6. Secondary: Bonet & Wood (2008).
28. Shape functions: isoparametric utility module (pure functions, no state). References: Hughes, Bathe.


---

## Additional Design Decisions

### X1. Math Module Interface Specification

The following cross-type operations are required in the math module. All must be implemented
before element integration loop code is written. All operations return native oneFEM objects.

| Operation | Input Types | Output Type | Usage |
|-----------|-------------|-------------|-------|
| `C ^ eps` | CTensor(4th, CONTR) ^ CTensor(2nd, COV) | CTensor(2nd, CONTR) | Material response (already exists) |
| `C.to_matrix()` | CTensor(4th) | Matrix (nVoigt × nVoigt) | Integration loop |
| `sigma.to_vector()` | CTensor(2nd) | Vector (nVoigt) | Force assembly |
| `B.T @ C @ B` | Matrix.T, CTensor(4th), Matrix | Matrix | Stiffness integration — **see note below** |
| `B.T @ sigma` | Matrix.T, CTensor(2nd) | Vector | Force integration |
| `Matrix + Matrix` | Matrix, Matrix | Matrix | Stiffness accumulation |
| `Vector + Vector` | Vector, Vector | Vector | Force accumulation |
| `Matrix * scalar` | Matrix, float | Matrix | detJ * w weighting |
| `Vector * scalar` | Vector, float | Vector | detJ * w weighting |

**Rule:** Operations between native objects always return native objects. If an operation
between two native types is needed and not in this table, add it to the math module before
using it in element code — never work around it with raw numpy.

> **Critical implementation note — `B.T @ C @ B`:** The existing CTensor `__xor__`
> handles `C:ε → σ` (4th-order contracted with a 2nd-order Voigt vector). The `B.T @ C @ B`
> stiffness integral requires C to act as a rank-2 Voigt matrix (nVoigt × nVoigt), which is
> a different contraction path. **Before Phase 2 coding begins**, verify that
> `CTensor.__matmul__(Matrix)` is implemented and returns the correct `C_voigt @ B` result —
> or explicitly call `C.to_matrix()` first and perform `B.T @ C.to_matrix() @ B` with
> standard Matrix multiply. A missing or incorrect `__matmul__` will silently produce a
> factor-of-2 error on shear stiffness terms that Cook's membrane will catch but the patch
> test may not. Confirm by checking pure-shear stiffness = G on a unit element before
> running any benchmark.


### X1b. Math Module API Verification (complete before Phase 2)

Before writing a single line of element integration code, run the following verification
script against the live math module. Phase 2 must not start until all assertions pass.

```python
import numpy as np
from oneFEM._systools.data import CTensor, Matrix, Vector

E, nu = 1.0, 0.25
G = E / (2*(1+nu))

# Build plane-stress elastic tangent as CTensor
C = CTensor(...)  # ElasticIsotropic PlaneStress tangent

# 1. to_matrix() returns correct 3×3 Voigt matrix
C_mat = C.to_matrix()          # must be Matrix, shape (3,3)
assert isinstance(C_mat, Matrix)
assert abs(float(C_mat[0,0]) - E/(1-nu**2)) < 1e-12,   "C_1111 wrong"
assert abs(float(C_mat[2,2]) - G) < 1e-12,              "C_1212 (shear) wrong — factor-of-2 error"

# 2. to_vector() on a stress CTensor
sig_data = [1.0, 0.5, 0.25]  # sigma_11, sigma_22, sigma_12
sig = CTensor(sig_data, ...)
sig_vec = sig.to_vector()     # must be Vector, length 3
assert isinstance(sig_vec, Vector)

# 3. Matrix @ CTensor(4th) @ Matrix — stiffness contraction
B = Matrix(...)               # shape (3, 8) example
K = B.T @ C @ B               # must be Matrix, shape (8, 8)
assert isinstance(K, Matrix)

# 4. Matrix @ CTensor(2nd) — force contraction
f = B.T @ sig                 # must be Vector, length 8
assert isinstance(f, Vector)

# 5. Shear stiffness sanity check on unit square element (E=1, nu=0.25, t=1)
#    Apply pure shear: K_12_12 should equal G*t = 0.4 for unit element
# ... (implement with actual Quad4 Linear stiffness assembly)
```

**If any assertion fails:** fix the math module before proceeding. Do not work around a
failing assertion with raw numpy — that defeats the entire abstraction. The most likely
failure is `C_1212 ≠ G`, caused by an inconsistent Voigt shear factor in `to_matrix()`.


### X2. Final Directory and File Tree

After the refactor, `model/element/` will have the following kinematics structure:

```
model/element/
│
├── kinematics/                          ← merged from coordTransformation/ + kinematics/
│   ├── __init__.py
│   ├── base.py                          ← Kinematics base class
│   │
│   ├── crdTransf/                       ← beam kinematics (renamed from coordTransformation/)
│   │   ├── __init__.py
│   │   ├── base.py                      ← CrdTransf(Kinematics)
│   │   ├── linear_2d.py                 ← LinearCrdTransf2d
│   │   ├── linear_3d.py                 ← LinearCrdTransf3d
│   │   ├── pdelta_2d.py                 ← PDeltaCrdTransf2d
│   │   ├── pdelta_3d.py                 ← PDeltaCrdTransf3d
│   │   ├── corot_2d.py                  ← CorotCrdTransf2d
│   │   ├── corot_3d.py                  ← CorotCrdTransf3d
│   │   ├── total_lagrangian_2d.py       ← TotalLagrangianCrdTransf2d
│   │   ├── total_lagrangian_3d.py       ← TotalLagrangianCrdTransf3d
│   │   ├── updated_lagrangian_2d.py     ← UpdatedLagrangianCrdTransf2d
│   │   └── updated_lagrangian_3d.py     ← UpdatedLagrangianCrdTransf3d
│   │
│   ├── continuum/                       ← solid element kinematics
│   │   ├── __init__.py
│   │   ├── base.py                      ← ContinuumKinematics(Kinematics)
│   │   ├── linear.py                    ← LinearContinuumKinematics
│   │   ├── corot.py                     ← CorotContinuumKinematics (EICR)
│   │   ├── total_lagrangian.py          ← TotalLagrangianContinuumKinematics
│   │   └── updated_lagrangian.py        ← UpdatedLagrangianContinuumKinematics
│   │
│   ├── shell/                           ← shell kinematics (interface only in this WP)
│   │   ├── __init__.py
│   │   ├── base.py                      ← ShellKinematics(Kinematics)
│   │   ├── linear.py                    ← LinearShellKinematics (stub)
│   │   ├── corot.py                     ← CorotShellKinematics (stub)
│   │   ├── total_lagrangian.py          ← TotalLagrangianShellKinematics (stub)
│   │   └── updated_lagrangian.py        ← UpdatedLagrangianShellKinematics (stub)
│   │
│   └── contact/                         ← contact + zero-length kinematics
│       ├── __init__.py
│       ├── base.py                      ← ContactKinematics(Kinematics)
│       ├── linear.py                    ← LinearContactKinematics
│       ├── corot.py                     ← CorotContactKinematics
│       ├── total_lagrangian.py          ← TotalLagrangianContactKinematics
│       └── updated_lagrangian.py        ← UpdatedLagrangianContactKinematics
│
├── continuum/                           ← solid elements
│   ├── __init__.py
│   ├── base.py                          ← ContinuumElement(Element) — integration loop lives here
│   ├── isoparametric.py                 ← shape functions + Gauss rules (pure functions, no state)
│   └── quad4.py                         ← Quad4(ContinuumElement) — only continuum element in this WP
│   # tri3.py, brick8.py, tet4.py — deferred to future porting WPs
│
├── beam/                                ← unchanged
├── truss/                               ← unchanged (kinematics refactor deferred)
├── shell/                               ← stub until ASDShellQ4 WP
└── zeroLength/                          ← stub until contact WP
```


### X3. New Element Implementor Checklist

When porting a new element (e.g., Tri3, Brick8, ASDShellQ4) into the unified framework,
the following must be implemented. Items marked *(inherited)* require no code if the base
class already provides them.

**In the element class:**

- [ ] Constructor accepts `kinematics=None`; defaults to the appropriate Linear kinematics
- [ ] `_domain()`: calls `self._kinematics.initialize(...)`, pre-allocates `u_e` if nonlinear,
      pre-allocates `_K`, `_f` arrays
- [ ] `_update()`: extracts `u_e`, calls `kinematics.update(gp, dN_dX, u_e)` per GP,
      calls `material._setTrialStrain(strain)` per GP
- [ ] `getTangentStiff()`: integration loop — `B.T @ C @ B * detJ * w` + geometric stiffness,
      returns `Matrix`
- [ ] `getResistingForce()`: integration loop — `B.T @ sigma * detJ * w`, returns `Vector`
- [ ] `_commit()`: calls `self._kinematics.commitState()` + material commits *(inherited if
      ContinuumElement base handles it)*
- [ ] `_revertToLastCommit()`: calls `self._kinematics.revertToLastCommit()` + material reverts
      *(inherited)*
- [ ] `getMass()`: element-specific mass matrix computation *(not in kinematics)*

**In the isoparametric utility module (if new topology):**

- [ ] `shape_functions(xi, eta)` — returns N array
- [ ] `shape_derivatives(xi, eta)` — returns dN/dxi array
- [ ] `gauss_points()` — returns (xi, eta, w) tuples for the default integration scheme


### X4. Kinematics Implementor Checklist

When implementing a new kinematics class (e.g., `TotalLagrangianContinuumKinematics`),
the following must be implemented. Items marked *(inherited)* require no code.

**Always inherited from `Kinematics` base:**
- `formulation` tag property *(inherited)*
- `copy()` — deep copy *(inherited if base provides it; otherwise implement)*
- `commitState()` — for stateless formulations (Linear) this is a no-op *(inherited)*
- `revertToLastCommit()` — no-op for stateless *(inherited)*

**Must implement in the branch base (e.g., `ContinuumKinematics`):**
- `initialize(gp_data, X_nodes, formulation)` — pre-allocate per-GP arrays, compute
  reference Jacobians, set up initial state
- `update(gp, dN_dX, u_e)` — compute and cache B, strain, detJ for this GP
- `getStrain(gp)` → CTensor (2nd, COV)
- `getBMatrix(gp)` → Matrix
- `getDetJ(gp)` → scalar
- `getGeometricStiffness(gp, stress)` → Matrix (return zero Matrix for Linear)

**Must override in TL/UL leaf classes:**
- `update(gp, dN_dX, u_e)` — compute F, update B_NL, update strain measure
- `getGeometricStiffness(gp, stress)` — compute K_sigma from stress and dN_dX
- `commitState()` — for UL: update reference coordinates to current deformed
- `revertToLastCommit()` — for UL: restore reference coordinates to last committed

> **Critical UL implementation rule — reference coordinates during Newton iteration:**
> During Newton-Raphson iteration, nodal coordinates in the Domain are updated in the
> trial state (current = reference + u_trial). The UL kinematics object must maintain
> its **own internal copy** of the reference coordinates (last committed configuration),
> updated only at `commitState()` — never during `update()`. When `update(gp, dN_dX, u_e)`
> is called during iteration, `dN_dX` must be computed using the internally stored reference
> coordinates, not the current trial node positions from the Domain. If `computePhysicalDerivatives`
> uses live node coordinates instead of the kinematics' committed reference copy, the UL
> reference configuration silently advances during iteration, producing incorrect results that
> may still appear to converge. Add an assertion in `update()` that the reference coordinates
> have not changed since the last `commitState()` during debugging.

**Must override in Corot leaf class:**
- `update(gp, dN_dX, u_e)` — polar decompose F → R, update corotated frame
- `commitState()` — store committed rotation
- `revertToLastCommit()` — restore committed rotation


### X5. Design Rules Summary

Binding architectural rules that fall out of all Q&A decisions above.
**These rules override any conflicting pattern found in existing code.**

1. **One kinematics object per element, always** — no per-GP kinematics objects.
2. **Kinematics is a pure geometry servant** — it computes B, strain, detJ, K_sigma on demand. It knows nothing about materials, sections, or integration weights.
3. **Element drives the integration loop** — kinematics never owns the loop.
4. **Element owns `u_e`** — kinematics never holds node references for continuum/shell/contact.
5. **No raw numpy in element or kinematics files** — all operations use native math objects (CTensor, Matrix, Vector).
6. **Math module is the single backend** — any numpy/backend usage lives only in the math module.
7. **Fix it once in the base, inherit everywhere** — no duplicated logic across sibling classes.
8. **Pre-allocate in `_domain()`, fill in-place in `_update()`** — no per-iteration allocation.
9. **Cache in `update(gp,...)`** — B, detJ, strain cached internally; subsequent getters are free.
10. **K_material + K_geometric combined in `getTangentStiff()`** — Domain/Analysis never sees geometric stiffness separately.
11. **UL reference update happens in `kinematics.commitState()`** — never during Newton iteration.
12. **`copy()` is always a full deep copy with independent state** — no aliasing.
13. **OpenSees API preserved on existing elements** — new kinematics API added on top, not replacing.
14. **Material interface is sacred** — do not change it under any circumstances.
15. **Write `docs/unified_kinematics_plan.md` before writing any code**.


### X6. Open Questions and Deferred Decisions

The following items are explicitly deferred and must be resolved in future work packages.
They are listed here so they are not forgotten.

| Item | Deferred to | Notes |
|------|-------------|-------|
| Truss kinematics formulation (F1-F3) | Truss refactor WP | Strain decomposition mechanically correct but not yet bound |
| Truss Section vs direct uniaxial Material (F2) | Truss refactor WP | — |
| TotalLagrangianCrdTransf / UpdatedLagrangianCrdTransf for beams | Beam nonlinear WP | Uncommon; forceBeamColumn port may clarify need |
| ZeroLength, ZeroLengthND, ZeroLengthSection ports (J3) | ZeroLength WP | Only ZeroLengthContactASDimplex in current WP |
| Contact large-deformation: updating normals and gap (K4) | Contact extension WP | ContactKinematics interface designed to accommodate |
| Arc-length control for snap-through benchmark | Analysis pipeline WP | Benchmark PASSED in this WP when limit load within 5% and solver diverges gracefully past limit point under displacement control. Full post-snap path recovery deferred. |
| ShellKinematics full implementation (I6) | ASDShellQ4 WP | Interface stubs created in this WP |
| 3D continuum elements: Brick8, Tet4 (H1) | 3D continuum WP | Node33 fixed in this WP |
| SSPQuad4UP, SSPBrick8UP ports (H1) | Future WPs | Source provided by PI at porting time |
| forceBeamColumn, dispBeamColumn, FiberSection ports | Nonlinear beam WP | Use existing CrdTransf unchanged |
| CorotContinuumKinematics full derivation + implementation | Continuum corot WP | EICR framework (Felippa 2005) selected. Interface stub with NotImplementedError created in Phase 4. Snap-through benchmark uses beam/truss corot, not Quad4 corot. |
| Node24, Node34, Node37 usage (R1, R2) | Mixed formulation WP | DOFs defined; fixed in node fix pass but not used in current WP |

---

*oneFEM — Unified Kinematics System Design Questionnaire — COMPLETE — v1.0 — March 2026*
*PI: Onur Deniz Akan*
