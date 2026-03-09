# oneFEM v2 Architecture Refactor -- Implementation Prompt

## Context

You are working on oneFEM, a Python finite element framework for computational
mechanics and earthquake engineering. A governing architecture document has
been written. Your job is to execute the refactor described in that document,
one phase at a time, with a full benchmark gate between every phase and
sub-phase.

## First: Read everything before touching any code

Read the full architecture document:

    docs/oneFEM_v2_architecture.md

Do not skim it. Every section matters. The invariants in that document are
non-negotiable. If you are ever unsure whether an implementation choice is
correct, re-read the relevant section before proceeding.

Then read the current state of the codebase:

    src/oneFEM/model/element/continuum/base.py
    src/oneFEM/model/element/continuum/hex8.py
    src/oneFEM/model/element/continuum/quad4.py
    src/oneFEM/model/element/continuum/isoparametric.py
    src/oneFEM/model/element/kinematics/continuum/base.py
    src/oneFEM/model/element/kinematics/continuum/nonlinear_base.py
    src/oneFEM/model/element/kinematics/continuum/linear.py
    src/oneFEM/model/element/kinematics/continuum/total_lagrangian.py
    src/oneFEM/model/element/kinematics/continuum/updated_lagrangian.py
    src/oneFEM/model/element/kinematics/continuum/corot.py

---

## Benchmark Gate

Run this after every phase and every sub-phase. All must pass before proceeding.

    python src/examples/hex8_benchmarks.py         # expect: 16/16
    python src/examples/hex8_diagnostic_checks.py  # expect: 24/24
    python src/examples/quad4_benchmarks.py        # expect: 8/8
    python src/examples/corot_benchmarks.py        # expect: 5/5
    python src/examples/dynamic_truss.py           # expect: PASS
    python src/examples/eigen_truss.py             # expect: PASS

If any benchmark regresses, stop. Fix the regression before touching anything
else. Do not proceed to the next phase with a broken gate.

---

## Implementation Sequence

Execute exactly in this order. Do not skip phases. Do not combine phases.

---

### Phase 0: Eliminate isoparametric.py + add element API

**Step 0a -- Move element-specific functions into element files**

Read `isoparametric.py` carefully first. Identify every function and classify:
- Element-specific (quad4_*, hex8_*) -> move into that element's own file
- Generic utilities (compute_jacobian, compute_physical_derivatives) -> move
  into `ContinuumElementBase` as a 2-line protected method

Specifically:
- `quad4_shape_functions`, `quad4_shape_derivatives`, `quad4_gauss_points`
  move to `element/continuum/cauchy/quad4.py` as private functions
- `hex8_shape_functions`, `hex8_shape_derivatives`, `hex8_gauss_points`
  move to `element/continuum/cauchy/hex8.py` as private functions
- `compute_jacobian` and `compute_physical_derivatives` move to
  `ContinuumElementBase` as protected static methods (or inline them)
- Delete `isoparametric.py`

The UL kinematics file imports `compute_physical_derivatives` from
`isoparametric.py` at line 113. Patch this import to use the new location
on `ContinuumElementBase`. This is a temporary fix -- Phase 3c will remove
the dependency entirely when UL calls `element.update_reference()` instead.

**Step 0b -- Add element API to ContinuumElement base class**

Add these methods to `element/continuum/base.py`. Implement by delegating to
precomputed data already stored on the element. This is purely additive --
do not break any existing data paths. Kinematics continue to use the old
`initialize()` path unchanged.

```python
# Geometry -- evaluated at natural coordinates xi (or gp index)
def get_N(self, xi)              # shape functions                      (nNodes,)
def get_dN_dxi(self, xi)         # shape fn derivatives wrt xi          (nNodes, nDim)
def get_jacobian(self, xi)       # J = dX/dxi                           (nDim, nDim)
def get_dN_dX(self, xi)          # dN/dX = J^-1 dN/dxi, ref config     (nNodes, nDim)
def get_dN_dx(self, xi)          # dN/dx, current config (for UL)       (nNodes, nDim)
def get_B(self, xi)              # linear strain-displacement matrix     (nVoigt, nDOF*nNodes)
def get_B_NL(self, xi)           # nonlinear B for K_geo               (nDim^2, nDOF*nNodes)
def get_F(self, xi)              # deformation gradient F = I + H       (nDim, nDim)
def get_H(self, xi)              # displacement gradient u^T dN/dX      (nDim, nDim)
def get_gauss_points(self)       # [(xi, weight), ...] quadrature rule

# State
def get_coords_ref(self)         # reference nodal coords               (nNodes, nDim)
def get_coords(self)             # current nodal coords                 (nNodes, nDim)
def get_disp(self)               # current nodal displacements          (nNodes*nDOF,)
def get_disp_committed(self)     # committed nodal displacements        (nNodes*nDOF,)
def get_stress(self, gp)         # stress at Gauss point                (nVoigt,)
def get_tangent(self, gp)        # constitutive tangent at Gauss point  (nVoigt, nVoigt)
def get_material(self, gp)       # material object at Gauss point
def get_dof_indices(self)        # global DOF indices for assembly      (nNodes*nDOF,)

# Dimensions
def get_nDim(self)               # spatial dimension                    int
def get_nNodes(self)             # number of nodes                      int
def get_nDOF_total(self)         # nNodes * nDOF                        int
def get_nVoigt(self)             # Voigt dimension                      int
def get_nGP(self)                # number of Gauss points               int

# Enrichment -- default implementations, overridden in Phase 5
def get_G(self, xi)              # incompatible mode B-matrix -> None
def get_nAlpha(self)             # number of internal modes  -> 0
def get_B_bar(self, xi)          # B-bar matrix              -> None
```

**Files changed:** `quad4.py`, `hex8.py`, `element/continuum/base.py`.
`isoparametric.py` deleted. UL kinematics import patched (one line).
**Benchmark gate:** all pass before proceeding.

---

### Phase 1: PhysicsFamily enum + move kinematics directory

**Step 1a -- Create PhysicsFamily enum**

Create `src/oneFEM/model/physics_family.py`:

```python
from enum import Enum

class PhysicsFamily(Enum):
    CONTINUUM_CAUCHY          = "continuum/cauchy"
    CONTINUUM_COSSERAT        = "continuum/cosserat"
    CONTINUUM_MICROMORPHIC    = "continuum/micromorphic"
    CONTINUUM_BIOT_CAUCHY     = "continuum/biot_cauchy"
    CONTINUUM_BIOT_COSSERAT   = "continuum/biot_cosserat"
    CONTINUUM_BIOT_MICRO      = "continuum/biot_micromorphic"
    SHELL                     = "shell"
    BEAM                      = "beam"
    ZL                        = "zl"
    CONTACT                   = "contact"
```

No other files changed. Benchmark gate: all pass.

**Step 1b -- Move kinematics directory**

Move:
    src/oneFEM/model/element/kinematics/
to:
    src/oneFEM/model/kinematics/

Reorganize to:
    src/oneFEM/model/kinematics/continuum/cauchy/linear.py
    src/oneFEM/model/kinematics/continuum/cauchy/total_lagrangian.py
    src/oneFEM/model/kinematics/continuum/cauchy/updated_lagrangian.py
    src/oneFEM/model/kinematics/continuum/cauchy/corot.py
    src/oneFEM/model/kinematics/continuum/cauchy/base.py
    src/oneFEM/model/kinematics/continuum/cauchy/nonlinear_base.py

Find every file that imports from the old location:

    grep -r "kinematics" src/ --include="*.py" -l

Update all import paths. Logic changes: none. Pure reorganization.
Benchmark gate: all pass.

---

### Phase 2: physics_family declarations and construction checks

**Goal:** Add `physics_family` class attribute and construction-time
validation to all elements, kinematics, and materials.

For each element add as a class attribute:
```python
from oneFEM.model.physics_family import PhysicsFamily
physics_family = PhysicsFamily.CONTINUUM_CAUCHY
```

For each element constructor add hard checks and warnings:
```python
import warnings

# Hard checks (raise AssertionError immediately)
assert len(nodes) == self.nNodes, \
    f"{self.__class__.__name__} requires {self.nNodes} nodes"
assert all(n.nDim == self.nDim and n.nDOF == self.nDOF for n in nodes), \
    f"{self.__class__.__name__} requires node_{self.nDim}_{self.nDOF} nodes"
assert kinematics.physics_family == self.physics_family, \
    f"{self.__class__.__name__}: kinematics physics_family mismatch. " \
    f"Expected {self.physics_family}, got {kinematics.physics_family}"
assert material.physics_family == self.physics_family, \
    f"{self.__class__.__name__}: material physics_family mismatch. " \
    f"Expected {self.physics_family}, got {material.physics_family}"

# Warnings (do not stop execution)
if hasattr(material, 'nu') and material.nu > 0.45 and not bbar:
    warnings.warn(
        f"{self.__class__.__name__}: nu={material.nu:.3f} nearly incompressible "
        f"but bbar=False. Volumetric locking likely. Use bbar=True.",
        UserWarning, stacklevel=2
    )
if not incompatible:
    warnings.warn(
        f"{self.__class__.__name__}: incompatible=False. Shear locking in "
        f"bending-dominated problems. Use incompatible=True for beam/arch geometry.",
        UserWarning, stacklevel=2
    )
if kinematics.__class__.__name__ == 'Linear':
    warnings.warn(
        f"{self.__class__.__name__}: Linear kinematics for small deformation only. "
        f"Use TotalLagrangian or UpdatedLagrangian for large deformation.",
        UserWarning, stacklevel=2
    )
```

Add `physics_family = PhysicsFamily.CONTINUUM_CAUCHY` to all kinematics and
material classes as well.

Add `import warnings; warnings.filterwarnings("ignore")` at the top of all
benchmark scripts to suppress the new construction warnings in test output.

**Files changed:** all element, kinematics, and material files.
**Benchmark gate:** all pass.

---

### Phase 3: Refactor kinematics to use element API

One kinematics class at a time. Full benchmark gate after each sub-phase.

**Core invariant for all sub-phases:** after refactoring, each kinematics
class must store ZERO geometry from the element. No `self._dN_dX`, no
`self._nNodes`, no `self._X_ref`. Every geometry access goes through the
element API.

New `initialize()` signature for all kinematics:
```python
def initialize(self, element):
    self._element = element   # store reference only, never geometry
```

**Phase 3a -- Linear kinematics**

Replace `initialize(nGP, nDim, nNodes, dN_dX_list)` with `initialize(element)`.
Remove all stored geometry: `self._B_list`, `self._dN_dX`, `self._nNodes`, etc.
`getStiffness()` and `getForce()` call `element.get_B(gp)`,
`element.get_tangent(gp)`, `element.get_jacobian(gp)` directly.
The `_buildBMatrix()` static method is no longer needed in kinematics -- it
was already moved into the element as `get_B(xi)` in Phase 0.
Benchmark gate: all pass.

**Phase 3b -- TotalLagrangian kinematics**

Replace `initialize()` with element-reference version. Remove all stored geometry.
- `_computeH(gp, u_e)` -> `element.get_H(gp)` (element computes it)
- `_buildBNL(gp)` -> use `element.get_B_NL(gp)`
- `_buildKgeo(gp)` -> use `element.get_dN_dX(gp)` and `element.get_stress(gp)`
- Remove `needs_incremental_u` property entirely. TL computes:
  `delta_u = element.get_disp() - element.get_disp_committed()`
- Remove `_setMaterialStrain`. Call `element.get_material(gp).setStrain()` directly.
Benchmark gate: all pass.

**Phase 3c -- UpdatedLagrangian kinematics**

Same pattern as TL. Additionally:
- `commitState()` calls `element.update_reference()` -- this removes the
  `isoparametric.py` cross-layer import added as a temporary patch in Phase 0.
  Delete that temporary patch now.
- Uses `element.get_dN_dx(gp)` for current-config spatial derivatives.
Benchmark gate: all pass.

**Phase 3d -- CorotContinuum kinematics**

Replace `initialize()` with element-reference version.
Remove stored `X_ref`, `C0`, `_B_local`.
`_extract_R()`: replace hardcoded 4-node unpacking (lines 86-128) with a
loop over `range(element.get_nNodes())`. The current code unpacks named
variables aX1, aX2, aX3, aX4 -- this hard-wires Quad4 and blocks Hex8. Fix
this to work with any nNodes.
`transformToGlobal()` uses `element.get_nNodes()` and `element.get_nDim()`
instead of stored `self._nNodes`.
Benchmark gate: all pass.

**Phase 3e -- Element base cleanup**

Remove from `element/continuum/base.py`:
- The `initialize()` call and all kwargs passed to kinematics (lines 93-96)
- The `needs_incremental_u` branch (line 135)
- The `_setMaterialStrain` call (line 148)
- Any dead code that only existed to serve kinematics

`element._update()` simplifies to:
```python
def _update(self):
    self._kinematics.update(self)
```
Benchmark gate: all pass.

---

### Phase 4: Kinematics owns the K/f integration loop

Move the Gauss point integration loop from `element._buildStiffnessAndForce()`
into each kinematics class. One class at a time, benchmark gate after each.

Each kinematics class gains:
```python
def getK(self, element) -> np.ndarray        # (nDOF_total, nDOF_total)
def get_f_int(self, element) -> np.ndarray   # (nDOF_total,)
```

**Phase 4a -- Linear**
```python
def getK(self, element):
    n = element.get_nDOF_total()
    K = np.zeros((n, n))
    for gp, (xi, w) in enumerate(element.get_gauss_points()):
        B = element.get_B(xi)
        J = np.linalg.det(element.get_jacobian(xi))
        C = element.get_tangent(gp)
        K += w * J * B.T @ C @ B
    return K

def get_f_int(self, element):
    n = element.get_nDOF_total()
    f = np.zeros(n)
    for gp, (xi, w) in enumerate(element.get_gauss_points()):
        B = element.get_B(xi)
        J = np.linalg.det(element.get_jacobian(xi))
        S = element.get_stress(gp)
        f += w * J * B.T @ S
    return f
```
Benchmark gate: all pass.

**Phase 4b -- TotalLagrangian** (K_mat + K_geo assembled in one loop).
```python
def getK(self, element):
    n = element.get_nDOF_total()
    K = np.zeros((n, n))
    for gp, (xi, w) in enumerate(element.get_gauss_points()):
        B    = element.get_B(xi)
        B_NL = element.get_B_NL(xi)
        J    = np.linalg.det(element.get_jacobian(xi))
        C    = element.get_tangent(gp)
        S    = element.get_stress(gp)
        S_mat = self._voigt_to_matrix(S)
        K   += w * J * (B.T @ C @ B + B_NL.T @ S_mat @ B_NL)
    return K
```
Benchmark gate: all pass.

**Phase 4c -- UpdatedLagrangian** (same structure, current-config quantities).
Benchmark gate: all pass.

**Phase 4d -- CorotContinuum** (rotation extraction via element API, local
stiffness assembled, then transformed to global via `transformToGlobal(element)`).
Benchmark gate: all pass.

**Phase 4e -- Element base final simplification.**
Remove `_buildStiffnessAndForce()`. `_update()` final form:
```python
def _update(self):
    self._kinematics.update(self)
    self._k = self._kinematics.getK(self)
    self._f = self._kinematics.get_f_int(self)
```
Benchmark gate: all pass.

---

### Phase 5: Enrichment flag infrastructure (stubs only, no math)

**Scope:** `element/continuum/cauchy/hex8.py` only.
Zero kinematics changes. Zero math implementation.

**Phase 5a -- Add flags and internal alpha state to Hex8 constructor:**
```python
def __init__(self, nodes, kinematics, material, incompatible=False, bbar=False):
    super().__init__(nodes, kinematics, material)
    self._incompatible     = incompatible
    self._bbar             = bbar
    self._alpha            = np.zeros(9)
    self._alpha_commit     = np.zeros(9)
```

**Phase 5b -- commit() and revert() save/restore alpha:**
```python
def commit(self):
    super().commit()
    self._alpha_commit = self._alpha.copy()

def revert(self):
    super().revert()
    self._alpha = self._alpha_commit.copy()
```

**Phase 5c -- Stub dispatch in getStiffness() and getForce():**
```python
def getStiffness(self):
    if self._incompatible: return self._stiffness_incompatible()
    if self._bbar:         return self._stiffness_bbar()
    return super().getStiffness()

def getForce(self):
    if self._incompatible: return self._force_incompatible()
    if self._bbar:         return self._force_bbar()
    return super().getForce()

def _stiffness_incompatible(self):
    raise NotImplementedError("Incompatible modes not yet implemented")
def _force_incompatible(self):
    raise NotImplementedError("Incompatible modes not yet implemented")
def _stiffness_bbar(self):
    raise NotImplementedError("B-bar not yet implemented")
def _force_bbar(self):
    raise NotImplementedError("B-bar not yet implemented")
```

**Phase 5d -- Override enrichment API:**
```python
def get_G(self, xi):
    if self._incompatible:
        raise NotImplementedError("get_G: incompatible modes not yet implemented")
    return None

def get_nAlpha(self):
    return 9 if self._incompatible else 0
```

**Benchmark gate:** all pass. Stubs are not triggered with default flags.

---

## Final Report

After all phases complete, report:
1. Final benchmark gate results -- all 6 scripts, explicit pass/fail counts
2. Summary of every file changed in each phase
3. Any unexpected issues encountered and how they were resolved
4. Explicit confirmation that zero changes were made to:
   Analysis, Algorithm, Integrator, Numberer, System, Assembler, any node file

**Do not implement incompatible modes math or B-bar math.** That is a
separate session after this refactor is complete and all benchmarks are green.
