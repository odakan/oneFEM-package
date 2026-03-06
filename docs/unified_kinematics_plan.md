# Unified Kinematics System — Implementation Plan
**Version 1.0 — March 2026**
**Branch: `python_module`**

---

## Status Table

| Phase | Objective | Status | Validation |
|-------|-----------|--------|------------|
| 1 | Node fixes + kinematics hierarchy skeleton + math module API verification | Complete | 56/56 node tests PASS, math API PASS, beam 10/10, buckling 8/8 |
| 2 | Quad4 Linear | Complete | Patch test 6/6 ALL PASS, Cook's membrane 2/2 ALL PASS |
| 3 | Quad4 TL + UL | Complete | Simple shear 12/12 ALL PASS, cantilever rollup 4/4 ALL PASS |
| 4a | Beam refactor (merge coordTransformation -> kinematics) | Complete | truss ALL PASS, dynamic ALL PASS, eigen ALL PASS, epp ALL PASS, beam 10/10, buckling 8/8 |
| 4b | CorotContinuumKinematics stub + corot benchmarks | Complete | Snap-through arch P_cr=0.5874 (ref 0.5878, 0.1%), Lee's frame qualitative 3/3 PASS, all 5/5 PASS |

---

## Phase 1: Node Fixes + Kinematics Hierarchy Skeleton + Math Module Extensions

**Objective:** Clear all node debt, establish the complete `Kinematics` class hierarchy with correct file layout, and extend the math module with all operations needed for Phase 2 integration loops.

### 1A. Fix All Broken Nodes

**Reference pattern:** `Node23` (working) and `Node36` (working). All fixed nodes must follow this exact pattern.

**Files to modify:**

| File | Node | Issues | Fix |
|------|------|--------|-----|
| `src/oneFEM/model/node/node_2_2.py` | Node22 | Stub only (5 lines). Passes extra arg `coords` to `Node.__init__()`. No attributes, no methods. | Full rewrite following Node23 pattern. nD=2, nDOF=2. |
| `src/oneFEM/model/node/node_3_3.py` | Node33 | Double-underscore `__attr` on all attributes (breaks base class). `_update()` has wrong signature (missing `force`). `_commit()` instead of `_commitState()`. No `_revertToLastCommit()`. Mutable default `coord=[]`. | Full rewrite following Node36 pattern. nD=3, nDOF=3. |
| `src/oneFEM/model/node/node_2_4.py` | Node24 | Stub only (5 lines). Same extra-arg bug as Node22. | Full rewrite following Node23 pattern. nD=2, nDOF=4. DOFs: ux, uy, theta_z, p. |
| `src/oneFEM/model/node/node_3_4.py` | Node34 | Stub only (5 lines). Same extra-arg bug. | Full rewrite following Node36 pattern. nD=3, nDOF=4. DOFs: ux, uy, uz, p. |
| `src/oneFEM/model/node/node_3_7.py` | Node37 | Stub only (5 lines). Same extra-arg bug. | Full rewrite following Node36 pattern. nD=3, nDOF=7. DOFs: ux, uy, uz, theta_x, theta_y, theta_z, p. |

**Each fixed node must have:**
- `__init__(self, node_id, coord=None, mass=None, fix=None)` with `None` defaults
- `super().__init__(node_id)` — no extra args
- Single-underscore `_attr` for ALL attributes
- `_nD`, `_nDOF` set to correct values
- All Vectors pre-initialized with correct sizes
- `_update(self, force, disp, vel=None, accel=None)` — matching Node base signature
- `_commitState(self)` — deep copy via `Vector(self._x_trial)`
- `_revertToLastCommit(self)` — deep copy via `Vector(self._x_commit)`
- `_revertToStart(self)` — reset all to zeros
- `_getTrialDisp/Vel/Accel`, `_getCommitDisp/Vel/Accel`
- `_setDOF`, `_setFix`, `_setMass`, `_getResult`, `_getCoordinates`, `_getDOFIndices`, `_getFixity`
- `__repr__`

**File to modify:** `src/oneFEM/model/node/__init__.py` — verify all nodes are exported.

### 1B. Kinematics Hierarchy Skeleton

**Restructure `model/element/kinematics/`** to the target directory tree from X2. Merge `coordTransformation/` into `kinematics/`.

**Directory restructure:**
```
model/element/kinematics/
├── __init__.py                     (update exports)
├── base.py                         (Kinematics base — move from main.py)
│
├── crdTransf/                      (beam kinematics — move from coordTransformation/)
│   ├── __init__.py
│   ├── base.py                     (CrdTransf — move from coordTransformation/main.py)
│   ├── linear_2d.py                (move from coordTransformation/)
│   ├── linear_3d.py                (move from coordTransformation/)
│   ├── pdelta_2d.py                (move from coordTransformation/)
│   ├── pdelta_3d.py                (move from coordTransformation/)
│   ├── corot_2d.py                 (move from coordTransformation/)
│   └── corot_3d.py                 (move from coordTransformation/)
│
├── continuum/                      (solid element kinematics)
│   ├── __init__.py
│   ├── base.py                     (ContinuumKinematics — move from continuum_main.py)
│   ├── linear.py                   (LinearContinuumKinematics — move from linear.py)
│   ├── total_lagrangian.py         (TotalLagrangianContinuumKinematics — Phase 3)
│   ├── updated_lagrangian.py       (UpdatedLagrangianContinuumKinematics — Phase 3)
│   └── corot.py                    (CorotContinuumKinematics — Phase 4b stub)
│
├── shell/                          (interface stubs only)
│   ├── __init__.py
│   └── base.py                     (ShellKinematics(Kinematics) — stub)
│
└── contact/                        (interface stubs only)
    ├── __init__.py
    └── base.py                     (ContactKinematics(Kinematics) — stub)
```

**Key changes to existing classes:**

**`kinematics/base.py` (Kinematics)** — enhanced from current `main.py`:
```python
class Kinematics(object):
    formulation = None  # 'linear', 'corotational', 'totalLagrangian', 'updatedLagrangian', 'pdelta'

    def initialize(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def commitState(self):
        pass

    def revertToLastCommit(self):
        pass

    def copy(self):
        raise NotImplementedError
```

**`kinematics/continuum/base.py` (ContinuumKinematics)** — refactored from `continuum_main.py`:
- Per-GP state management (one kinematics object per element, GP index argument)
- New signature: `initialize(self, nGP, nNodes, nDim)` — pre-allocates per-GP arrays
- New signature: `update(self, gp, dN_dX, u_e)` — computes and caches B, strain, detJ for GP
- New accessors: `getStrain(self, gp)`, `getBMatrix(self, gp)`, `getDetJ(self, gp)`, `getGeometricStiffness(self, gp, stress)`
- All return native math objects (Matrix, CTensor, scalar)

**`kinematics/continuum/linear.py` (LinearContinuumKinematics)** — refactored:
- Caches B matrices per GP (B is constant for linear, computed once in `initialize()` or first `update()`)
- Returns `Matrix` (not raw numpy) from `getBMatrix(gp)`
- Returns `CTensor` (2nd, COV) from `getStrain(gp)`
- Returns `None` from `getGeometricStiffness(gp, stress)` (linear has none)

**`kinematics/shell/base.py` (ShellKinematics)** — stub:
```python
class ShellKinematics(Kinematics):
    """Shell kinematics base. Interface designed for ASDShellQ4 port."""
    pass
```

**`kinematics/contact/base.py` (ContactKinematics)** — stub:
```python
class ContactKinematics(Kinematics):
    """Contact/zero-length kinematics base. Manages orientation -> rotation matrix T."""
    pass
```

**Files to delete after move:**
- `src/oneFEM/model/element/coordTransformation/` (entire directory — contents moved to `kinematics/crdTransf/`)
- `src/oneFEM/model/element/kinematics/main.py` (→ `base.py`)
- `src/oneFEM/model/element/kinematics/continuum_main.py` (→ `continuum/base.py`)
- `src/oneFEM/model/element/kinematics/linear.py` (→ `continuum/linear.py`)

**Import updates required in:**
- `src/oneFEM/model/element/__init__.py` — remove `coordTransformation` import, update `kinematics` exports
- `src/oneFEM/model/element/beam/elastic_beam_column_2d.py` — update CrdTransf import path
- `src/oneFEM/model/element/beam/elastic_beam_column_3d.py` — update CrdTransf import path
- Any other file importing from `coordTransformation/`

### 1C. Math Module Extensions

**File to modify:** `src/oneFEM/_systools/data/matrix.py`

Add the following methods to `Matrix`:
```python
@property
def T(self):
    """Return transpose as a new Matrix."""
    return Matrix(init=self.__data.T, dtype=self.__dtype)

def __matmul__(self, other):
    """Matrix @ Matrix -> Matrix, Matrix @ Vector -> Vector, Matrix @ CTensor -> handled."""
    from .vector import Vector
    from .ctensor import CTensor
    if isinstance(other, Matrix):
        return Matrix(init=self.__data @ other.data, dtype=self.__dtype)
    if isinstance(other, Vector):
        result = self.__data @ other.data
        return Vector(list(result), dtype=self.__dtype)
    if isinstance(other, CTensor):
        # Matrix @ CTensor(4th) -> Matrix (via Voigt matrix)
        # Matrix @ CTensor(2nd) -> Vector (via Voigt vector)
        if other._order == 4:
            C_mat = other.to_matrix()
            return Matrix(init=self.__data @ C_mat.data, dtype=self.__dtype)
        elif other._order == 2:
            s_vec = other.to_vector()
            result = self.__data @ s_vec.data
            return Vector(list(result), dtype=self.__dtype)
    return NotImplemented
```

**File to modify:** `src/oneFEM/_systools/data/ctensor.py`

Add the following methods to `CTensor`:
```python
def to_matrix(self):
    """Convert 4th-order CTensor to Matrix (Voigt nxn).
    Returns the contravariant representation matrix (correct for B^T C B)."""
    from .matrix import Matrix
    if self._order != 4:
        raise ValueError("to_matrix() only valid for 4th-order CTensor")
    n = self._nRows
    data = [[self._data[i * n + j] for j in range(n)] for i in range(n)]
    return Matrix(data, dtype=float)

def to_vector(self):
    """Convert 2nd-order CTensor to Vector (Voigt column).
    Returns contravariant components (stress-like: sigma_12 * 2 for shear)."""
    from .vector import Vector
    if self._order != 2:
        raise ValueError("to_vector() only valid for 2nd-order CTensor")
    n = self._nRows
    return Vector([self._data[i] for i in range(n)], dtype=float)
```

**File to modify:** `src/oneFEM/_systools/data/vector.py`

Add `__iadd__` for in-place accumulation:
```python
def __iadd__(self, other):
    if isinstance(other, Vector):
        self.__data += other.data
    else:
        self.__data += asarray(other)
    return self
```

### 1D. Math Module API Verification Test

**File to create:** `tests/math/test_integration_loop_ops.py`

Tests all X1 operations on concrete small examples:
1. `CTensor(4th).to_matrix()` → correct 3x3 for PlaneStress (C_1111 = E/(1-nu^2), C_1212 = G)
2. `CTensor(2nd).to_vector()` → correct length-3 vector
3. `Matrix.T` → correct transpose
4. `Matrix @ Matrix` → correct product
5. `Matrix @ Vector` → correct product
6. `Matrix @ CTensor(4th)` → correct `C_voigt @ B` result
7. `Matrix.T @ CTensor(2nd)` → correct `B^T sigma` result
8. `B.T @ C.to_matrix() @ B` → correct 8x8 stiffness for unit Quad4
9. Shear stiffness sanity check: pure shear → G = E/(2(1+nu))

### 1E. Isoparametric Utility Module

**File to create:** `src/oneFEM/model/element/continuum/isoparametric.py`

Pure functions (no state):
```python
def quad4_shape_functions(xi, eta):
    """4-node bilinear quad shape functions.
    Returns: N array of shape (4,)"""

def quad4_shape_derivatives(xi, eta):
    """Natural derivatives dN/d(xi,eta).
    Returns: dN_dxi array of shape (4, 2)"""

def quad4_gauss_points():
    """2x2 Gauss quadrature points and weights.
    Returns: list of (xi, eta, w) tuples, 4 points"""

def compute_jacobian(dN_dxi, X_nodes):
    """Compute Jacobian matrix J = dN_dxi^T @ X_nodes.
    Returns: J (2x2 or 3x3 numpy array), detJ (scalar)"""

def compute_physical_derivatives(dN_dxi, X_nodes):
    """Compute dN/dX = J^{-1} @ dN/dxi.
    Returns: dN_dX array of shape (nNodes, nDim), detJ (scalar)"""
```

### Phase 1 Validation

1. **Node validation script** (`tests/nodes/test_all_nodes.py`):
   - Instantiate each node type (Node22, Node23, Node24, Node33, Node34, Node36, Node37)
   - Verify `_nD`, `_nDOF`, `_coord` dimensions
   - Test `_update()` → `_commitState()` → `_revertToLastCommit()` cycle
   - Verify deep copy (modify trial after commit, revert must restore)
   - Verify `_setDOF`, `_setFix`, `_setMass` with correct and wrong-length inputs
   - All must PASS

2. **Math API test** (`tests/math/test_integration_loop_ops.py`):
   - All 9 assertions from X1b must pass

3. **Import test**: `python -c "from oneFEM.model.element.kinematics import Kinematics, ContinuumKinematics, LinearContinuumKinematics; from oneFEM.model.element.kinematics.crdTransf import CrdTransf, LinearCrdTransf2d, LinearCrdTransf3d"` — must succeed

4. **Existing benchmark regression**: Run `src/truss.py`, `src/examples/beam.py` — all must still PASS (import paths may have changed for CrdTransf)

---

## Phase 2: Quad4 Linear

**Objective:** Implement `ContinuumElement` base class and `Quad4` element with `LinearContinuumKinematics`, validated by patch test and Cook's membrane.

### 2A. ContinuumElement Base Class

**File to create:** `src/oneFEM/model/element/continuum/base.py`

```python
class ContinuumElement(Element):
    """Base class for continuum (solid) elements.
    Owns the integration loop. Subclasses define topology (shape functions).
    """

    def __init__(self, tag, nodes, material, kinematics=None, thickness=1.0):
        super().__init__(tag)
        # Store constructor args (material is template, copied per GP in _domain)
        # kinematics defaults to LinearContinuumKinematics if None

    def _domain(self):
        """Initialize element geometry, allocate arrays, copy materials per GP."""
        # 1. Get node coordinates -> self._X_nodes (Matrix, nNodes x nDim)
        # 2. self._kinematics.initialize(nGP, nNodes, nDim)
        # 3. Create material copies: self._materials = [material.getCopy() for _ in range(nGP)]
        # 4. Pre-allocate: self._K, self._f, self._u_e (if nonlinear)
        # 5. Compute and cache dN_dX and detJ for each GP (for linear, these are constant)
        # 6. Set self._nDOF, self._nD, self._nodes

    def _update(self):
        """Extract nodal displacements, update kinematics and materials per GP."""
        # For nonlinear: extract u_e from nodes, fill self._u_e in-place
        # For each GP: kinematics.update(gp, dN_dX, u_e) -> get strain -> material._setTrialStrain(strain)

    def getTangentStiff(self):
        """Assemble element tangent stiffness from all GPs."""
        # K = sum over GPs: B^T @ C @ B * detJ * w * thickness
        # + kinematics.getGeometricStiffness(gp, stress) (if nonlinear)
        # Return Matrix

    def getResistingForce(self):
        """Assemble element internal force from all GPs."""
        # f = sum over GPs: B^T @ sigma * detJ * w * thickness
        # Return Vector

    def _commit(self):
        """Commit kinematics and materials."""
        # self._kinematics.commitState()
        # for mat in self._materials: mat._commitState()

    def _revert(self):
        """Revert kinematics and materials."""
        # self._kinematics.revertToLastCommit()
        # for mat in self._materials: mat._revertToLastCommit()

    def getMass(self):
        """Lumped mass matrix (if rho > 0)."""
        # Default: None

    def getDamp(self):
        """Damping matrix."""
        # Default: None

    def getInitialStiff(self):
        """Stiffness from initial tangent."""

    def getForce(self):
        """Return committed internal force."""
        return self._f

    def getStiffness(self):
        """Return committed stiffness."""
        return self._k
```

### 2B. Quad4 Element

**File to create:** `src/oneFEM/model/element/continuum/quad4.py`

```python
class Quad4(ContinuumElement):
    """4-node bilinear quadrilateral element (2D).

    Nodes: 4 x Node22 (2 translational DOFs each), total 8 DOFs.
    Integration: 2x2 Gauss (full).
    Material: nDMaterial (PlaneStress or PlaneStrain).
    Kinematics: injected, defaults to LinearContinuumKinematics.
    """

    def __init__(self, tag, nodes, material, kinematics=None, thickness=1.0):
        if kinematics is None:
            kinematics = LinearContinuumKinematics()
        super().__init__(tag, nodes, material, kinematics, thickness)

    def _domain(self):
        """Set up Quad4-specific geometry using isoparametric module."""
        # Call quad4_gauss_points() for GP locations and weights
        # Call super()._domain() or implement directly:
        #   - extract 4 node coordinates -> X_nodes (4x2)
        #   - for each GP: compute dN_dxi, then dN_dX and detJ via isoparametric module
        #   - cache dN_dX[gp] and detJ[gp]
        #   - initialize kinematics
        #   - copy materials per GP
```

### 2C. Refactor LinearContinuumKinematics

**File:** `src/oneFEM/model/element/kinematics/continuum/linear.py`

Refactor to per-GP caching pattern:
```python
class LinearContinuumKinematics(ContinuumKinematics):
    formulation = 'linear'

    def initialize(self, nGP, nNodes, nDim):
        """Pre-allocate B matrix cache. For linear, B depends only on dN_dX (constant)."""
        self._nGP = nGP
        self._nDim = nDim
        self._nNodes = nNodes
        self._B = [None] * nGP       # cached B matrices (Matrix)
        self._detJ = [None] * nGP    # cached Jacobian determinants

    def update(self, gp, dN_dX, u_e):
        """Compute and cache B matrix for this GP. For linear, B is displacement-independent."""
        if self._B[gp] is None:
            self._B[gp] = self._buildBMatrix(dN_dX)
        # Strain = B @ u_e (only needed when u_e is provided)

    def getStrain(self, gp):
        """Return cached strain as CTensor(2nd, COV)."""

    def getBMatrix(self, gp):
        """Return cached B matrix as Matrix."""
        return self._B[gp]

    def getDetJ(self, gp):
        """Return cached Jacobian determinant."""
        return self._detJ[gp]

    def getGeometricStiffness(self, gp, stress):
        """Linear: no geometric stiffness."""
        return None
```

### 2D. Update __init__.py Files

- `src/oneFEM/model/element/continuum/__init__.py` — export `ContinuumElement`, `Quad4`
- `src/oneFEM/model/element/__init__.py` — add `continuum` module

### Phase 2 Validation

**File to create:** `src/examples/quad4_linear.py`

**Benchmark 1 — Patch Test (3 load cases):**
- MacNeal-Harder 5-element patch on unit square, 9 nodes (node 9 at [0.24, 0.22])
- E=1.0, nu=0.25, plane stress, t=1.0
- 3 load cases: pure sigma_x, pure sigma_y, pure tau_xy
- Prescribe boundary node displacements, leave node 9 free
- **Pass:** Stress at ALL 20 Gauss points matches constant state to rel error < 1e-10
- **Pass:** Node 9 displacement matches analytical field exactly

**Benchmark 2 — Cook's Membrane (mesh convergence):**
- Tapered panel, E=1.0, nu=1/3, plane stress, t=1.0
- 5 meshes: 1x1, 2x2, 4x4, 8x8, 16x16
- Distributed shear V=1.0 on right edge
- **Pass:** 16x16 tip displacement within 0.1 of 23.91
- **Pass:** Monotonic convergence from coarse to fine
- **Output:** Convergence plot saved to `docs/validation/cooks_membrane_convergence.png`
- **Output:** Deformed mesh plot for 16x16 saved to `docs/validation/cooks_membrane_deformed.png`

---

## Phase 3: Quad4 TL + UL

**Objective:** Implement TotalLagrangian and UpdatedLagrangian kinematics for Quad4, validated by simple shear, cantilever rollup, and thick-walled cylinder.

### 3A. TotalLagrangianContinuumKinematics

**File to create:** `src/oneFEM/model/element/kinematics/continuum/total_lagrangian.py`

```python
class TotalLagrangianContinuumKinematics(ContinuumKinematics):
    formulation = 'totalLagrangian'

    def initialize(self, nGP, nNodes, nDim):
        """Pre-allocate: F, B_L, B_NL, E, detJ per GP. Store reference dN_dX."""

    def update(self, gp, dN_dX, u_e):
        """Compute deformation gradient F = I + grad(u).
        Build Green-Lagrange strain E = 0.5*(F^T F - I).
        Build B = B_L + B_NL(u) (linearized strain-displacement).
        Cache F, E, B, detJ."""

    def getStrain(self, gp):
        """Return E as CTensor(2nd, COV)."""

    def getBMatrix(self, gp):
        """Return B (material part) as Matrix."""

    def getDetJ(self, gp):
        """Return detJ_0 (reference Jacobian)."""

    def getGeometricStiffness(self, gp, stress):
        """Compute K_sigma from 2nd Piola-Kirchhoff stress S and dN_dX.
        K_sigma[IJ] = sum_ij S_ij * (dN_I/dX_i * dN_J/dX_j) * delta_ab
        Return as Matrix."""

    def commitState(self):
        """TL: no reference update needed (reference = initial always)."""
        pass

    def copy(self):
        """Deep copy with independent per-GP state."""
```

**Key formulas (Bathe Ch. 6, Bonet & Wood Ch. 7):**
- `F = I + du/dX` where `du/dX = sum_I (u_I outer dN_I/dX)`
- `E = 0.5 * (F^T F - I)` (Green-Lagrange)
- `B_L` = standard linear B (from reference config dN/dX)
- `B_NL` = displacement-dependent nonlinear B matrix
- `K_sigma` = initial stress (geometric) stiffness

### 3B. UpdatedLagrangianContinuumKinematics

**File to create:** `src/oneFEM/model/element/kinematics/continuum/updated_lagrangian.py`

```python
class UpdatedLagrangianContinuumKinematics(ContinuumKinematics):
    formulation = 'updatedLagrangian'

    def initialize(self, nGP, nNodes, nDim):
        """Pre-allocate per-GP arrays. Store reference node coords (initial config).
        Store committed reference coords (updated at commitState)."""
        self._X_ref = None           # committed reference coordinates
        self._X_ref_commit = None    # backup for revert

    def update(self, gp, dN_dX, u_e):
        """Compute F relative to last committed config.
        Build Almansi strain e = 0.5*(I - F^{-T} F^{-1}).
        Build B in current config.
        Cache F, e, B, detJ_current."""

    def getStrain(self, gp):
        """Return e (Almansi) as CTensor(2nd, COV)."""

    def getDetJ(self, gp):
        """Return detJ (current config Jacobian)."""

    def getGeometricStiffness(self, gp, stress):
        """Compute K_sigma from Cauchy stress sigma and dN/dx (current config)."""

    def commitState(self):
        """Update reference coords to current deformed coords (deep copy)."""
        self._X_ref_commit = [x.copy() for x in self._X_ref]  # backup
        # X_ref = X_ref + u_e (current deformed becomes new reference)

    def revertToLastCommit(self):
        """Restore reference coords from backup."""
        self._X_ref = [x.copy() for x in self._X_ref_commit]

    def copy(self):
        """Deep copy with independent per-GP state and reference coords."""
```

**Critical UL rule:** Reference coordinates update ONLY in `commitState()`, NEVER during `update()`. During Newton iteration, `dN_dX` must be computed from the kinematics' committed reference copy, not from live node coordinates.

### 3C. Update ContinuumElement for Nonlinear

Modify `src/oneFEM/model/element/continuum/base.py`:
- `_update()`: extract `u_e` from nodes for nonlinear formulations
- `getTangentStiff()`: add geometric stiffness contribution
- Handle `detJ` from kinematics (current config for UL, reference for TL)

### Phase 3 Validation

**File to create:** `src/examples/quad4_nonlinear.py`

**Benchmark 3 — Simple Shear (kinematic unit test):**
- Single Quad4, unit square, E=1000, nu=0.0
- Prescribed displacements (bypass solver): u_x = gamma * Y, u_y = 0
- Test gamma = 0.1, 0.5, 1.0, 2.0
- **Pass (TL):** F, E match analytical to < 1e-14 at centroid
- **Pass (UL):** F, e match analytical to < 1e-14 at centroid
- **Pass:** det(F) = 1.0 to machine precision (isochoric)

**Benchmark 4 — Cantilever Rollup:**
- 40x2 mesh, L=10, H=1, E=1.2e6, nu=0, plane stress
- Tip moment M_full = EI*2pi/L, 20 equal load steps
- **Pass:** Tip coords within 1% of analytical at each step
- **Pass:** TL and UL agree to 6 significant figures
- **Pass:** Quadratic Newton convergence (print residuals for steps 5, 10, 15, 20)
- **Output:** Load-displacement curve to `docs/validation/cantilever_rollup.png`
- **Output:** Deformed mesh at steps 5, 10, 15, 20 to `docs/validation/cantilever_deformed.png`
- **Output:** Newton convergence plot to `docs/validation/cantilever_newton.png`

**Benchmark 8 — Thick-Walled Cylinder:**
- Quarter model, a=1, b=2, 8 elements radially, plane strain
- E=210000, nu=0.3, P_in = 0.01*E = 2100
- **Pass:** u_r within 1% of plane strain Lame at all radial positions
- **Pass:** TL and UL agree to 4 significant figures
- **Output:** Radial displacement plot to `docs/validation/cylinder_ur.png`

---

## Phase 4a: Beam Refactor (Zero Regression)

**Objective:** Merge `coordTransformation/` into `kinematics/crdTransf/`, update all import paths, confirm zero regression on all existing benchmarks.

### Changes

This is purely a file-move and import-path operation. No behavioral changes.

1. If not done in Phase 1: Move all CrdTransf files from `coordTransformation/` to `kinematics/crdTransf/`
2. Update imports in:
   - `src/oneFEM/model/element/beam/elastic_beam_column_2d.py`
   - `src/oneFEM/model/element/beam/elastic_beam_column_3d.py`
   - `src/oneFEM/model/element/beam/dispBeamColumn.py` (if it imports CrdTransf)
   - `src/oneFEM/model/element/__init__.py` (remove old `coordTransformation` import)
   - Any benchmark scripts that import CrdTransf directly
3. Delete `src/oneFEM/model/element/coordTransformation/` directory

### Phase 4a Validation

Run ALL existing benchmarks — every single one must produce identical results:

| Benchmark | File | Expected |
|-----------|------|----------|
| 3D truss | `src/truss.py` | ALL PASS |
| Dynamic truss | `src/dynamic_truss.py` | ALL PASS |
| Eigen truss | `src/eigen_truss.py` | ALL PASS |
| EPP truss | `src/epp_truss.py` | ALL PASS |
| Beam 2D/3D | `src/examples/beam.py` | ALL PASS (10/10) |
| Column buckling | `src/examples/column_buckling.py` | ALL PASS (8/8) |

Any failure = regression = Phase 4a not complete.

---

## Phase 4b: CorotContinuumKinematics Stub + Corot Benchmarks

**Objective:** Create the CorotContinuumKinematics interface stub (NotImplementedError body), then validate corotational formulation using existing beam/truss elements on snap-through arch and Lee's frame.

### 4b-A. CorotContinuumKinematics Stub

**File to create:** `src/oneFEM/model/element/kinematics/continuum/corot.py`

```python
class CorotContinuumKinematics(ContinuumKinematics):
    """Corotational continuum kinematics (EICR framework, Felippa & Haugen 2005).
    Interface stub — full implementation deferred to Continuum Corot WP.
    """
    formulation = 'corotational'

    def initialize(self, nGP, nNodes, nDim):
        raise NotImplementedError("CorotContinuumKinematics: deferred to Continuum Corot WP")

    def update(self, gp, dN_dX, u_e):
        raise NotImplementedError

    def getStrain(self, gp):
        raise NotImplementedError

    def getBMatrix(self, gp):
        raise NotImplementedError

    def getGeometricStiffness(self, gp, stress):
        raise NotImplementedError

    def copy(self):
        return CorotContinuumKinematics()
```

### 4b-B. Corotational Benchmarks (Beams/Truss)

**File to create:** `src/examples/corot_benchmarks.py`

**Benchmark 5 — Snap-Through Arch (pre-snap only):**
- Shallow circular arch, rise H=0.5, half-span L=5.0
- 10 ElasticBeamColumn2d elements with CorotCrdTransf2d
- E=1e4, A=0.1, I=small (near-truss behavior), nu=0
- Displacement control on apex node
- **Pass:** Limit load P_cr within 5% of reference
- **Pass:** Solver diverges gracefully past limit point (no crash, no false equilibrium)
- **Output:** Load-displacement curve to `docs/validation/snap_through_arch.png`

**Benchmark 6 — Lee's Frame:**
- Right-angle frame, L=10 each member
- E=7.2e6, nu=0.3, b=h=0.3, I=6.75e-4
- 10 ElasticBeamColumn2d per member with CorotCrdTransf2d
- P_max=50, 50 steps of dP=1.0
- **Pass:** Tip displacements at P=50 within 2% of Simo & Vu-Quoc (u_x=23.48, u_y=-13.89)
- **Pass:** Newton convergence < 10 iterations per step
- **Output:** Load-displacement curve to `docs/validation/lees_frame.png`
- **Output:** Deformed frame at P=10,20,30,40,50 to `docs/validation/lees_frame_deformed.png`

---

## File Inventory Summary

### New Files (created in this WP)

| Phase | File | Purpose |
|-------|------|---------|
| 1 | `src/oneFEM/model/element/kinematics/base.py` | Kinematics base (moved from main.py) |
| 1 | `src/oneFEM/model/element/kinematics/crdTransf/__init__.py` | CrdTransf exports |
| 1 | `src/oneFEM/model/element/kinematics/crdTransf/base.py` | CrdTransf base (moved) |
| 1 | `src/oneFEM/model/element/kinematics/crdTransf/linear_2d.py` | (moved) |
| 1 | `src/oneFEM/model/element/kinematics/crdTransf/linear_3d.py` | (moved) |
| 1 | `src/oneFEM/model/element/kinematics/crdTransf/pdelta_2d.py` | (moved) |
| 1 | `src/oneFEM/model/element/kinematics/crdTransf/pdelta_3d.py` | (moved) |
| 1 | `src/oneFEM/model/element/kinematics/crdTransf/corot_2d.py` | (moved) |
| 1 | `src/oneFEM/model/element/kinematics/crdTransf/corot_3d.py` | (moved) |
| 1 | `src/oneFEM/model/element/kinematics/continuum/__init__.py` | Continuum kinematics exports |
| 1 | `src/oneFEM/model/element/kinematics/continuum/base.py` | ContinuumKinematics (moved) |
| 1 | `src/oneFEM/model/element/kinematics/continuum/linear.py` | LinearContinuumKinematics (moved+refactored) |
| 1 | `src/oneFEM/model/element/kinematics/shell/__init__.py` | Shell exports |
| 1 | `src/oneFEM/model/element/kinematics/shell/base.py` | ShellKinematics stub |
| 1 | `src/oneFEM/model/element/kinematics/contact/__init__.py` | Contact exports |
| 1 | `src/oneFEM/model/element/kinematics/contact/base.py` | ContactKinematics stub |
| 1 | `src/oneFEM/model/element/continuum/isoparametric.py` | Shape functions + Gauss rules |
| 1 | `tests/math/test_integration_loop_ops.py` | Math API verification |
| 1 | `tests/nodes/test_all_nodes.py` | Node validation |
| 2 | `src/oneFEM/model/element/continuum/base.py` | ContinuumElement base |
| 2 | `src/oneFEM/model/element/continuum/quad4.py` | Quad4 element |
| 2 | `src/examples/quad4_linear.py` | Patch test + Cook's membrane |
| 3 | `src/oneFEM/model/element/kinematics/continuum/total_lagrangian.py` | TL kinematics |
| 3 | `src/oneFEM/model/element/kinematics/continuum/updated_lagrangian.py` | UL kinematics |
| 3 | `src/examples/quad4_nonlinear.py` | Simple shear + cantilever + cylinder |
| 4b | `src/oneFEM/model/element/kinematics/continuum/corot.py` | CorotContinuumKinematics stub |
| 4b | `src/examples/corot_benchmarks.py` | Snap-through + Lee's frame |

### Modified Files

| Phase | File | Change |
|-------|------|--------|
| 1 | `src/oneFEM/model/node/node_2_2.py` | Full rewrite |
| 1 | `src/oneFEM/model/node/node_3_3.py` | Full rewrite |
| 1 | `src/oneFEM/model/node/node_2_4.py` | Full rewrite |
| 1 | `src/oneFEM/model/node/node_3_4.py` | Full rewrite |
| 1 | `src/oneFEM/model/node/node_3_7.py` | Full rewrite |
| 1 | `src/oneFEM/model/node/__init__.py` | Verify exports |
| 1 | `src/oneFEM/_systools/data/matrix.py` | Add `.T`, `__matmul__` |
| 1 | `src/oneFEM/_systools/data/ctensor.py` | Add `to_matrix()`, `to_vector()` |
| 1 | `src/oneFEM/_systools/data/vector.py` | Add `__iadd__` |
| 1 | `src/oneFEM/model/element/__init__.py` | Update imports |
| 1 | `src/oneFEM/model/element/kinematics/__init__.py` | Restructured exports |
| 4a | `src/oneFEM/model/element/beam/elastic_beam_column_2d.py` | Update CrdTransf import |
| 4a | `src/oneFEM/model/element/beam/elastic_beam_column_3d.py` | Update CrdTransf import |

### Deleted Files/Directories

| Phase | Path | Reason |
|-------|------|--------|
| 1/4a | `src/oneFEM/model/element/coordTransformation/` | Merged into `kinematics/crdTransf/` |
| 1 | `src/oneFEM/model/element/kinematics/main.py` | Renamed to `base.py` |
| 1 | `src/oneFEM/model/element/kinematics/continuum_main.py` | Moved to `continuum/base.py` |

---

## Critical Rules (from Questionnaire)

These rules are binding throughout all phases:

1. **One kinematics object per element** — per-GP state is internal to kinematics
2. **Element drives the integration loop** — kinematics is a pure geometry servant
3. **No raw numpy in element or kinematics files** — all via CTensor/Matrix/Vector
4. **Pre-allocate in `_domain()`, fill in-place in `_update()`**
5. **Cache in `update(gp,...)`** — B, detJ, strain cached; subsequent getters are free
6. **UL reference update in `commitState()` only** — never during Newton iteration
7. **Material interface is sacred** — do not change
8. **Write tests before proceeding to next phase**

---

*oneFEM — Unified Kinematics Implementation Plan — v1.0 — March 2026*
