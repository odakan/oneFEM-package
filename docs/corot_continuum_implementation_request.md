# CorotContinuumKinematics — Implementation Request

## Purpose

This document requests the full implementation of `CorotContinuumKinematics` for
continuum elements (Quad4 and Hex8) using the Element-Independent Corotational
Reference frame (EICR) framework of Felippa & Haugen (2005). Claude Code must read
this document in full before writing a single line of code.

**Primary reference:**
Felippa, C.A. & Haugen, B. (2005). "A unified formulation of small-strain corotational
finite elements: I. Theory." *Computer Methods in Applied Mechanics and Engineering*,
**194**(21–24), 2285–2335.

**The interface stub already exists.** The file
`oneFEM/model/element/kinematics/continuum/corot.py` was created in Phase 4b with
all methods raising `NotImplementedError`. This WP fills in those methods.

---

## Prerequisites

All of the following must PASS before implementation begins:

| Requirement | Check |
|-------------|-------|
| Quad4 B1 — Patch Test | PASS |
| Quad4 B3 — Simple Shear (TL + UL machine precision) | PASS |
| Quad4 B4 — Cantilever Rollup (TL + UL, quadratic NR) | PASS |
| Hex8 B1 — 3D Patch Test | PASS |
| Hex8 B5 — 3D Simple Shear (TL + UL machine precision) | PASS |
| Hex8 B6 — 3D Cantilever Rollup (TL + UL, quadratic NR) | PASS |

**Rationale:** `CorotContinuumKinematics` shares the same `initialize()`,
`getStrain()`, `getBMatrix()`, and `getDetJ()` infrastructure as TL and UL. If those
are not passing, Corot has nothing to build on. Do not implement Corot as a
workaround for failing TL/UL benchmarks.

---

## What EICR Is and Is Not

### What it is

EICR is a corotational framework for continuum elements that:

1. At each load step, extracts a best-fit rigid-body rotation **R** from the element
   deformation using polar decomposition of the mean deformation gradient.
2. Rotates the element to a **local corotated frame** that moves and rotates with the
   element but does not include stretching.
3. Computes **small-strain linear kinematics in the local frame** — the strain in the
   corotated frame is the infinitesimal engineering strain ε (not Green-Lagrange, not
   Almansi).
4. Evaluates the material constitutive law in the local frame (Cauchy stress σ in
   the corotated frame via linear elasticity).
5. Transforms the local forces and stiffness back to the global frame through R.

The result is a formulation that handles **arbitrary large rotations with small
strains**. This is the appropriate regime for:
- Structures undergoing large overall motion but elastic with small local deformation
- Snap-through and post-buckling where strains remain small
- Rigid-body rotation tests (result should be exactly zero stress)

### What it is NOT

- EICR is **not** a large-strain formulation. If the element stretches significantly
  (‖ε‖ > ~5%), TL or UL are more appropriate.
- EICR is **not** the same as the beam corotational transformation (`CrdTransf`).
  The beam version implicitly handles K_σ through the transformation geometry.
  EICR for continuum elements does **not** — K_σ must be computed and assembled
  explicitly (see Geometric Stiffness section below).
- EICR is **not** a Total Lagrangian or Updated Lagrangian formulation. The reference
  configuration does not move (unlike UL) and the strain is not Green-Lagrange
  (unlike TL). It is a third, distinct formulation.

---

## Theory: EICR in Detail

### Step 1 — Mean deformation gradient

For a continuum element with nGP Gauss points, compute the mean deformation gradient
by averaging over all Gauss points:

```
F_mean = (1/nGP) * sum_{gp=1}^{nGP} F[gp]
```

where each F[gp] = I + dN_dX[gp].T @ u_e (standard isoparametric computation,
identical to TL).

**Note:** Felippa (2005) also describes a volume-weighted average.
Use a simple unweighted average (equal GP weights) for this WP — it gives identical
results for the uniform integration rules used in Quad4 (2×2) and Hex8 (2×2×2).

### Step 2 — Polar decomposition of F_mean

Decompose the mean deformation gradient into a rotation R and symmetric stretch U:

```
F_mean = R @ U     (right polar decomposition)
```

Compute via SVD: if F_mean = V @ S @ W.T (numpy/scipy convention), then:

```
R = V @ W.T
U = W @ S @ W.T
```

For the 2D case, R is a 2×2 rotation matrix. For 3D, R is a 3×3 rotation matrix.

**Implementation note:** Use `scipy.linalg.svd` or `numpy.linalg.svd`. The math
module may not expose SVD — call scipy directly here. This is one justified
exception to the "no raw numpy" rule: polar decomposition of a dense 2×2 or 3×3
matrix has no math module equivalent and never involves per-GP loops.

**Numerical check:** After decomposition, verify det(R) = +1.0 to < 1e-12.
If det(R) = −1.0, there is a reflection embedded in the SVD branch — negate V[:,−1]
and recompute. Log a warning at debug level.

### Step 3 — Corotated displacements

Rotate global nodal displacements to the corotated frame:

```
u_e_local = block_diag(R.T, R.T, ...) @ u_e_global
```

where `block_diag` repeats R.T once per node DOF block (2 DOFs/node for 2D,
3 DOFs/node for 3D). Store as `self._u_e_corot[gp_call_count]`.

**Alternative formulation:** Equivalently, express the local displacements as the
difference between the rotated current position and the reference position:

```
x_current = X_ref + u_e_global          (current nodal positions)
x_corot   = R.T @ x_current - X_ref    (in corotated frame, minus reference)
```

This form is often cleaner in code. Either form gives the same result.

### Step 4 — Linear strain in corotated frame

Compute the standard small-strain B matrix and strain in the corotated frame:

```
B_local  = B_linear(dN_dX)       # same as LinearContinuumKinematics
eps_local = B_local @ u_e_corot   # small-strain ε in corotated frame
```

`dN_dX` here is computed from the **reference configuration** (fixed, same as
initialized — Corot uses a fixed reference just like TL).

This is the key simplification of EICR: once in the corotated frame, the strain
computation is identical to linear kinematics. There is no B_NL, no F^T F,
no Almansi tensor.

### Step 5 — Stress in corotated frame

Pass ε_local to the material and get Cauchy stress σ_local in the corotated frame:

```python
self._materials[gp]._setTrialStrain(eps_local)
sigma_local = self._materials[gp].getStress()     # Cauchy σ in corotated frame
```

### Step 6 — Internal force in global frame

Transform the local contribution back to global:

```
f_e = B_local.T @ sigma_local * detJ * w[gp]     (local contribution per GP)
f_global = block_diag(R, R, ...).T @ sum_{gp} f_e
         = block_diag(R, R, ...) @ f_e_total      (since block_diag(R) is orthogonal)
```

Or equivalently, rotate σ to global first:
```
sigma_global = R @ sigma_voigt_matrix @ R.T       (rotate stress tensor)
```
then use the global B and global σ directly. Either approach is correct; use
whichever is cleaner to implement given the existing B matrix machinery.

### Step 7 — Material stiffness in global frame

```
K_mat_local = sum_{gp} B_local.T @ C @ B_local * detJ * w[gp]
K_mat_global = T.T @ K_mat_local @ T
```

where T = block_diag(R, R, ...) is the full nDOF×nDOF rotation matrix.

For efficiency, note T.T @ K_mat_local @ T is a similarity transformation and can
be computed without forming T explicitly:

```python
# For each block row i and block col j:
K_global[2i:2i+2, 2j:2j+2] = R @ K_local[2i:2i+2, 2j:2j+2] @ R.T
```

This avoids forming a large (nDOF × nDOF) sparse rotation matrix.

---

## Geometric Stiffness K_σ — Critical Distinction from Beam Corot

**This is the most common implementation error for EICR on continuum elements.**

The beam corotational transformation (`CrdTransf`) implicitly includes the geometric
stiffness in the transformation algebra — the projection operator and load-correction
terms in `getGlobalStiffMatrix(kb, q)` encode K_σ without the implementer having to
add it separately.

**EICR for continuum elements does not do this.** K_σ must be computed and assembled
explicitly, exactly as in TL and UL. The formula is identical to TL:

```
K_σ[gp] = sum_I sum_J (dN_I/dX).T @ sigma_voigt @ (dN_J/dX) * detJ * w[gp]
```

where `sigma_voigt` here is the Cauchy stress resolved back to global coordinates.
The index convention is the same as in TL — see the geometric stiffness derivation
in the TL/UL implementation notes.

**Verification:** If K_σ is missing, the cantilever rollup will show linear
Newton-Raphson convergence rather than quadratic. This is the same diagnostic as for
TL/UL — use the rollup benchmark to confirm K_σ is present and correct before
running the snap-through and buckling benchmarks.

---

## Interface: Methods to Implement

The stub in `corot.py` currently raises `NotImplementedError` for all methods.
Implement the following, in this order:

### `initialize(gp_data, X_nodes, formulation)`

Pre-allocate per-GP storage. Identical structure to TL — reference configuration
is fixed and never changes (unlike UL):

```python
def initialize(self, gp_data, X_nodes, formulation):
    nGP = len(gp_data)
    nNodes, nDim = X_nodes.shape
    self._nDim    = nDim
    self._nNodes  = nNodes
    self._X_ref   = X_nodes.copy()       # reference config — NEVER modified
    self._F       = [None] * nGP         # F[gp] for mean computation
    self._B_local = [None] * nGP         # linear B in corotated frame
    self._eps     = [None] * nGP         # small strain in corotated frame
    self._detJ    = [0.0]  * nGP         # reference Jacobian determinant
    self._dN_dX   = [None] * nGP         # reference physical derivatives
    self._R       = None                 # current corotated rotation (element-level)
    self._R_committed = Matrix.eye(nDim) # committed rotation for revert
    # Pre-compute reference quantities
    for gp_idx, gp in enumerate(gp_data):
        dN_dxi = gp.dN_dxi                         # (nNodes, nDim)
        J      = dN_dxi.T @ X_nodes                # (nDim, nDim) Jacobian
        self._dN_dX[gp_idx]  = dN_dxi @ J.inv()   # (nNodes, nDim)
        self._detJ[gp_idx]   = J.det()
```

### `update(gp_idx, dN_dX, u_e)`

Main kinematics update called at each Newton iteration for each Gauss point:

```python
def update(self, gp_idx, dN_dX, u_e):
    # Step 1 — F at this GP (standard isoparametric)
    nDim = self._nDim
    F_gp = Matrix.eye(nDim)
    for I in range(self._nNodes):
        u_I = u_e[nDim*I : nDim*I + nDim]   # displacement of node I
        for i in range(nDim):
            for j in range(nDim):
                F_gp[i,j] += u_I[i] * dN_dX[I,j]
    self._F[gp_idx] = F_gp
    # Steps 2-5 are performed in _apply_corot_frame() after all GPs are updated
    # Individual GP update stores F; frame extraction happens element-wide
```

**Note:** EICR requires F_mean over all GPs before polar decomposition.
This means `update()` cannot complete the corotated strain computation
for a single GP in isolation — it needs all GPs. The element must call
`update(gp_idx, ...)` for all GPs first, then call `applyCorotFrame(u_e)` once.
Design options:

**Option A (recommended):** Add `applyCorotFrame(u_e)` as a second method called
by the element after the per-GP update loop:

```python
# In element._update():
for gp_idx in range(nGP):
    self._kinematics.update(gp_idx, dN_dX[gp_idx], u_e)
self._kinematics.applyCorotFrame(u_e)   # EICR-specific second pass
```

Add `applyCorotFrame` as a public method on `CorotContinuumKinematics` and a
no-op default on `ContinuumKinematics` base. This keeps the element loop clean.

**Option B:** Perform both passes inside a single `update(gp_idx=None, ...)` call
where `gp_idx=None` triggers the element-level step. Less clean interface.

Use Option A.

### `applyCorotFrame(u_e)` (new method)

```python
def applyCorotFrame(self, u_e):
    nGP = len(self._F)
    nDim = self._nDim
    # Step 2 — Mean F
    F_mean = Matrix.zeros(nDim, nDim)
    for F_gp in self._F:
        F_mean = F_mean + F_gp
    F_mean = F_mean * (1.0 / nGP)
    # Step 3 — Polar decomposition via SVD
    import numpy as np
    F_np = np.array([[F_mean[i,j] for j in range(nDim)] for i in range(nDim)])
    V, S, Wt = np.linalg.svd(F_np)
    R_np = V @ Wt
    if np.linalg.det(R_np) < 0:           # handle reflection branch
        V[:, -1] *= -1
        R_np = V @ Wt
    # Wrap back to math module Matrix
    self._R = Matrix([[R_np[i,j] for j in range(nDim)] for i in range(nDim)])
    # Step 4 — Corotated displacements and linear strain per GP
    for gp_idx in range(nGP):
        dN_dX = self._dN_dX[gp_idx]       # reference physical derivatives
        B = self._build_B_linear(dN_dX)   # standard linear B matrix
        self._B_local[gp_idx] = B
        # Rotate u_e to local frame: u_local[node_i_dof] = R.T @ u_global[node_i_dof]
        u_local = self._rotate_dofs(u_e, self._R.T)
        self._eps[gp_idx] = B @ u_local   # small strain in corotated frame
```

### `getStrain(gp_idx)` → CTensor

Return the small-strain CTensor (covariant, 2nd order) in the corotated frame.
Wrap `self._eps[gp_idx]` as a CTensor of the same Voigt structure used by
`LinearContinuumKinematics`.

### `getBMatrix(gp_idx)` → Matrix

Return `self._B_local[gp_idx]`. This is the standard linear B matrix in the
reference configuration — not a corotated or transformed version. The rotation
is applied at the force and stiffness assembly level, not here.

### `getDetJ(gp_idx)` → scalar

Return `self._detJ[gp_idx]`. Reference Jacobian is fixed (Corot uses a fixed
reference like TL).

### `getGeometricStiffness(gp_idx, stress)` → Matrix

Identical formula to TL. `stress` is the Cauchy stress in the global frame
(caller must rotate σ_local back before passing). The K_σ matrix size is
(nNodes·nDim) × (nNodes·nDim):

```python
def getGeometricStiffness(self, gp_idx, stress_global):
    dN_dX = self._dN_dX[gp_idx]
    nNodes, nDim = dN_dX.shape
    nDOF = nNodes * nDim
    K_geo = Matrix.zeros(nDOF, nDOF)
    # sigma_voigt to stress matrix: for 2D [sx, sy, txy] → [[sx, txy],[txy, sy]]
    sig = self._voigt_to_matrix(stress_global)
    for I in range(nNodes):
        gI = dN_dX[I, :]              # (nDim,) gradient of shape fn I
        for J in range(nNodes):
            gJ = dN_dX[J, :]
            kIJ = gI @ sig @ gJ       # scalar = g_I^T σ g_J
            for d in range(nDim):
                K_geo[nDim*I+d, nDim*J+d] += kIJ
    return K_geo   # bare, unweighted — caller applies detJ * w[gp]
```

Returns a bare (unweighted) matrix. The element integration loop applies
`* detJ * w[gp]` symmetrically with the material stiffness path.

### `commitState()` → None

Store the current rotation as the committed state:

```python
def commitState(self):
    self._R_committed = self._R.copy()
```

Corot does **not** update a reference configuration at commit (unlike UL).
The reference configuration is always the original undeformed geometry.

### `revertToLastCommit()` → None

Restore rotation to last committed state:

```python
def revertToLastCommit(self):
    self._R = self._R_committed.copy()
```

### `copy()` → CorotContinuumKinematics

Deep copy all state including `_R`, `_R_committed`, all per-GP arrays.
Follow the same pattern as TL and UL `copy()` methods.

---

## Pre-Coding Verifications

Before writing any implementation code, Claude Code must perform the following
checks and confirm each one in writing:

**V1 — Stub exists and raises NotImplementedError:**
```bash
grep -n "NotImplementedError\|def update\|def commitState\|def getStrain\|def getBMatrix\|def getDetJ\|def getGeometricStiffness" \
    oneFEM/model/element/kinematics/continuum/corot.py
```
Confirm all required methods are present. If the file does not exist, stop and
report — do not create it. It should have been created in Phase 4b.

**V2 — `applyCorotFrame` no-op default exists on base or can be added safely:**
Read `oneFEM/model/element/kinematics/continuum/base.py`. If `applyCorotFrame`
is not defined there, add a no-op default before implementing it on `CorotContinuumKinematics`:
```python
def applyCorotFrame(self, u_e):
    """No-op for non-corotational formulations."""
    pass
```

**V3 — Element calls per-GP update before accessing strain:**
Read the element integration loop in `oneFEM/model/element/continuum/quad4.py` (and
`hex8.py`). Confirm the structure is:
```
for gp in range(nGP):
    kinematics.update(gp, ...)
[optionally: kinematics.applyCorotFrame(u_e)]
for gp in range(nGP):
    strain = kinematics.getStrain(gp)
    ...
```
If the element interleaves update and getStrain in a single loop, the Corot two-pass
requirement means the element loop must be restructured. Report this before coding.

**V4 — Confirm math module SVD availability or absence:**
```python
from oneFEM.core.math import Matrix
# Try: does Matrix expose SVD?
# If not, numpy.linalg.svd is the approved fallback.
```
Confirm in writing which path will be used. Either is acceptable; document it.

**V5 — Confirm `_rotate_dofs` helper does not exist elsewhere:**
```bash
grep -rn "_rotate_dofs\|rotate_dofs\|block_diag.*R" \
    oneFEM/model/element/kinematics/continuum/
```
If an equivalent helper already exists in TL or UL, reuse it. Do not duplicate.

---

## State Machine: Trial vs. Committed

The Corot kinematics object must maintain two rotation states:

| State | Variable | Updated by | Restored by |
|-------|----------|------------|-------------|
| Trial | `self._R` | `applyCorotFrame()` at each NR iteration | `revertToLastCommit()` |
| Committed | `self._R_committed` | `commitState()` at each converged step | never (persists) |

**Initial state:** At construction and after `initialize()`, both `_R` and
`_R_committed` are identity matrices (I_nDim). The first `applyCorotFrame()` call
will overwrite `_R` with the actual rotation.

**Invariant:** After `revertToLastCommit()`, `_R == _R_committed` to machine precision.
Assert this in tests.

---

## Integration with the Element

The element's `_update()`, `getTangentStiff()`, and `getResistingForce()` methods
currently contain if/else blocks on the kinematics formulation type (or call the
kinematics polymorphically). For Corot, the element must:

1. Call `kinematics.update(gp_idx, ...)` for all GPs in a first loop.
2. Call `kinematics.applyCorotFrame(u_e)` once (no-op for non-Corot).
3. In a second loop, call `kinematics.getStrain(gp)` to get ε_local.
4. Set trial strain on material with ε_local (same path as Linear).
5. Get stress σ_local from material.
6. **Rotate σ_local to global:** `sigma_global = R @ sigma_matrix @ R.T`
7. Compute `f_gp = B.T @ sigma_global_voigt * detJ * w[gp]`
8. Apply rotation to f_e_total: `f_global = T.T @ f_e_local` (equivalent to
   assembling in local then rotating, or just using global σ directly in step 7).
9. For K_mat: `K_global += T.T @ (B.T @ C @ B) @ T * detJ * w[gp]`
   (implemented as block rotation as described in Step 7 of theory section).
10. For K_σ: `K_global += getGeometricStiffness(gp, sigma_global) * detJ * w[gp]`
    (sigma_global already in global frame from step 6).

---

## Test: Pure Rigid-Body Rotation

Before any benchmark, add this unit test to `tests/kinematics/test_corot_continuum.py`:

### Model

Single Quad4 element, unit square [0,1]². Apply a rigid-body rotation of θ degrees
as nodal displacements directly (no solver):

```python
import numpy as np
theta = np.radians(angle_deg)
R_test = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta),  np.cos(theta)]])
for node_i, (X, Y) in enumerate(ref_coords):
    x_rot = R_test @ [X, Y]
    u_nodes[node_i] = x_rot - [X, Y]
```

Run for θ = 10°, 30°, 90°, 180°.

### Pass Criteria

- `getStrain(gp)` returns ε = 0 at all GPs to < 1e-10 for all θ. (This is the
  fundamental property of EICR — it returns zero strain for pure rigid-body motion.)
- Internal force vector `getResistingForce()` returns zero vector to < 1e-10.
- `_R` extracted from `applyCorotFrame()` equals the prescribed rotation matrix R_test
  to < 1e-10.

If this test fails, EICR is broken at the frame extraction level. Fix before
running any of the following benchmarks.

---

## Validation Benchmarks

These benchmarks are already fully specified in `docs/quad4_validation_request.md`
(Benchmark 5) and `docs/hex8_validation_request.md` (Benchmarks 8 and 9).
Run them in the order listed. Do not proceed to the next benchmark if the current
one fails.

| Step | Benchmark | File | Pass Criterion |
|------|-----------|------|----------------|
| 0 | Pure rotation unit test | `tests/kinematics/test_corot_continuum.py` | ε = 0 for θ=10°,30°,90°,180° |
| 1 | Quad4 B5 — Snap-Through Arch | `examples/quad4_benchmarks.py` | P_cr within 5% of 0.5878 |
| 2 | Hex8 B8 — Column Buckling 3D | `examples/hex8_benchmarks.py` | Max R_z within 5% of 2.056 |
| 3 | Hex8 B9 — Snap-Through Arch 3D | `examples/hex8_benchmarks.py` | P_cr within 5% of 0.5878 |

**Additional cross-check — Rollup agreement with TL/UL:**

Run the Quad4 cantilever rollup (B4) and the Hex8 3D rollup (B6) with Corot
kinematics. The Corot formulation is only valid for small strains, so agreement
with TL/UL is expected only at small load levels (θ < π/4 approximately). At larger
rotations, Corot and TL/UL will diverge — this is expected and physically correct.
Document the θ at which they first diverge by more than 5% — this is the regime
boundary for Corot validity on this geometry.

**Quadratic Newton-Raphson convergence is a formal pass criterion** for all
nonlinear benchmarks. Print residual norms at each Newton iteration for the
benchmark check steps. Linear convergence = K_σ is missing or K_mat rotation
is wrong.

---

## Common Bugs and How to Diagnose Them

| Symptom | Likely Cause | Diagnostic |
|---------|--------------|------------|
| ε ≠ 0 for pure rotation | Polar decomposition reflection | Check det(R) = +1 |
| ε ≠ 0 for pure rotation | Wrong frame: R instead of R.T used on u_e | Check rotation direction |
| Nonzero stress but correct strain | Material gets ε_global, not ε_local | Confirm u_local = R.T @ u_global |
| Linear NR convergence | K_σ missing | Run with/without K_σ and compare NR rates |
| Linear NR convergence | K_mat not rotated (K_global ≠ T.T @ K_local @ T) | Compare K against numerical finite difference |
| Results correct at small loads, wrong at large | Committed R not updated | Confirm commitState() updates _R_committed |
| Divergence on revert | _R not restored from _R_committed | Check revertToLastCommit() copies, not aliases |
| F_mean computation wrong | Averaging before polar decomp omitted | Print F_mean and check det(F_mean) ≈ 1 for near-rigid motion |
| 2D vs 3D mismatch | Block-diagonal R rotation applied in wrong dimension | Confirm nDim is read from X_nodes.shape |

---

## File and Class Inventory

**Files to modify:**

| File | Change |
|------|--------|
| `oneFEM/model/element/kinematics/continuum/corot.py` | Implement all methods (primary deliverable) |
| `oneFEM/model/element/kinematics/continuum/base.py` | Add `applyCorotFrame()` no-op default |
| `oneFEM/model/element/continuum/quad4.py` | Add `applyCorotFrame()` call between update loops |
| `oneFEM/model/element/continuum/hex8.py` | Same as Quad4 |

**Files to create:**

| File | Purpose |
|------|---------|
| `tests/kinematics/test_corot_continuum.py` | Pure rotation unit test (required before benchmarks) |

**Files NOT to modify:** TL, UL, Linear kinematics classes. This WP is additive only —
it fills in the existing stub. No changes to passing TL/UL benchmarks.

---

## Implementation Order (strict)

1. Read this document in full.
2. Perform all 5 Pre-Coding Verifications (V1–V5). Report results.
3. Add `applyCorotFrame()` no-op to `ContinuumKinematics` base.
4. Add two-pass structure to Quad4 and Hex8 element loops.
5. Implement `initialize()`.
6. Implement `applyCorotFrame()` with polar decomposition + corotated strain.
7. Implement `getStrain()`, `getBMatrix()`, `getDetJ()`.
8. Implement `getGeometricStiffness()`.
9. Implement `commitState()` and `revertToLastCommit()`.
10. Implement `copy()`.
11. Write and run pure rotation unit test. **MUST PASS before any benchmark.**
12. Run Quad4 B5 (snap-through). **MUST PASS before Hex8.**
13. Run Hex8 B8 (column buckling).
14. Run Hex8 B9 (snap-through 3D).
15. Run rollup cross-check. Document Corot validity regime.
16. Full regression (all existing benchmarks must still pass).

---

## Full Regression After All Benchmarks Pass

```
examples/truss.py
examples/beam.py
examples/column_buckling.py
examples/corot_benchmarks.py
examples/quad4_benchmarks.py
examples/hex8_benchmarks.py
```

Zero regressions required. Corot implementation is additive — it must not affect
any existing TL, UL, or Linear benchmark result.

---

## Benchmark Status Table

Update in-place as each benchmark passes.

| # | Benchmark | Status | Notes |
|---|-----------|--------|-------|
| V1–V5 | Pre-coding verifications | ⬜ Pending | All 5 required before coding |
| 0 | Pure rotation unit test | ⬜ Pending | ε=0 for θ=10°,30°,90°,180° |
| 1 | Quad4 B5 — Snap-Through Arch | ⬜ Pending | P_cr within 5% |
| 2 | Hex8 B8 — Column Buckling 3D | ⬜ Pending | P_cr within 5% |
| 3 | Hex8 B9 — Snap-Through Arch 3D | ⬜ Pending | P_cr within 5% |
| 4 | Rollup cross-check | ⬜ Pending | Document validity regime |
| 5 | Full regression | ⬜ Pending | Zero regressions |

---

## References

- Felippa, C.A. & Haugen, B. (2005). "A unified formulation of small-strain
  corotational finite elements: I. Theory." *CMAME* **194**(21–24), 2285–2335.
  — Primary reference. Sections 2–5 cover EICR theory, polar decomposition,
  and the corotated frame construction.

- Felippa, C.A. (2000). "A systematic approach to the element-independent corotational
  dynamics of finite elements." Technical Report CU-CAS-00-03, University of Colorado.
  — More accessible derivation with explicit implementation pseudocode.

- Nour-Omid, B. & Rankin, C.C. (1991). "Finite rotation analysis and consistent
  linearization using projectors." *CMAME* **93**(3), 353–384.
  — Original EICR formulation; useful for understanding K_σ construction.

- Crisfield, M.A. (1991). *Non-Linear Finite Element Analysis of Solids and Structures*,
  Vol. 1, Ch. 7. — Alternative corotational derivation; good for cross-checking K_σ.

- Bathe, K.J. (1996). *Finite Element Procedures*, Ch. 6. — TL/UL reference for
  comparison with Corot strain measures.

- Existing oneFEM: `examples/corot_benchmarks.py`, `examples/column_buckling.py` —
  beam corotational benchmarks that Corot-continuum results should qualitatively match.
