# CorotContinuumKinematics (EICR) — Implementation Plan

## Context

The `CorotContinuumKinematics` stub (`corot.py`) was created in Phase 4b with all
methods raising `NotImplementedError`. This WP fills in those methods using the
Element-Independent Corotational Reference (EICR) framework of Felippa & Haugen (2005).

**Specification:** `docs/corot_continuum_implementation_request.md`

**Primary reference implementation:** `docs/reference/ASDShellQ4CorotationalTransformation.h`
(Petracca, ASDEA/OpenSees). The continuum implementation is a direct simplification of
this shell implementation. Read it before writing any code.

EICR extracts a best-fit rigid rotation **R** from the element deformation via polar
decomposition, then computes small-strain kinematics in a corotated frame. This gives
zero strain for pure rigid-body rotations and handles large rotations with small strains
— the correct regime for snap-through, post-buckling, and structures with large overall
motion but elastic small local deformation.

**Key results from Petracca source (affect Phase 2 directly):**

1. **2D frame extraction uses an exact closed-form formula, not SVD.** Petracca computes
   the polar decomp rotation angle analytically as `alpha = atan2(f21−f12, f11+f22)`,
   where f11..f22 are the in-plane deformation gradient components evaluated at the
   centroid of the bilinear Quad4 via the area-coordinate mapping. This is cheaper than
   SVD, has no branch-cut issues, and is the correct formula for Quad4. SVD is only
   correct for 3D (Hex8).

2. **G = 0 exactly for polar-decomp frame alignment.** The Petracca source confirms:
   *"when attaching the local system using the polar decomposition, it turns out that
   the G matrix should be exactly 0"*. For continuum elements (no rotational DOFs),
   H = I and P = I as well. The full EICR tangent collapses to `T'·K_mat·T + K_sigma`.
   No P/S/H/G machinery is needed.

3. **Deformational local displacements require proper centring.** Petracca's
   `calculateLocalDisplacements()` centres positions at the current centroid before
   rotating: `u_local[I] = R.T @ (x_cur[I] − C_cur) − (X_ref[I] − C0)`. Simply
   applying `R.T @ u_global[I]` is wrong — it omits the centring and fails the pure
   rotation test.

---

## Prerequisites

All of the following must PASS before implementation begins:

| Requirement | Status |
|-------------|--------|
| Quad4 B1 — Patch Test | PASS |
| Quad4 B3 — Simple Shear (TL + UL) | PASS |
| Quad4 B4 — Cantilever Rollup (TL + UL) | PASS |
| Hex8 B1 — 3D Patch Test | PASS |
| Hex8 B5 — 3D Simple Shear (TL + UL) | Not yet implemented |
| Hex8 B6 — 3D Cantilever Rollup (TL + UL) | Not yet implemented |

**Gate:** Hex8 TL/UL benchmarks (B5, B6) must pass before Corot implementation starts.
These will exercise the 3D nonlinear kinematics path and flush out the `_buildKgeo` bug
documented in Phase 0 below.

---

## Files to Modify

| File | Change |
|------|--------|
| `src/oneFEM/model/element/kinematics/continuum/nonlinear_base.py` | Fix `_buildKgeo` 3D Voigt index bug (lines 128-130) |
| `src/oneFEM/model/element/kinematics/continuum/base.py` | Add `applyCorotFrame()` and `transformToGlobal()` no-ops |
| `src/oneFEM/model/element/continuum/base.py` | Split `_update()` loop, separate K_mat/K_geo in `_buildStiffnessAndForce()` |
| `src/oneFEM/model/element/kinematics/continuum/corot.py` | Full EICR implementation (replace entire file) |

## Files to Create

| File | Purpose |
|------|---------|
| `tests/kinematics/test_corot_continuum.py` | Pure rotation unit test (gate before benchmarks) |
| `src/examples/corot_continuum_benchmarks.py` | Snap-through + buckling benchmarks |

---

## Pre-Coding Verifications

Confirm each in writing before writing any implementation code.

**V1 — Stub exists and raises NotImplementedError:**
```bash
grep -n "NotImplementedError\|def update\|def commitState\|def getStrain\|def getBMatrix\|def getDetJ\|def getGeometricStiffness" \
    src/oneFEM/model/element/kinematics/continuum/corot.py
```

**V2 — How `initialize()` receives node coordinates:**
Read how TL and UL kinematics objects are initialized from the element. Confirm whether
`X_ref` (reference node coordinates, shape `(nNodes, nDim)`) is already passed as a
parameter or kwargs. If not, identify where in the element the coordinates live and how
to pass them. The Corot implementation requires `X_ref` and `C0 = mean(X_ref)` — these
must be stored in `initialize()`.

**V3 — ASD reference files are present:**
```bash
ls docs/reference/ASDShellQ4CorotationalTransformation.h \
       docs/reference/ASDShellQ4LocalCoordinateSystem.h \
       docs/reference/ASDShellQ4Transformation.h \
       docs/reference/ASDShellQ4.cpp
```

**V4 — `_NonlinearContinuumBase` exposes what is needed, and `_F[gp]` return type:**
```bash
grep -n "_computeH\|_computeF\|_buildKgeo\|_F\[\|return.*F\b" \
    src/oneFEM/model/element/kinematics/continuum/nonlinear_base.py | head -30
```
Confirm `_computeH(gp, u_e)`, `_computeF(H)`, `_buildKgeo(gp, stress)`, and `self._F`
all exist with the expected signatures.

**Critical:** Determine whether `_computeF(H)` returns a `Matrix`-like object with a
`.data` attribute, or a raw numpy array. `_extract_R_3d()` calls `self._F[gp].data`.
If `_computeF` returns a raw numpy array, change the accumulation line to:

```python
F_mean += self._F[gp]          # if _F[gp] is already numpy
# instead of:
F_mean += self._F[gp].data     # only if _F[gp] is a Matrix wrapper
```

Report the return type during V4 and adjust the code accordingly before writing Phase 2.

**V5 — `LinearContinuumKinematics._buildBMatrix` is a static/class method:**
```bash
grep -n "_buildBMatrix\|@staticmethod\|@classmethod" \
    src/oneFEM/model/element/kinematics/continuum/linear.py | head -10
```
Confirm it is callable as `LinearContinuumKinematics._buildBMatrix(dN_dX, nDim)`.

**V6 — Quad4 node ordering matches Petracca's assumption:**

The C1..C9 coefficients in `_extract_R_2d()` are transcribed directly from Petracca's
shell, which assumes a **counter-clockwise** node numbering:

```
4 --- 3
|     |
1 --- 2
```

Verify oneFEM's Quad4 uses the same CCW ordering by checking the reference:
```bash
grep -n "node\|isopar\|xi\|eta\|shape" \
    src/oneFEM/model/element/continuum/quad4.py | head -20
```

If the ordering is different (e.g., 1-2-4-3 or CW), the C1..C9 formula will give a
wrong rotation angle. The pure rotation test will catch this, but it must be verified
before attributing any failure to something else. If ordering differs, remap indices
in `_extract_R_2d()` before the coefficient computation.

---



**File:** `src/oneFEM/model/element/kinematics/continuum/nonlinear_base.py`

### The Bug

The 3D branch maps stress Voigt indices to the wrong positions in the stress tensor
matrix. The oneFEM Voigt order is `[xx, yy, zz, xy, yz, xz]` (positions 0-5), confirmed
by the B matrix row order in `linear.py` (line 178: row 3 = gamma_xy) and the CTensor
header comment `[sigma11, sigma22, sigma33, sigma12, sigma23, sigma13]`.

### Exact Edit

Find this text (lines 128-130):
```
            S_mat[1, 2] = S_mat[2, 1] = S_vec[3] / 2.0
            S_mat[0, 2] = S_mat[2, 0] = S_vec[4] / 2.0
            S_mat[0, 1] = S_mat[1, 0] = S_vec[5] / 2.0
```

Replace with:
```
            S_mat[0, 1] = S_mat[1, 0] = S_vec[3] / 2.0
            S_mat[1, 2] = S_mat[2, 1] = S_vec[4] / 2.0
            S_mat[0, 2] = S_mat[2, 0] = S_vec[5] / 2.0
```

### Impact

**Dormant bug.** All existing benchmarks are unaffected because:
- 2D Quad4: uses the 2D branch (only 3 Voigt components, correct)
- 3D Hex8: only tested with `LinearContinuumKinematics` which returns `K_geo = zero Matrix`
- The 3D K_geo path (TL/UL/Corot on Hex8) has never been exercised

---

## Phase 1: Backward-Compatible Scaffolding

All changes in this phase are behavioral no-ops for existing formulations.

### Step 1.1 — Add `applyCorotFrame()` and `transformToGlobal()` no-ops

**File:** `src/oneFEM/model/element/kinematics/continuum/base.py`

Find this text (lines 73-79 — the `update` method):
```
    def update(self, gp, u_e):
        """Update GP state for current trial displacements.

        :param gp: Gauss point index
        :param u_e: Vector — element nodal displacements (flat, nDOF)
        """
        pass
```

Replace with:
```
    def update(self, gp, u_e):
        """Update GP state for current trial displacements.

        :param gp: Gauss point index
        :param u_e: Vector — element nodal displacements (flat, nDOF)
        """
        pass

    def applyCorotFrame(self, u_e):
        """Apply corotational frame extraction. No-op for non-corotational."""
        pass

    def transformToGlobal(self, K_mat, f):
        """Transform material stiffness and force to global frame.
        No-op for non-corotational. Returns (K_mat, f) unchanged."""
        return K_mat, f
```

Linear, TL, UL all inherit these as no-ops. Zero behavioral change.

### Step 1.2 — Split element `_update()` into two passes

**File:** `src/oneFEM/model/element/continuum/base.py`

EICR requires all F[gp] computed before polar decomposition. The current single loop
interleaves `update()` and `getStrain()` per GP. Split into two passes.

**Edit 1.2a — `_update()` method (lines 125-128):**

Find:
```
        for gp in range(len(self._gp_data)):
            self._kinematics.update(gp, u_e)
            strain = self._kinematics.getStrain(gp)
            self._materials[gp]._setTrialStrain(strain)
```

Replace with:
```
        # Pass 1: update kinematics at all GPs
        for gp in range(len(self._gp_data)):
            self._kinematics.update(gp, u_e)
        self._kinematics.applyCorotFrame(u_e)

        # Pass 2: push strain to materials
        for gp in range(len(self._gp_data)):
            strain = self._kinematics.getStrain(gp)
            self._materials[gp]._setTrialStrain(strain)
```

**Edit 1.2b — `_domain()` initial K computation (lines 104-105):**

Find:
```
        for gp in range(nGP):
            self._kinematics.update(gp, u_zero)
            B = self._kinematics.getBMatrix(gp)
```

Replace with:
```
        for gp in range(nGP):
            self._kinematics.update(gp, u_zero)
        self._kinematics.applyCorotFrame(u_zero)
        for gp in range(nGP):
            B = self._kinematics.getBMatrix(gp)
```

**Edit 1.2c — `getInitialStiff()` (lines 207-208):**

Find:
```
        for gp in range(len(self._gp_data)):
            self._kinematics.update(gp, u_zero)
            B = self._kinematics.getBMatrix(gp)
```

Replace with:
```
        for gp in range(len(self._gp_data)):
            self._kinematics.update(gp, u_zero)
        self._kinematics.applyCorotFrame(u_zero)
        for gp in range(len(self._gp_data)):
            B = self._kinematics.getBMatrix(gp)
```

**Backward compatible:** For Linear/TL/UL, `update()` caches strain internally, so
`getStrain()` in pass 2 returns the same value. `applyCorotFrame()` is a no-op.

### Step 1.3 — Separate K_mat and K_geo in `_buildStiffnessAndForce()`

**File:** `src/oneFEM/model/element/continuum/base.py`

**Why:** K_mat (material stiffness) needs rotation to global for Corot, but K_geo
(geometric stiffness computed from global stress) does not. Must accumulate separately.

Find the entire method body (lines 133-155):
```
    def _buildStiffnessAndForce(self, u_e):
        """Assemble element K and f from all GPs."""
        nDOF = self._nDOF_total
        t = self._thickness

        K = Matrix(shape=[nDOF, nDOF])
        f = Vector(shape=nDOF)

        for gp in range(len(self._gp_data)):
            B = self._kinematics.getBMatrix(gp)
            C_mat = self._materials[gp].getTangent().to_matrix()
            sig_vec = self._materials[gp].getStress().to_vector()
            detJ, w = self._gp_data[gp]
            dV = detJ * w * t

            K += B.T @ C_mat @ B * dV
            f += B.T @ sig_vec * dV

            K_geo = self._kinematics.getGeometricStiffness(gp, self._materials[gp].getStress())
            K += K_geo * dV

        self._k = K
        self._f = f
```

Replace with:
```
    def _buildStiffnessAndForce(self, u_e):
        """Assemble element K and f from all GPs."""
        nDOF = self._nDOF_total
        t = self._thickness

        K_mat = Matrix(shape=[nDOF, nDOF])
        K_geo = Matrix(shape=[nDOF, nDOF])
        f = Vector(shape=nDOF)

        for gp in range(len(self._gp_data)):
            B = self._kinematics.getBMatrix(gp)
            C_mat = self._materials[gp].getTangent().to_matrix()
            sig_vec = self._materials[gp].getStress().to_vector()
            detJ, w = self._gp_data[gp]
            dV = detJ * w * t

            K_mat += B.T @ C_mat @ B * dV
            f += B.T @ sig_vec * dV

            Kg = self._kinematics.getGeometricStiffness(gp, self._materials[gp].getStress())
            K_geo += Kg * dV

        K_mat, f = self._kinematics.transformToGlobal(K_mat, f)
        self._k = K_mat + K_geo
        self._f = f
```

**Backward compatible:** For Linear/TL/UL, `transformToGlobal()` returns (K_mat, f)
unchanged, and `K_mat + K_geo` gives the same result as accumulating into a single K.

### Step 1.4 — REGRESSION GATE

Run ALL existing benchmarks after scaffolding changes. Zero changes expected. **Strict
gate before Phase 2.**

```bash
cd src
python truss.py
python dynamic_truss.py
python eigen_truss.py
python epp_truss.py
python examples/beam.py
python examples/column_buckling.py
python examples/corot_benchmarks.py
python examples/quad4_patch_test.py
python examples/quad4_cooks_membrane.py
python examples/quad4_simple_shear.py
python examples/quad4_cantilever_rollup.py
python examples/hex8_benchmarks.py
```

---

## Phase 2: Implement CorotContinuumKinematics

Replace the entire file `src/oneFEM/model/element/kinematics/continuum/corot.py` with the following complete implementation.

**Read `docs/reference/ASDShellQ4CorotationalTransformation.h` before writing any code.**
The three methods `createLocalCoordinateSystem()`, `calculateLocalDisplacements()`, and
`RotationGradient()` are the direct templates for `_extract_R_2d()`, `_get_u_local()`,
and the G=0 result respectively.

```python
##-----------------------------------------------------------------------##
#                                                                         #
#        #--oneFEM--#: One FEM software in a galaxy far far away          #
#                                                                         #
#                   Computational Mechanics 2022                          #
#                       University of Pavia                               #
#               Written by: Onur Deniz AKAN, IUSS Pavia                   #
#                         15 January 2022                                 #
#                                                                         #
##-----------------------------------------------------------------------##
#
# Author: Onur Deniz Akan
# Date: 05/03/2026
# Version: 1.0
#
# CorotContinuumKinematics (EICR — Felippa & Haugen 2005)
#   Corotational formulation for continuum elements.
#
#   Frame extraction follows Petracca (ASDEA/OpenSees),
#   ASDShellQ4CorotationalTransformation.h, USE_POLAR_DECOMP_ALLIGN path.
#
#   2D (Quad4): exact closed-form polar decomp via atan2 formula.
#   3D (Hex8):  polar decomp of mean F via SVD.
#
#   Because G = 0 exactly for polar-decomp frame alignment (confirmed in
#   Petracca source), the tangent reduces to T'*K_mat*T + K_sigma with no
#   P/S/H/G spin machinery needed.

import math
from .nonlinear_base import _NonlinearContinuumBase
from .linear import LinearContinuumKinematics
from ....._systools.data import Vector, Matrix
from ....._systools.data.ctensor import CTensor
from ....._systools.backend import np


class CorotContinuumKinematics(_NonlinearContinuumBase):
    """Corotational (EICR) kinematics for continuum elements.

    Extracts rigid rotation R per element via polar decomposition of the
    element deformation, applies deformation in a locally corotated frame.
    Zero strain for pure rigid-body rotations (translation + rotation).

    Inherits from _NonlinearContinuumBase for:
      - _computeH(gp, u_e): displacement gradient at GP
      - _computeF(H): deformation gradient F = I + H
      - _buildKgeo(gp, stress): geometric stiffness K_sigma
      - Per-GP _F, _strain arrays

    Uses LinearContinuumKinematics._buildBMatrix() for constant linear B.
    """

    formulation = 'corotational'

    def initialize(self, nGP, nDim, nNodes, dN_dX_list, X_ref=None, **kwargs):
        """Initialize per-GP arrays, pre-build linear B matrices, init rotation.

        :param X_ref: numpy array (nNodes, nDim) — reference node coordinates.
                      Required for the correct deformational displacement formula.
                      Must be provided; pass from element reference coordinates.
        """
        super().initialize(nGP, nDim, nNodes, dN_dX_list, **kwargs)

        # Store reference node coordinates and centroid.
        # These are needed in _get_u_local() to centre positions before rotating,
        # following Petracca's calculateLocalDisplacements() exactly.
        if X_ref is None:
            raise ValueError(
                "CorotContinuumKinematics.initialize: X_ref (reference node "
                "coordinates) must be provided. Pass the element's reference "
                "coordinate array of shape (nNodes, nDim).")
        self._X_ref = np.array(X_ref, dtype=float)       # (nNodes, nDim)
        self._C0    = self._X_ref.mean(axis=0)            # (nDim,) reference centroid

        # Pre-build constant linear B matrices per GP (reference config, never changes).
        self._B_local = []
        for gp in range(nGP):
            self._B_local.append(
                LinearContinuumKinematics._buildBMatrix(dN_dX_list[gp], nDim))

        # Rotation state stored as raw numpy arrays for efficiency.
        self._R           = np.eye(nDim)
        self._R_committed = np.eye(nDim)

    # ------------------------------------------------------------------
    # EICR frame extraction helpers
    # ------------------------------------------------------------------

    def _extract_R_2d(self, u_e_np):
        """Extract 2D corotational rotation via Petracca's exact atan2 formula.

        Direct translation of createLocalCoordinateSystem() (USE_POLAR_DECOMP_ALLIGN)
        from ASDShellQ4CorotationalTransformation.h.

        alpha = atan2(f21 - f12, f11 + f22)
        where f11..f22 are the 2D deformation gradient at the element centroid.

        No SVD — closed-form, no branch-cut issues, exact for Quad4.

        :param u_e_np: (nDOF,) numpy array of nodal displacements
        :return: (2,2) numpy rotation matrix R
        """
        # Reference centred coords (aX_i, aY_i)
        aX = self._X_ref[:, 0] - self._C0[0]
        aY = self._X_ref[:, 1] - self._C0[1]

        # Current centred coords (bX_i, bY_i)
        x_cur = self._X_ref + u_e_np.reshape(self._nNodes, 2)
        C_cur = x_cur.mean(axis=0)
        bX = x_cur[:, 0] - C_cur[0]
        bY = x_cur[:, 1] - C_cur[1]

        aX1, aX2, aX3, aX4 = aX[0], aX[1], aX[2], aX[3]
        aY1, aY2, aY3, aY4 = aY[0], aY[1], aY[2], aY[3]
        bX1, bX2, bX3, bX4 = bX[0], bX[1], bX[2], bX[3]
        bY1, bY2, bY3, bY4 = bY[0], bY[1], bY[2], bY[3]

        # Petracca C1..C9 coefficients — copied verbatim from the ASD shell source.
        # C1 is the reciprocal of a reference-configuration area-related determinant.
        C1 = 1.0 / (aX1*aY2 - aX2*aY1 - aX1*aY4 + aX2*aY3
                   - aX3*aY2 + aX4*aY1 + aX3*aY4 - aX4*aY3)
        C2 = (bY1 + bY2 - bY3 - bY4) / 4.0
        C3 = (bY1 - bY2 - bY3 + bY4) / 4.0
        C4 = (bX1 + bX2 - bX3 - bX4) / 4.0
        C5 = (bX1 - bX2 - bX3 + bX4) / 4.0
        C6 =  aX1 + aX2 - aX3 - aX4
        C7 =  aX1 - aX2 - aX3 + aX4
        C8 =  aY1 + aY2 - aY3 - aY4
        C9 =  aY1 - aY2 - aY3 + aY4

        # 2D deformation gradient components at element centroid
        f11 = 2.0 * C1 * (C5*C8 - C4*C9)
        f12 = 2.0 * C1 * (C4*C7 - C5*C6)
        f21 = 2.0 * C1 * (C3*C8 - C2*C9)
        f22 = 2.0 * C1 * (C2*C7 - C3*C6)

        # Rotation angle from polar decomp: F = R*U → alpha = rotation of R
        alpha = math.atan2(f21 - f12, f11 + f22)
        ca, sa = math.cos(alpha), math.sin(alpha)
        return np.array([[ca, -sa], [sa,  ca]])

    def _extract_R_3d(self):
        """Extract 3D corotational rotation via SVD of mean deformation gradient.

        F_mean = (1/nGP) * sum_gp F[gp]
        F_mean = V @ diag(S) @ Wt  →  R = V @ Wt  (polar decomposition)

        Handles SVD reflection branch (det(R) < 0) by negating last column of V.

        :return: (3,3) numpy rotation matrix R
        """
        F_mean = np.zeros((3, 3))
        for gp in range(self._nGP):
            # _F[gp] may be a Matrix wrapper or a raw numpy array depending on
            # what _computeF() returns (confirmed during V4).
            # Use whichever form is correct — do NOT guess. Verify during V4.
            f_gp = self._F[gp]
            F_mean += f_gp.data if hasattr(f_gp, 'data') else f_gp
        F_mean /= self._nGP

        V, S, Wt = np.linalg.svd(F_mean)
        R = V @ Wt
        if np.linalg.det(R) < 0:    # handle reflection branch
            V[:, -1] *= -1
            R = V @ Wt
        return R

    def _get_u_local(self, u_e_np, R, x_cur=None):
        """Compute deformational local displacements in corotated frame.

        Follows Petracca's calculateLocalDisplacements() with rotational DOF
        terms removed (continuum elements have translational DOFs only).

        Formula:  u_local[I] = R.T @ (x_cur[I] - C_cur) - (X_ref[I] - C0)

        where C_cur is the centroid of current node positions.
        The centring step removes rigid-body translation; R.T removes rotation.
        Together they isolate the pure deformational displacement.

        :param u_e_np: (nDOF,) numpy array of global nodal displacements
        :param R: (nDim, nDim) numpy corotational rotation matrix
        :param x_cur: optional precomputed (nNodes, nDim) current positions
        :return: (nDOF,) numpy array of local deformational displacements
        """
        nDim = self._nDim
        if x_cur is None:
            x_cur = self._X_ref + u_e_np.reshape(self._nNodes, nDim)
        C_cur = x_cur.mean(axis=0)

        u_local = np.zeros(self._nNodes * nDim)
        for I in range(self._nNodes):
            X0_I = self._X_ref[I] - self._C0         # centred reference position
            x_I  = x_cur[I]       - C_cur            # centred current position
            u_local[nDim*I : nDim*I+nDim] = R.T @ x_I - X0_I
        return u_local

    # ------------------------------------------------------------------
    # Main kinematics interface
    # ------------------------------------------------------------------

    def update(self, gp, u_e):
        """Cache F[gp] for later use in 3D polar decomp.

        For 2D (Quad4): F is not used in frame extraction (atan2 formula uses
        node coordinates directly), but is computed here for consistency and
        in case getF() is called externally.

        For 3D (Hex8): F[gp] is averaged in applyCorotFrame() to get F_mean
        for SVD polar decomp.
        """
        H = self._computeH(gp, u_e)
        self._F[gp] = self._computeF(H)

    def applyCorotFrame(self, u_e):
        """Core EICR: extract R, compute deformational local displacements,
        compute corotated linear strain at all GPs.

        Called once per Newton iteration after all per-GP update() calls.
        Follows the sequence in ASDShellQ4.cpp calculateAll():
          createLocalCoordinateSystem(UG)  →  extract R
          calculateLocalDisplacements(...)  →  _get_u_local(...)
          Gauss loop: E = B * UL            →  strain[gp] = B_local[gp] @ u_local
        """
        nDim    = self._nDim
        u_e_np  = u_e.data if hasattr(u_e, 'data') else np.array(u_e)

        # Step 1 — Extract corotational rotation R
        if nDim == 2:
            # Petracca exact atan2 formula (Quad4) — no SVD, no branch issues
            R = self._extract_R_2d(u_e_np)
        else:
            # SVD of mean F (Hex8)
            R = self._extract_R_3d()
        self._R = R

        # Step 2 — Deformational local displacements (Petracca's formula with centring)
        x_cur = self._X_ref + u_e_np.reshape(self._nNodes, nDim)
        u_local_np = self._get_u_local(u_e_np, R, x_cur=x_cur)

        # Step 3 — Linear strain per GP in corotated frame
        u_local_vec = Vector(u_local_np)
        for gp in range(self._nGP):
            eps_vec = self._B_local[gp] @ u_local_vec
            self._strain[gp] = CTensor(eps_vec.data.tolist(),
                                       self._nVoigt, CTensor.COV)

    def getStrain(self, gp):
        """Return cached corotated strain CTensor (2nd order, COV)."""
        return self._strain[gp]

    def getBMatrix(self, gp):
        """Return constant linear B matrix (reference config, never changes)."""
        return self._B_local[gp]

    def getF(self, gp):
        """Return deformation gradient at GP (used externally if needed)."""
        return self._F[gp]

    def getDetJ(self, gp):
        """Not used for Corot — detJ is read from gp_data by the element directly.

        Raises NotImplementedError to fail loudly if called unexpectedly, rather
        than returning None and silently producing wrong results downstream.
        If any code path calls getDetJ() polymorphically and requires a float,
        that path must be updated to read detJ from gp_data instead.
        """
        raise NotImplementedError(
            "CorotContinuumKinematics.getDetJ: detJ is fixed at the reference "
            "configuration and is read from gp_data by the element, not from "
            "the kinematics object. Check the call site."
        )

    def getGeometricStiffness(self, gp, stress):
        """Build geometric stiffness K_sigma using stress rotated to global frame.

        The material returns stress in the corotated frame (sigma_local).
        K_sigma is computed from reference dN_dX (global frame coordinates),
        so stress must be rotated to global first.

        K_sigma is NOT rotated by transformToGlobal() — only K_mat is rotated.
        This is correct: K_sigma is already in the global frame by construction.

        Voigt index convention (must match _buildKgeo in nonlinear_base.py):
          2D: [sigma_xx, sigma_yy, gamma_xy]  (gamma = 2*epsilon)
          3D: [sigma_xx, sigma_yy, sigma_zz, gamma_xy, gamma_yz, gamma_xz]
        """
        S_vec = stress.make_vector()
        S_local = np.zeros((self._nDim, self._nDim))
        if self._nDim == 2:
            S_local[0, 0] = S_vec[0]
            S_local[1, 1] = S_vec[1]
            S_local[0, 1] = S_local[1, 0] = S_vec[2] / 2.0
        else:
            S_local[0, 0] = S_vec[0]
            S_local[1, 1] = S_vec[1]
            S_local[2, 2] = S_vec[2]
            S_local[0, 1] = S_local[1, 0] = S_vec[3] / 2.0
            S_local[1, 2] = S_local[2, 1] = S_vec[4] / 2.0
            S_local[0, 2] = S_local[2, 0] = S_vec[5] / 2.0

        # Rotate stress to global: sigma_global = R @ sigma_local @ R.T
        S_global = self._R @ S_local @ self._R.T

        # Wrap back as CTensor (CONTR) — engineering shear (factor 2 on off-diagonals)
        if self._nDim == 2:
            sig_data = [S_global[0, 0], S_global[1, 1],
                        2.0 * S_global[0, 1]]
        else:
            sig_data = [S_global[0, 0], S_global[1, 1], S_global[2, 2],
                        2.0 * S_global[0, 1],
                        2.0 * S_global[1, 2],
                        2.0 * S_global[0, 2]]
        sig_global = CTensor(sig_data, self._nVoigt, CTensor.CONTR)

        return self._buildKgeo(gp, sig_global)

    def transformToGlobal(self, K_mat, f):
        """Rotate K_mat and f from corotated frame to global frame.

        T = block_diag(R, R, ..., R)  — block-diagonal rotation, one R per node.
        K_global[I,J] = R @ K_local[I,J] @ R.T     (each nDim x nDim block)
        f_global[I]   = R @ f_local[I]

        Note on sign convention for f:
          f_local[I]  = B.T @ sigma_local  (force in corotated frame)
          f_global[I] = R @ f_local[I]     (rotate to global)
        R here is the rotation that maps local vectors to global, which is the
        same R extracted from the polar decomposition (local->global direction).
        """
        R    = self._R
        nDim = self._nDim

        f_data = f.data.copy()
        for I in range(self._nNodes):
            sl = slice(nDim * I, nDim * I + nDim)
            f_data[sl] = R @ f_data[sl]

        K_data = K_mat.data.copy()
        for I in range(self._nNodes):
            for J in range(self._nNodes):
                si = slice(nDim * I, nDim * I + nDim)
                sj = slice(nDim * J, nDim * J + nDim)
                K_data[si, sj] = R @ K_data[si, sj] @ R.T

        return Matrix(init=K_data), Vector(f_data)

    def commitState(self, **kwargs):
        """Commit rotation. Corot does NOT update reference config (unlike UL).

        Calls super().commitState() first in case _NonlinearContinuumBase manages
        any shared state (B_NL arrays, _F cache, etc.). Verify during V4 whether
        the parent's commitState() does anything — if it is a no-op, the super()
        call is harmless; if it manages state, omitting it would be a bug.
        """
        super().commitState(**kwargs)
        self._R_committed = self._R.copy()

    def revertToLastCommit(self):
        """Revert to last committed rotation."""
        self._R = self._R_committed.copy()

    def copy(self):
        """Return a deep copy of this kinematics object, including rotation state.

        Preserves _R and _R_committed so that copy() called mid-solution
        (e.g., for line search or parallel assembly) retains the committed state.
        """
        c = CorotContinuumKinematics()
        if hasattr(self, '_nGP') and self._nGP is not None:
            # Confirm the attribute name matches what the base class stores.
            # It may be _dN_dX_list rather than _dN_dX — check during V4 and fix here.
            dN_dX_arg = self._dN_dX if hasattr(self, '_dN_dX') else self._dN_dX_list
            c.initialize(self._nGP, self._nDim, self._nNodes,
                         dN_dX_arg,
                         X_ref=self._X_ref.copy())
            # initialize() resets _R to eye — overwrite with current state
            c._R           = self._R.copy()
            c._R_committed = self._R_committed.copy()
        return c
```

**NOTE:** The `__init__.py` for the kinematics/continuum package already exports
`CorotContinuumKinematics` — no changes needed there.

### Element call-site change for `initialize()`

The element that creates `CorotContinuumKinematics` must now pass `X_ref`.
Find where the element calls `self._kinematics.initialize(...)` in
`src/oneFEM/model/element/continuum/base.py` and add `X_ref`:

```python
# Before:
self._kinematics.initialize(nGP, nDim, nNodes, dN_dX_list)

# After:
self._kinematics.initialize(nGP, nDim, nNodes, dN_dX_list,
                             X_ref=self._X_ref)   # (nNodes, nDim) reference coords
```

`X_ref` is already stored in the element as `self._X_ref` (confirmed: it is used to
compute the reference Jacobian). The keyword argument is ignored by `**kwargs` in all
other kinematics classes — backward compatible.

---

## Phase 3: Tests and Benchmarks

### Step 3.1 — Pure Rotation Unit Test (GATE)

**File:** `tests/kinematics/test_corot_continuum.py`

Single Quad4 element, unit square `[0,1]^2`. Apply rigid-body rotation as nodal
displacements (no solver):

```python
theta = np.radians(angle_deg)
R_test = np.array([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta),  np.cos(theta)]])
for node_i, (X, Y) in enumerate(ref_coords):
    u_nodes[node_i] = R_test @ [X, Y] - [X, Y]
```

Run for theta = 10, 30, 90, 180 degrees.

**Pass criteria (all four angles):**
- `getStrain(gp)` returns ε = 0 at all GPs to < 1e-10
- `getResistingForce()` returns zero vector to < 1e-10
- `kinematics._R` equals `R_test` to < 1e-10

**MUST PASS before any benchmark.**

#### Additional required tests (same file)

**Test 2 — `revertToLastCommit`:**

Critical for arc-length and line-search solvers where reverts are frequent.

```python
def test_revert():
    # Build kinematics at zero displacement
    kin = CorotContinuumKinematics()
    kin.initialize(nGP, 2, 4, dN_dX_list, X_ref=ref_coords)
    u_zero = Vector(np.zeros(8))

    # Step 1: apply zero displacement, commit
    for gp in range(nGP): kin.update(gp, u_zero)
    kin.applyCorotFrame(u_zero)
    kin.commitState()
    R_committed = kin._R_committed.copy()   # should be I

    # Step 2: apply a 30-degree rotation (trial, not committed)
    u_rot = make_rigid_rotation_disp(ref_coords, 30.0)
    for gp in range(nGP): kin.update(gp, u_rot)
    kin.applyCorotFrame(u_rot)
    assert not np.allclose(kin._R, R_committed)   # R should have changed

    # Step 3: revert
    kin.revertToLastCommit()

    # Pass criteria:
    assert np.allclose(kin._R, R_committed, atol=1e-14)       # R restored
    assert kin._R is not kin._R_committed                      # not aliased
    for gp in range(nGP):
        eps = kin.getStrain(gp)
        # After revert, strain should reflect the committed (zero) state.
        # Note: getStrain() returns the last computed strain — element will
        # call applyCorotFrame(u_zero) again on next update, which resets it.
        # Test that _R_committed == I is sufficient for the revert contract.
    assert np.allclose(kin._R_committed, np.eye(2), atol=1e-14)
```

**Test 3 — K_mat is rotated, K_geo is not:**

This tests the K_mat/K_geo separation directly, without relying on NR convergence.

```python
def test_Kmat_rotated_Kgeo_not():
    # Build kinematics at a known 45-degree rotation
    kin = CorotContinuumKinematics()
    kin.initialize(nGP, 2, 4, dN_dX_list, X_ref=ref_coords)
    u_45 = make_rigid_rotation_disp(ref_coords, 45.0)
    for gp in range(nGP): kin.update(gp, u_45)
    kin.applyCorotFrame(u_45)

    # Synthesize a known K_mat_local (identity for simplicity)
    K_local = Matrix(init=np.eye(8))
    f_local = Vector(np.ones(8))

    # Apply transformToGlobal
    K_global, f_global = kin.transformToGlobal(K_local, f_local)

    # K_global should satisfy: K_global[I,J] = R @ I_{2x2} @ R.T = I_{2x2}
    # (identity is invariant under rotation — use a non-trivial K_local in practice)
    R = kin._R
    for I in range(4):
        for J in range(4):
            si = slice(2*I, 2*I+2); sj = slice(2*J, 2*J+2)
            K_expected = R @ K_local.data[si, sj] @ R.T
            assert np.allclose(K_global.data[si, sj], K_expected, atol=1e-14)

    # K_geo should NOT be passed through transformToGlobal.
    # Build a synthetic K_geo_raw and verify the element adds it unmodified.
    # (This is an integration test — verify in the element's _buildStiffnessAndForce
    # that K_geo is accumulated separately and not passed to transformToGlobal.)
    # Verify: after _buildStiffnessAndForce, self._k = K_mat_global + K_geo_raw
    # where K_geo_raw comes straight from getGeometricStiffness() * dV.
```

### Step 3.2 — Quad4 Snap-Through Arch (B5)

**File:** `src/examples/corot_continuum_benchmarks.py`

Shallow arch, Quad4 + Corot kinematics, DisplacementControl + Newton.
Reference: Crisfield P_cr = 0.5878. Pass: within 5%.
Quadratic NR convergence is a formal pass criterion.

### Step 3.3 — Hex8 Benchmarks (B8, B9)

Add to same benchmark file:
- **B8 — Column Buckling 3D:** Max reaction within 5% of reference
- **B9 — Snap-Through Arch 3D:** P_cr within 5% of 0.5878

### Step 3.4 — Rollup Cross-Check

Run Quad4 cantilever rollup (B4) with Corot kinematics. Corot is valid for small strains
only — agreement with TL/UL expected at small loads (theta < pi/4). Document theta at
which Corot diverges from TL/UL by > 5%.

### Step 3.5 — Full Regression

Run all existing benchmark suites. Zero regressions required:

```bash
cd src
python truss.py
python dynamic_truss.py
python eigen_truss.py
python epp_truss.py
python examples/beam.py
python examples/column_buckling.py
python examples/corot_benchmarks.py
python examples/quad4_patch_test.py
python examples/quad4_cooks_membrane.py
python examples/quad4_simple_shear.py
python examples/quad4_cantilever_rollup.py
python examples/hex8_benchmarks.py
```

---

## Key Design Decisions

### 1. 2D frame extraction: Petracca atan2 formula, not SVD

The original plan used `numpy.linalg.svd` for both 2D and 3D. This is wrong for 2D.
Petracca's `createLocalCoordinateSystem()` computes the polar decomp rotation angle
analytically from the bilinear isoparametric mapping:

```
alpha = atan2(f21 - f12, f11 + f22)
```

where f11..f22 are the 2D deformation gradient at the element centroid. This formula
is exact for Quad4, cheaper than SVD, and has no branch-cut or reflection issues.
SVD is only used for 3D (Hex8) where no closed-form equivalent exists.

### 2. Deformational displacements require centring (not just rotation)

The original plan computed `u_local[I] = R.T @ u_global[I]`. This is wrong — it only
removes rotation but not rigid-body translation. Petracca's `calculateLocalDisplacements()`
centres both reference and current positions at their respective centroids before rotating:

```
u_local[I] = R.T @ (x_cur[I] - C_cur) - (X_ref[I] - C0)
```

Without centring, the pure rotation unit test fails for any element not centred at the
origin. `X_ref` and `C0` must be stored in `initialize()`.

### 3. G = 0: no P/S/H/G spin machinery needed

The Petracca source confirms that polar-decomp frame alignment makes G exactly zero.
For continuum elements (no rotational DOFs), H = I and P = I as well. The EICR tangent
collapses to `T'·K_mat·T + K_sigma`. No spin projector algebra is needed.

### 4. Inherit from `_NonlinearContinuumBase`

Reuses `_computeH()`, `_computeF()`, `_buildKgeo()` — identical for Corot, TL, UL.
Also provides per-GP `_F` and `_strain` array allocation. The cost is inheriting unused
`_B_NL` arrays (trivial memory). Alternative (inheriting from `ContinuumKinematics`
directly) would duplicate H and F computation.

### 5. Two-pass element loop

EICR requires all node positions (and for Hex8, all F[gp]) before polar decomposition.
Alternative (triggering on last GP) is fragile. The two-pass approach is clean,
explicit, and backward-compatible via the `applyCorotFrame()` no-op on the base class.

### 6. `transformToGlobal()` on kinematics, not element

Encapsulates rotation logic in the kinematics object. The element calls
`transformToGlobal(K_mat, f)` polymorphically — no `if formulation == 'corot'` branches.

### 7. Store R as numpy array internally

R is 2×2 or 3×3, used in many small matrix multiplications per Newton iteration.
Raw numpy avoids Matrix wrapper overhead for these inner-loop operations.

### 8. K_mat/K_geo separation

K_mat is computed in the corotated frame and must be rotated to global via `T'·K·T`.
K_geo is computed from global stress and reference dN_dX — it is already in the global
frame and must NOT be rotated. Separating them in the element prevents incorrect
rotation of K_geo.

---

## Common Bugs and Diagnostics

| Symptom | Likely Cause | Diagnostic |
|---------|--------------|------------|
| ε ≠ 0 for pure rotation (any θ) | Missing centring — `R.T @ u_global` instead of Petracca formula | Verify `_get_u_local` uses `R.T @ (x_cur[I]-C_cur) - (X_ref[I]-C0)` |
| ε ≠ 0 for pure rotation (element not at origin) | `X_ref` or `C0` not stored in `initialize()` | Add `X_ref=` kwarg and store both in `initialize()` |
| ε ≠ 0 for pure rotation (2D, systematic) | Node ordering mismatch — C1..C9 assume CCW 1-2-3-4 | Check Quad4 node order vs Petracca's assumption; remap indices if different |
| ε ≠ 0 for pure rotation (2D) | SVD used instead of atan2 formula | Replace with `_extract_R_2d()` using C1..C9 from Petracca |
| ε ≠ 0 for pure rotation (sign) | Polar decomp reflection, det(R) = −1 (3D) | Check `np.linalg.det(R) > 0` after SVD |
| ε ≠ 0, atan2 gives wrong angle | C1..C9 index or sign error | Compare character-for-character with ASD shell source |
| `AttributeError: 'ndarray' has no .data` | `_F[gp]` is raw numpy but code uses `.data` | Confirm `_computeF` return type in V4; use `hasattr(f_gp, 'data')` guard |
| Nonzero stress, correct strain | Material receives ε_global not ε_local | Confirm material called with `_strain[gp]` after `applyCorotFrame()` |
| Linear NR convergence | K_σ missing | Run with/without K_σ, compare NR convergence rate |
| Linear NR convergence | K_mat not rotated | Compare K against finite-difference tangent at θ=30° |
| K_geo incorrectly rotated | `transformToGlobal()` applied to K_geo | Only K_mat enters `transformToGlobal()`; K_geo assembled separately |
| Revert inconsistency (arc-length/line-search) | `copy()` called mid-solution drops rotation state | Confirm `copy()` copies `_R` and `_R_committed` after `initialize()` |
| Correct at small loads, wrong at large | `_R_committed` not updated at commit | Confirm `commitState()` calls `self._R_committed = self._R.copy()` |
| Divergence after revert | `_R` aliased not copied | Confirm `revertToLastCommit()` calls `.copy()` |
| `initialize()` raises ValueError | `X_ref` not passed from element call-site | Add `X_ref=self._X_ref` at element's `initialize()` call |
| Unexpected behavior after commit | `super().commitState()` not called | Check parent manages any shared state and call super first |
| `getDetJ()` returns wrong value silently | Caller expecting float gets `NotImplementedError` | Callers must read detJ from `gp_data`, not from kinematics |

---

## Benchmark Status Table

Update in-place as each step passes.

| # | Step | Status | Notes |
|---|------|--------|-------|
| V1 | Stub raises NotImplementedError | Pending | Before touching code |
| V2 | `initialize()` receives `X_ref` | Pending | Confirm call-site |
| V3 | ASD reference files present | Pending | 4 files |
| V4 | `_NonlinearContinuumBase` API + `_F[gp]` return type | Pending | Determine `.data` vs raw numpy |
| V5 | `_buildBMatrix` is static/class method | Pending | |
| V6 | Quad4 node ordering is CCW 1-2-3-4 | Pending | **Medium risk — verify before Phase 2** |
| 0 | Fix `_buildKgeo` 3D Voigt bug | Pending | 3-line fix |
| 1.1 | Base no-ops | Pending | `applyCorotFrame` + `transformToGlobal` |
| 1.2 | Split element loop | Pending | Two-pass + `applyCorotFrame` |
| 1.3 | Separate K_mat/K_geo | Pending | `transformToGlobal` call |
| 1.4 | Regression gate | Pending | All existing benchmarks |
| 2 | Implement CorotContinuumKinematics | Pending | Replace corot.py |
| 3.1a | Pure rotation test (ε=0) | Pending | θ=10°,30°,90°,180° |
| 3.1b | `revertToLastCommit` test | Pending | R restored, not aliased |
| 3.1c | K_mat rotated / K_geo not test | Pending | Direct block check |
| 3.2 | Quad4 snap-through (B5) | Pending | P_cr within 5%, quadratic NR |
| 3.3 | Hex8 buckling + snap-through (B8, B9) | Pending | Within 5%, quadratic NR |
| 3.4 | Rollup cross-check | Pending | Document validity regime |
| 3.5 | Full regression | Pending | Zero regressions |

---

## References

- Petracca, M. (ASDEA). `ASDShellQ4CorotationalTransformation.h` (OpenSees) — **primary
  reference implementation.** The `_extract_R_2d()` atan2 formula is transcribed directly
  from `createLocalCoordinateSystem()`. The `_get_u_local()` centring formula follows
  `calculateLocalDisplacements()`. The G=0 result is documented in `RotationGradient()`.
- Petracca, M. (ASDEA). `ASDShellQ4LocalCoordinateSystem.h` — shows how alpha is used
  to rotate the local X axis about the normal via quaternion, confirming the 2D rotation
  matrix interpretation.
- Felippa, C.A. & Haugen, B. (2005). "A unified formulation of small-strain corotational
  finite elements: I. Theory." *CMAME* **194**(21–24), 2285–2335.
- Nour-Omid, B. & Rankin, C.C. (1991). "Finite rotation analysis and consistent
  linearization using projectors." *CMAME* **93**(3), 353–384.
- Crisfield, M.A. (1991). *Non-Linear Finite Element Analysis of Solids and Structures*,
  Vol. 1, Ch. 7.
