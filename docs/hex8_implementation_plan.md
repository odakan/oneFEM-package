# Hex8 B-bar Element — Implementation Plan

## Context

The oneFEM package has a working 2D continuum pipeline (Quad4 with Linear/TL/UL kinematics, all benchmarked). The 3D kinematics code exists (3D branches in B-matrix, K_geo, Voigt mapping) but has never been exercised because there is no 3D continuum element. This plan implements a Hex8 brick element with B-bar volumetric locking treatment, validating the full 3D path.

**Prerequisites already satisfied:**
- Node33 was fixed in Phase 1A.2 — no bugs remain (single underscore attrs, correct _update signature, _commitState with deep copy)
- ElasticIsotropic supports `type='3D'` with correct 6×6 tangent, Voigt ordering [xx, yy, zz, xy, yz, xz]
- LinearContinuumKinematics already has 3D B-matrix branch (nDim==3, 6 Voigt rows, same ordering)
- ContinuumElement base (Option B architecture) handles single kinematics + gp-indexed calls

**Design request reference:** `docs/hex8_design_request.md` — all architecture decisions pre-answered by PI.

---

## Critical Issue: ContinuumElement Base is 2D-Only

`base.py` line 77 unpacks GP tuples as `for xi, eta, w in gauss_points:` and line 78 calls `_getShapeDerivatives(xi, eta)`. This hardcodes 2D.

**Fix (2-line change):**
```python
# Before (line 77-78):
for xi, eta, w in gauss_points:
    dN_dxi = self._getShapeDerivatives(xi, eta)

# After:
for gp_tuple in gauss_points:
    *coords, w = gp_tuple
    dN_dxi = self._getShapeDerivatives(*coords)
```

Quad4 continues to work unchanged — its GPs are still 3-tuples `(xi, eta, w)`, its `_getShapeDerivatives(xi, eta)` still takes 2 args via `*coords` unpacking.

---

## Pre-Coding Verifications (do these BEFORE writing any code)

Two questions about `base.py` must be answered by reading the file before implementation begins.
Do not assume — open the file and confirm.

### V1: `_gp_data` existence and population order

`Hex8._buildMass()` uses `self._gp_data[gp_idx]` to retrieve `(detJ, w)` per Gauss point.
Verify that `ContinuumElement._domain()` populates `_gp_data = list(zip(detJ_list, w_list))`
(or equivalent) **before** calling `_buildMass()`. If `_gp_data` does not exist in `base.py`,
change `_buildMass()` to recompute detJ inline from Gauss coordinates and nodal positions.
Do not assume the attribute exists.

### V2: `thickness` parameter in ContinuumElement

Verify whether `ContinuumElement.__init__` stores thickness and whether `_domain()` multiplies
`detJ` by it anywhere in the integration loop. If yes, either:
- Add a guard in the base: `if self._thickness is not None: detJ *= self._thickness`
  (Hex8 passes `thickness=None` to skip multiplication), or
- Override the integration loop in Hex8 entirely rather than passing a dummy value.

Do NOT pass `thickness=1.0` to a 3D element as a silent workaround without confirming it is a
true no-op in the base. If it is a true no-op (multiplying by 1.0 throughout), document this
with a comment in `Hex8.__init__`. If it is not a no-op, apply the fix above.

---

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `model/element/continuum/base.py` | MODIFY | Generalize GP loop for 2D/3D, pass gp_weights kwarg, fix mass init |
| `model/element/continuum/isoparametric.py` | MODIFY | Add hex8 shape functions/derivatives/gauss points |
| `model/element/continuum/hex8.py` | CREATE | Hex8 element (thin subclass of ContinuumElement) |
| `model/element/continuum/__init__.py` | MODIFY | Export Hex8 |
| `model/element/kinematics/continuum/linear.py` | MODIFY | Add bbar flag + B-bar computation |
| `examples/hex8_benchmarks.py` | CREATE | 4 benchmarks |

---

## Phase 1: Prerequisites

### 1.1 Generalize ContinuumElement base for 3D

**File**: `src/oneFEM/model/element/continuum/base.py`

Changes:

1. **GP loop** (line 77): Change `for xi, eta, w` to `for gp_tuple` + `*coords, w = gp_tuple` + `_getShapeDerivatives(*coords)`. This is dimension-agnostic.

2. **Pass gp_weights kwarg** to kinematics initialize (needed for B-bar volume computation):
```python
self._kinematics.initialize(nGP, nDim, nNodes, dN_dX_list,
                            dN_dxi_list=dN_dxi_list,
                            X_ref=Matrix(init=X_nodes),
                            gp_weights=list(zip(detJ_list, w_list)))
```
This is always passed; Linear ignores it unless bbar=True. TL/UL ignore it. Same kwargs pattern already used for `dN_dxi_list` and `X_ref`.

3. **Fix getMass()**: Add `self._m = None` in `__init__`. Update `getMass()`:
```python
def getMass(self):
    if self._m is None:
        return Matrix(shape=[self._nDOF_total, self._nDOF_total])
    return self._m
```

### 1.2 Add Hex8 isoparametric functions

**File**: `src/oneFEM/model/element/continuum/isoparametric.py`

Add to existing module (same file as Quad4 functions — do NOT create a new file):

**`hex8_shape_functions(xi, eta, zeta)` → N array (8,)**

Node ordering (right-hand rule, CCW bottom then CCW top):
```
    7---6
   /|  /|
  4---5 |       z
  | 3-|-2       |  y
  |/  |/        | /
  0---1          x
```

Local node coordinates:
| Node | xi | eta | zeta |
|------|----|-----|------|
| 0    | -1 | -1  | -1   |
| 1    | +1 | -1  | -1   |
| 2    | +1 | +1  | -1   |
| 3    | -1 | +1  | -1   |
| 4    | -1 | -1  | +1   |
| 5    | +1 | -1  | +1   |
| 6    | +1 | +1  | +1   |
| 7    | -1 | +1  | +1   |

```python
N_I = (1/8)(1 + xi_I*xi)(1 + eta_I*eta)(1 + zeta_I*zeta)
```

**`hex8_shape_derivatives(xi, eta, zeta)` → dN_dxi array (8, 3)**

Rows = nodes, cols = [dN/dxi, dN/deta, dN/dzeta]

**`hex8_gauss_points()` → list of 8 tuples (xi, eta, zeta, weight)**

2×2×2 Gauss quadrature at ±1/√3 ≈ ±0.577350269, weight = 1.0 each.

### 1.3 Verify: Quad4 regression

After base.py changes, run:
- `examples/quad4_patch_test.py` — expect 6/6 PASS
- `examples/quad4_cooks_membrane.py` — expect 2/2 PASS

---

## Phase 2: B-bar in LinearContinuumKinematics

**File**: `src/oneFEM/model/element/kinematics/continuum/linear.py`

### 2.1 Add `bbar` flag to constructor

```python
def __init__(self, bbar=False):
    ...existing init...
    self._bbar = bbar
    self._gp_weights = None   # stored at initialize() for use in copy()
    self._B_bar = []          # populated when bbar=True, empty otherwise
```

### 2.2 B-bar computation in `initialize()`

For linear kinematics, B is constant (independent of u_e), so B-bar can be fully precomputed in `initialize()` — no per-iteration cost.

After building standard B matrices (existing code), add:

```python
if self._bbar and nDim == 3:
    gp_weights = kwargs.get('gp_weights', None)
    if gp_weights is None:
        raise ValueError("B-bar requires gp_weights kwarg from ContinuumElement._domain()")

    # Store for copy()
    self._gp_weights = gp_weights

    # Pass 1: compute element volume and mean volumetric B
    V_e = 0.0
    B_vol_bar_data = np.zeros((nVoigt, nDOF))
    for gp in range(nGP):
        B_std = self._B[gp].data
        B_vol_gp = self._extract_B_vol(B_std)
        detJ_gp, w_gp = gp_weights[gp]
        V_e += detJ_gp * w_gp
        B_vol_bar_data += B_vol_gp * detJ_gp * w_gp
    B_vol_bar_data /= V_e

    # Pass 2: assemble B_bar = B_dev + B_vol_bar at each GP
    self._B_bar = []
    for gp in range(nGP):
        B_std = self._B[gp].data
        B_vol_gp = self._extract_B_vol(B_std)
        B_dev_gp = B_std - B_vol_gp
        self._B_bar.append(Matrix(init=B_dev_gp + B_vol_bar_data))
```

### 2.3 B_vol extraction (private static method)

```python
@staticmethod
def _extract_B_vol(B_std):
    """Extract volumetric part of B matrix (3D only).

    The volumetric strain is eps_vol = eps_xx + eps_yy + eps_zz = trace(eps).
    B_vol redistributes the mean dilatation equally to all three normal strain
    rows and zeroes the shear rows:

      B_vol[0] = B_vol[1] = B_vol[2] = (B_std[0] + B_std[1] + B_std[2]) / 3
      B_vol[3] = B_vol[4] = B_vol[5] = 0   (shear rows — unchanged)

    This implements Hughes (1980) mean dilatation split: B = B_dev + B_vol.

    Note on Voigt convention: rows 3-5 of B_std use engineering shear
    (gamma = 2*eps_ij), consistent with LinearContinuumKinematics._buildBMatrix().
    The CTensor covariant/contravariant machinery (Helnwein 2001) handles the
    factor-of-2 at the material boundary. The volumetric split only touches rows
    0-2 (normal strains), so the shear convention never enters this method.
    """
    B_vol = np.zeros_like(B_std)
    trace_row = (B_std[0] + B_std[1] + B_std[2]) / 3.0
    B_vol[0] = trace_row
    B_vol[1] = trace_row
    B_vol[2] = trace_row
    # rows 3-5 remain zero — shear does not contribute to volumetric strain
    return B_vol
```

### 2.4 getBMatrix returns B_bar when active

```python
def getBMatrix(self, gp):
    if self._bbar and self._B_bar:
        return self._B_bar[gp]
    return self._B[gp]
```

### 2.5 Update `update()` for B-bar

The existing `update()` computes `eps_vec = self._B[gp] @ u_e`. Must use the correct B:

```python
def update(self, gp, u_e):
    B = self.getBMatrix(gp)  # returns B_bar when active
    eps_vec = B @ u_e
    self._strain[gp] = CTensor(eps_vec.data.tolist(), self._nVoigt, CTensor.COV)
```

### 2.6 Update `copy()` to preserve bbar flag and gp_weights

```python
def copy(self):
    c = LinearContinuumKinematics(bbar=self._bbar)
    if self._nGP is not None:
        c.initialize(self._nGP, self._nDim, self._nNodes,
                     [m for m in self._dN_dX],
                     gp_weights=self._gp_weights)   # required when bbar=True
    return c
```

`self._gp_weights` is `None` when `bbar=False` (set in `__init__`). The `kwargs.get()`
call in `initialize()` returns `None` in that case, and the `if self._bbar and nDim == 3`
guard prevents reaching the `ValueError`, so the Quad4 path (bbar=False) is unaffected.

### 2.7 Verify: Quad4 regression

Quad4 uses `LinearContinuumKinematics()` (bbar=False default) — zero change in behavior.
Run `quad4_patch_test.py` and `quad4_cooks_membrane.py`.

---

## Phase 3: Hex8 Element

### 3.1 Create Hex8 element

**File**: `src/oneFEM/model/element/continuum/hex8.py`

Thin subclass of ContinuumElement, same pattern as Quad4:

```python
from .base import ContinuumElement
from .isoparametric import hex8_gauss_points, hex8_shape_derivatives, hex8_shape_functions
from ..kinematics.continuum.linear import LinearContinuumKinematics
from ...._systools.backend import np
from ...._systools.data import Vector, Matrix


class Hex8(ContinuumElement):
    """8-node trilinear hexahedral element (3D) with optional B-bar.

    Node ordering (right-hand rule, CCW bottom then CCW top):
        7---6
       /|  /|
      4---5 |       z
      | 3-|-2       |  y
      |/  |/        | /
      0---1          x

    :param tag: Element ID
    :param nodes: list of 8 Node33 instances
    :param material: nDMaterial template (type='3D')
    :param kinematics: ContinuumKinematics (default: LinearContinuumKinematics(bbar=True))
    :param rho: mass density (default: 0.0)
    :param body_force: [bx, by, bz] array (default: None)
    """

    def __init__(self, tag, nodes, material, kinematics=None, rho=0.0, body_force=None):
        if kinematics is None:
            kinematics = LinearContinuumKinematics(bbar=True)
        # thickness is verified to be a true no-op multiplier (= 1.0) in
        # ContinuumElement._domain() for 3D elements — see Pre-Coding Verification V2.
        # If base.py uses thickness as a section-integral factor, this must be
        # replaced with a thickness=None guard in the base instead.
        super().__init__(tag, nodes, material, kinematics, thickness=1.0)
        self._rho = rho
        self._body_force = body_force

    def _getGaussPoints(self):
        return hex8_gauss_points()

    def _getShapeDerivatives(self, xi, eta, zeta):
        return hex8_shape_derivatives(xi, eta, zeta)

    def _domain(self):
        super()._domain()
        if self._rho > 0:
            self._buildMass()

    def _buildMass(self):
        """Consistent mass matrix: M = sum_gp rho * N_mat^T @ N_mat * detJ * w

        N_mat is (3 x 24) interpolation matrix:
        N_mat = [[N1, 0, 0, N2, 0, 0, ..., N8, 0, 0],
                 [0, N1, 0, 0, N2, 0, ..., 0, N8, 0],
                 [0, 0, N1, 0, 0, N2, ..., 0, 0, N8]]

        Requires self._gp_data to be populated by super()._domain() before this
        method is called (verified in Pre-Coding Verification V1).
        """
        nDOF = self._nDOF_total
        M = np.zeros((nDOF, nDOF))
        gauss_pts = self._getGaussPoints()

        for gp_idx, gp_tuple in enumerate(gauss_pts):
            *coords, w = gp_tuple
            N = hex8_shape_functions(*coords)  # (8,)
            detJ, _ = self._gp_data[gp_idx]   # populated by super()._domain()

            # Build N_mat (3 x 24)
            N_mat = np.zeros((3, nDOF))
            for a in range(8):
                N_mat[0, 3*a]     = N[a]
                N_mat[1, 3*a + 1] = N[a]
                N_mat[2, 3*a + 2] = N[a]

            M += self._rho * (N_mat.T @ N_mat) * detJ * w

        self._m = Matrix(init=M)

    def __repr__(self):
        return "Hex8(ID={})".format(self._ID)
```

### 3.2 Update `continuum/__init__.py`

```python
from .hex8 import Hex8
# Add "Hex8" to __all__
```

---

## Phase 4: Benchmarks

**File**: `src/examples/hex8_benchmarks.py`

### Benchmark 1: 3D Patch Test (GATE — must pass before proceeding)

Single distorted Hex8 element. Apply linear displacement field as BCs at all 8 nodes for each of 6 independent strain states:

1. eps_xx = constant (u = a*x, v=0, w=0)
2. eps_yy = constant (u=0, v = a*y, w=0)
3. eps_zz = constant (u=0, v=0, w = a*z)
4. gamma_xy = constant (u = a*y, v = a*x, w=0)
5. gamma_yz = constant (u=0, v = a*z, w = a*y)
6. gamma_xz = constant (u = a*z, v=0, w = a*x)

Check: strain at all 8 GPs matches analytical value to < 1e-12.

**Note on B-bar and patch test:** A linear displacement field produces a constant strain
field, which means B_vol[gp] is identical at every Gauss point. Therefore
`B_vol_bar == B_vol[gp]` for all gp, and `B_bar == B_std` identically. The patch test
passes with both `bbar=True` and `bbar=False` and will give identical results — this is
correct and expected, not a coverage gap. Run the patch test with both flags and assert
that the results are numerically identical (difference < 1e-15). The thick-walled cylinder
(Benchmark 2) is the only benchmark that exercises the B-bar modification.

### Benchmark 2: Thick-Walled Cylinder (B-bar proof — NON-NEGOTIABLE)

**Purpose**: Scientific justification for B-bar.

- Quarter-cylinder model, r_i=1.0, r_o=3.0
- E=1000, nu=0.499 (nearly incompressible)
- Internal pressure p_i=1.0, external free
- 4×4×1 Hex8 mesh (radial × circumferential × axial)
- Symmetry BCs on cut faces and top/bottom

**Exact Lamé solution:**
```
u_r(r) = [p_i * r_i² / (E * (r_o² - r_i²))] * [(1-2ν)*r + (1+ν)*r_o²/r]
```

**Two runs:**
1. `bbar=True` → u_r within 2% of Lamé (**PASS**)
2. `bbar=False` → u_r error > 50% (**expected failure, proves locking**)

### Benchmark 3: 3D Cantilever

- L=10, b=1, h=1, 10×1×1 mesh
- E=1000, nu=0.3
- Fixed at x=0, tip load P=1 distributed on x=L face
- Reference: delta = PL³/(3EI), I = bh³/12 = 1/12
- Pass: within 5% of Euler-Bernoulli

### Benchmark 4: Cook's Membrane 3D

- Same Cook's membrane geometry as Quad4 benchmark
- Extruded 1 unit in z, 1 layer of Hex8
- uz=0 on both z-faces (plane strain condition)
- Compare apex y-displacement to Quad4 PlaneStrain result
- Pass: within 1% on same in-plane mesh density

---

## Phase 5: Full Regression

Run all existing benchmarks — zero regressions:
1. `src/truss.py`
2. `src/examples/beam.py`
3. `src/examples/column_buckling.py`
4. `src/examples/corot_benchmarks.py`
5. `src/examples/quad4_patch_test.py`
6. `src/examples/quad4_cooks_membrane.py`
7. `src/examples/hex8_benchmarks.py`

---

## Implementation Order (strict sequence)

| Step | Phase | Description | Gate |
|------|-------|-------------|------|
| 0 | Pre | Resolve V1 (_gp_data) and V2 (thickness) by reading base.py | Required before any code |
| 1 | 1.1 | Generalize base.py GP loop + gp_weights kwarg + mass init | — |
| 2 | 1.2 | Add hex8 shape functions to isoparametric.py | — |
| 3 | 1.3 | Quad4 regression (patch 6/6, Cook's 2/2) | PASS required |
| 4 | 2.1-2.6 | B-bar in LinearContinuumKinematics | — |
| 5 | 2.7 | Quad4 regression again | PASS required |
| 6 | 3.1-3.2 | Hex8 element + __init__.py | — |
| 7 | 4.1 | Patch test (6 strain states, bbar=True and bbar=False) | **GATE** — PASS before proceeding |
| 8 | 4.2 | Thick-walled cylinder (bbar=True + bbar=False) | PASS required |
| 9 | 4.3 | 3D Cantilever | PASS required |
| 10 | 4.4 | Cook's membrane 3D | PASS required |
| 11 | 5 | Full regression (all 7 benchmark suites) | Zero regressions |

---

## Key Technical Details

### Voigt ordering (confirmed consistent)
Both `ElasticIsotropic(type='3D')` and `LinearContinuumKinematics._buildBMatrix(nDim=3)` use:
- [eps_xx, eps_yy, eps_zz, gamma_xy, gamma_yz, gamma_xz]
- Engineering shear (gamma = 2*eps_ij) in B-matrix rows 3-5
- Same ordering in material tangent C (6×6)

The CTensor covariant/contravariant convention (Helnwein 2001) handles the Voigt factor-of-2
correctly at the material boundary. The `_extract_B_vol` method operates only on rows 0-2
(normal strains), leaving shear rows 3-5 zero by construction, so the engineering-vs-tensor
shear distinction never enters the volumetric split computation.

### B-bar scope
- Only activates when `bbar=True` AND `nDim == 3`
- Silently skips for 2D. Note: plane strain (nDim==2, plane_strain=True) locks just as
  severely as 3D for nu→0.5; B-bar for 2D is simply not implemented here. A future
  B-bar Quad4 would require extending this branch to nDim==2.
- Hex8 defaults to `bbar=True`; Quad4 defaults to `bbar=False`

### What B-bar does (Hughes 1980)
Standard Hex8 over-constrains the volumetric strain field → locks for nu→0.5. B-bar replaces per-GP volumetric B with the element-mean volumetric B, relaxing the constraint. The deviatoric part is unchanged.

```
B_bar[gp] = B_dev[gp] + B_vol_bar
B_dev[gp] = B_std[gp] - B_vol[gp]
B_vol_bar = (1/V_e) * sum_gp(B_vol[gp] * detJ[gp] * w[gp])
V_e = sum_gp(detJ[gp] * w[gp])
```

### Mass matrix
Consistent mass: `M = sum_gp rho * N_mat^T @ N_mat * detJ * w`
where N_mat (3×24) maps nodal DOFs to displacement at GP.
Only built if `rho > 0`. getMass() returns zero Matrix otherwise.

---

## Success Criteria (from design request §7)

1. Node33 works correctly (already done)
2. LinearContinuumKinematics supports bbar=True/False with zero Quad4 regression
3. Hex8 exists with correct 24-DOF assembly
4. All 4 hex8 benchmarks pass
5. All existing benchmarks still pass
6. Thick-walled cylinder benchmark explicitly demonstrates locking failure (bbar=False) AND fix (bbar=True) — NON-NEGOTIABLE
