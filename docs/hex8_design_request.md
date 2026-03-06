# Hex8 B-bar Element — Design Request for Claude Code

## How to Use This Document

**Read this before writing any code or plan.**

This document is a complete, pre-answered design specification for implementing a
trilinear isoparametric Hex8 brick element with B-bar volumetric locking treatment
in the oneFEM package. All architectural decisions are already made. Your job is to:

1. Read this document fully
2. Read `docs/oneFEM_blueprint.md` and `CLAUDE.md`
3. Read the existing `Quad4` and `LinearContinuumKinematics` implementations
4. Write an implementation plan to `docs/hex8_plan.md`
5. Get PI approval on the plan before writing any code
6. Implement step by step, running benchmarks after each phase

**Stop-and-ask rule:** If anything contradicts the blueprint, CLAUDE.md, or the
existing Quad4 implementation, stop and ask before proceeding.

---

## 1. What Exists Already (as of this request)

### Kinematics hierarchy (`model/element/kinematics/`)
- `Kinematics` — base class: `commitState()`, `revertToLastCommit()`, `copy()`
- `ContinuumKinematics(Kinematics)` — intermediate base for continuum elements
  - Interface: `initialize(X_nodes, nDim)`, `update(u_e)`, `getStrain()`,
    `getBMatrix(dN_dX)`, `getGeometricStiffness(dN_dX, stress)`
- `LinearContinuumKinematics(ContinuumKinematics)` — infinitesimal strain, 2D and 3D,
  no geometric stiffness. **Already works for Quad4. Will be extended to Hex8.**
- `TLContinuumKinematics(ContinuumKinematics)` — stub or partial (check current state)
- `ULContinuumKinematics(ContinuumKinematics)` — stub or partial (check current state)
- `CorotContinuumKinematics(ContinuumKinematics)` — stub (created in Phase 4b)

### Continuum elements (`model/element/continuum/`)
- `Quad4` — fully implemented and benchmarked (patch test, Cook's membrane)
  Uses `LinearContinuumKinematics` via the standard kinematics strategy pattern.
  **This is the primary reference implementation for Hex8.**

### Node types
- `Node33` — 3D node with 3 translational DOFs (ux, uy, uz). **May have bugs.**
  Check current state before proceeding; fix if needed.
- `Node36` — 3D node with 6 DOFs (used by beam and shell, not needed here)

### nD Material
- `ElasticIsotropic` — supports `type='3D'` which returns a 6×6 tangent in Voigt
  notation (sigma = [s11, s22, s33, s12, s23, s13], standard solid mechanics order).
  **This is the material Hex8 will use.**

### Isoparametric utilities
- There is a utility module for Quad4 shape functions. **Hex8 shape functions go in
  the same location** — add to the existing utility, do not create a new module.

---

## 2. Element Formulation — Complete Specification

### 2.1 Element topology

- 8 nodes, trilinear interpolation
- Node numbering: standard right-hand rule, counter-clockwise on bottom face
  then counter-clockwise on top face:

```
    7---6
   /|  /|
  4---5 |       z
  | 3-|-2       |  y
  |/  |/        | /
  0---1          x
```

Local node coordinates (xi, eta, zeta) in reference cube [-1,1]^3:

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

Shape functions: N_I = (1/8)(1 + xi_I*xi)(1 + eta_I*eta)(1 + zeta_I*zeta)

### 2.2 Integration scheme

- **2×2×2 Gauss quadrature** — 8 integration points
- Gauss points at ±1/√3 ≈ ±0.577350269 in each direction, weight = 1.0 each
- This is standard full integration for trilinear hex. DO NOT use 1-point
  reduced integration (hourglass modes appear without stabilization).

### 2.3 B-bar volumetric locking treatment

This is the critical contribution. Standard full integration of trilinear Hex8
locks in nearly-incompressible problems (nu→0.5) because the volumetric strain
field is over-constrained. B-bar (mean dilatation method) fixes this.

**Reference:** Hughes (1980) "Generalization of selective integration procedures
to anisotropic and nonlinear media", IJNME 15:1413–1418. Also Hughes (2000)
"The Finite Element Method", Section 4.5.

**Algorithm:**

For each Gauss point p, the standard B matrix has 6 rows (voigt notation) and
24 columns (8 nodes × 3 DOFs):

```
B_std[p] = [ dN1/dx    0        0     dN2/dx  ...  dN8/dx    0        0     ]
           [    0    dN1/dy      0        0    ... ]
           [    0        0    dN1/dz      0    ... ]
           [ dN1/dy  dN1/dx      0     ...         ]
           [    0    dN1/dz   dN1/dy  ...          ]
           [ dN1/dz     0    dN1/dx   ...          ]
```

The B-bar modification:

1. Split B at each Gauss point into deviatoric and volumetric parts:
   - B_vol[p] extracts the volume change (rows 1-3, averaged): (1/3) * I_vol * (B_11 + B_22 + B_33)
     where I_vol maps back to all three normal strain rows
   - B_dev[p] = B_std[p] - B_vol[p]

2. Compute the mean volumetric B over the element:
   - B_vol_bar = (1/V_e) * sum_p( B_vol[p] * detJ[p] * w[p] )
   - V_e = sum_p( detJ[p] * w[p] )  (element volume)

3. Assemble modified B at each Gauss point:
   - B_bar[p] = B_dev[p] + B_vol_bar

4. Use B_bar[p] everywhere in place of B_std[p]:
   - Strain: eps[p] = B_bar[p] @ u_e
   - Stiffness: K += B_bar[p]^T @ C @ B_bar[p] * detJ[p] * w[p]
   - Internal force: f_int += B_bar[p]^T @ sigma[p] * detJ[p] * w[p]

**Important:** B_vol_bar is computed ONCE per call to `update()`, before the
main Gauss point loop. It requires a preliminary pass over all 8 Gauss points
to compute the element volume and the mean volumetric B.

### 2.4 Voigt notation convention

oneFEM uses the following 6-component Voigt ordering (same as ElasticIsotropic 3D):
- [eps_11, eps_22, eps_33, eps_12, eps_23, eps_13]
- [sig_11, sig_22, sig_33, sig_12, sig_23, sig_13]

Engineering shear strains (gamma = 2*eps_ij) are used in the B matrix rows 4-6,
consistent with the material tangent convention in ElasticIsotropic. Confirm
by checking the 3D case of ElasticIsotropic and the existing Quad4 B matrix.

---

## 3. Architecture Decisions — All Pre-Answered

### 3.1 Where does B-bar live?

**In `LinearContinuumKinematics`**, as a conditional path activated by a flag.

The kinematics object is initialized with `bbar=False` by default. When `bbar=True`,
`getBMatrix()` returns `B_bar` instead of `B_std`. The element code does not change
— it just calls `getBMatrix()` the same way it does for Quad4.

Reasoning: B-bar is a kinematics-level modification (how strain is computed from
displacements), not a material or element-level concern. It fits naturally in the
existing `ContinuumKinematics` interface without breaking the Quad4 path.

Constructor signature:
```python
class LinearContinuumKinematics(ContinuumKinematics):
    def __init__(self, bbar=False):
        ...
```

The B-bar computation requires knowing all Gauss points simultaneously (to compute
the mean), so `initialize()` stores the Gauss point layout and `update(u_e)` does
the two-pass computation when `bbar=True`.

### 3.2 Where does the Hex8 element live?

In `model/element/continuum/`, same directory as Quad4. File: `hex8.py`.
Class name: `Hex8`.

### 3.3 Constructor signature

```python
Hex8(elem_id, nodes, material, kinematics=None, body_force=None)
```

- `nodes`: list of 8 Node33 instances, ordered per Section 2.1
- `material`: an `nDMaterial` instance with `type='3D'` (one copy will be made
  per Gauss point via `material.getCopy()`)
- `kinematics`: a `ContinuumKinematics` instance. Defaults to
  `LinearContinuumKinematics(bbar=True)` — B-bar ON by default for Hex8
  because without it the element is useless for any realistic problem.
- `body_force`: optional numpy array [bx, by, bz], default None

### 3.4 Node33 status

Before implementing Hex8, verify Node33 works correctly:
- Check for double-underscore mangling bugs
- Verify `_update()` signature matches other node types
- Verify `_commitState()` exists and works
- Write a single trivial test (create a Node33, set trial disp, commit, verify)
- Fix any bugs found. This is a prerequisite, not optional.

### 3.5 Shape function utilities

Add Hex8 shape functions to the existing isoparametric utility module (wherever
Quad4 shape functions currently live). Do not create a new file.

Functions needed:
- `hex8_shape_functions(xi, eta, zeta)` — returns N array (8,)
- `hex8_shape_function_derivatives(xi, eta, zeta)` — returns dN_dxi array (8, 3)
- `hex8_gauss_points()` — returns list of (xi, eta, zeta, weight) tuples, 8 points

### 3.6 Relation to TL/UL kinematics

TLContinuumKinematics and ULContinuumKinematics are OUT OF SCOPE for this work
package. Hex8 with `LinearContinuumKinematics(bbar=True)` is the only target.
Do not stub, refactor, or touch TL/UL during this work.

### 3.7 Element DOF layout

8 nodes × 3 DOFs = 24 DOFs per element.
Local DOF ordering: [u1x, u1y, u1z, u2x, u2y, u2z, ..., u8x, u8y, u8z]
(node-major ordering, same pattern as Quad4's [u1x, u1y, u2x, u2y, ...])

### 3.8 Mass matrix

Consistent mass matrix using the same 2×2×2 Gauss integration.
`getMass()` returns the 24×24 consistent mass matrix.
Density `rho` is a constructor argument, default 0.0.

---

## 4. Validation Benchmarks — Required in Order

All benchmarks go in `examples/hex8_benchmarks.py`. Run them after each phase.

### Benchmark 1: 3D Patch Test (linear displacement field)

**Purpose:** Verify shape functions, Jacobian, and B matrix are correct for
arbitrary (non-rectangular) hex element geometry.

**Setup:** Single Hex8 element with distorted (non-rectangular) geometry.
Apply linear displacement field u = [a1*x + a2*y + a3*z, ...] as boundary
conditions at all 8 nodes. Check that the strain field is exactly constant
and equal to the analytical strain, to machine precision (< 1e-10).

Run for all three normal strains and all three shear strains independently.

**Pass criterion:** All 6 strain components exact to < 1e-12 relative error.

### Benchmark 2: Thick-walled cylinder under internal pressure (volumetric locking)

**Purpose:** This is the definitive test for B-bar. A standard Hex8 without
B-bar gives grossly wrong results for nu=0.499 (nearly incompressible).
B-bar should recover the exact Lamé solution.

**Setup:**
- Quarter-cylinder model (symmetry BCs), inner radius r_i = 1.0, outer r_o = 3.0
- E = 1000.0, nu = 0.499 (nearly incompressible)
- Internal pressure p_i = 1.0, external traction = 0
- Mesh: 4×4×1 Hex8 elements in the radial×circumferential×axial directions
  (1 element thick in axial direction with symmetry BCs top and bottom)

**Exact solution (Lamé):**
- u_r(r) = [p_i * r_i^2 / (E*(r_o^2 - r_i^2))] * [(1-2nu)*r + (1+nu)*r_o^2/r]
- At inner radius: u_r(r_i) = analytical value
- At outer radius: u_r(r_o) = analytical value

**Pass criterion:** Radial displacement at inner and outer radius within 2%
of Lamé solution. Without B-bar the error should be > 50% — confirm both
behaviors (with and without bbar) to demonstrate the fix.

### Benchmark 3: 3D Cantilever under tip load

**Purpose:** Verify bending response and element assembly for a multi-element model.

**Setup:**
- Cantilever: L=10, b=1, h=1, mesh: 10×1×1 elements (length × width × height)
- E=1000, nu=0.3
- Base fully fixed (x=0 face)
- Tip load P=1 applied at x=L face, distributed as nodal forces
- Reference: Euler-Bernoulli tip deflection delta = P*L^3 / (3*E*I)
  where I = b*h^3/12 = 1/12

**Pass criterion:** Tip deflection within 5% of Euler-Bernoulli reference.
(Hex8 is a solid element — shear locking is present but mild for 10-element mesh
with aspect ratio 10:1:1. B-bar does not fix shear locking, only volumetric.)

### Benchmark 4: Cook's membrane in 3D (single layer)

**Purpose:** Cross-validate with the existing Quad4 result using a 3D model.

**Setup:**
- Same Cook's membrane geometry as the Quad4 benchmark, but extruded 1 unit in z
- 1 layer of Hex8 elements, plane strain conditions (uz=0 on both z-faces)
- Compare apex y-displacement to the Quad4 PlaneStrain result

**Pass criterion:** Apex displacement within 1% of Quad4 PlaneStrain result
on the same in-plane mesh density.

---

## 5. Implementation Order

Claude Code must implement in this exact order:

**Phase 1: Prerequisites**
1. Verify and fix Node33
2. Add Hex8 shape functions to the isoparametric utility module
3. Unit test: shape functions sum to 1, partition of unity, correct corner values

**Phase 2: B-bar extension to LinearContinuumKinematics**
1. Add `bbar` flag to constructor
2. Implement two-pass B-bar computation in `update(u_e)`
3. Unit test: for a rectangular element, B_vol_bar == average of B_vol at all GPs
4. Run Quad4 benchmarks to confirm zero regression (`examples/quad4_patch_test.py`,
   `examples/quad4_cooks_membrane.py`)

**Phase 3: Hex8 element**
1. Implement `Hex8` in `model/element/continuum/hex8.py`
2. Add export to `model/element/continuum/__init__.py`
3. Run Benchmark 1 (patch test) before proceeding

**Phase 4: Benchmarks**
1. Benchmark 2: thick-walled cylinder — confirm B-bar fix and B_std failure
2. Benchmark 3: 3D cantilever
3. Benchmark 4: Cook's membrane 3D

**Phase 5: Full regression**
Run all existing benchmarks in order:
- `examples/truss.py`
- `examples/beam.py`
- `examples/column_buckling.py`
- `examples/corot_benchmarks.py`
- `examples/quad4_patch_test.py`
- `examples/quad4_cooks_membrane.py`
- `examples/hex8_benchmarks.py`

Zero regressions required before declaring complete.

---

## 6. Coding Conventions (from CLAUDE.md)

- Single underscore `_attr` for all instance attributes in non-Domain classes
- Trial/committed two-state pattern: every stateful quantity has `_trial` and
  `_committed` versions; `commitState()` deep-copies trial→committed;
  `revertToLastCommit()` deep-copies committed→trial
- `getCopy()` must return a fully independent copy (no shared references)
- `super().__init__()` required at the top of every `__init__`
- Element subclasses must NOT call `super()._commit()` or `super()._update()`
- No new top-level directories under `src/oneFEM/` without PI approval
- Material interface is sacred — do not modify `nDMaterial` or `ElasticIsotropic`

---

## 7. What Success Looks Like

At the end of this work package:

1. `Node33` works correctly and has a trivial test
2. `LinearContinuumKinematics` supports `bbar=True/False` with zero Quad4 regression
3. `Hex8` exists with correct 24-DOF assembly
4. All 4 hex8 benchmarks pass
5. All existing benchmarks still pass
6. The thick-walled cylinder benchmark explicitly demonstrates the locking failure
   of standard Hex8 (bbar=False) AND the fix with B-bar (bbar=True)

Item 6 is non-negotiable. It is the scientific justification for B-bar.

---

## 8. References

- Hughes (1980), "Generalization of selective integration procedures", IJNME 15:1413
- Hughes (2000), "The Finite Element Method", Sections 4.4–4.5 (B-bar derivation)
- Belytschko, Liu, Moran (2000), "Nonlinear Finite Elements for Continua and
  Structures", Chapter 8 (Hex8 and locking)
- Existing oneFEM: `model/element/continuum/quad4.py` — primary reference implementation
