# oneFEM v2 Architecture

This is the governing design document for oneFEM. It captures the full
architectural philosophy. Future contributors should be able to read this and
understand every design decision and why it was made.

---

## 1. Philosophy

oneFEM is structured around four independent, explicitly chosen components:

```
node          -> pure geometric/DOF vessel
element       -> geometry, interpolation, physics family declaration
kinematics    -> formulation choice within a physics family
material      -> constitutive response within a physics family
```

The user makes four explicit choices at model-building time. The framework
validates compatibility at construction time. Nothing is inferred, nothing is
magic, nothing fails silently inside a Newton loop.

---

## 2. Directory Structure

```
src/oneFEM/model/
  +-- node/
  |   +-- node_2_2.py
  |   +-- node_2_3.py
  |   +-- node_3_3.py
  |   +-- node_3_6.py
  |   +-- node_3_7.py
  |   +-- ...
  |
  +-- element/
  |   +-- continuum/
  |   |   +-- cauchy/          <- each element is ONE self-contained file
  |   |   |   +-- hex8.py      <- shape funcs, dN/dxi, gauss pts, B, API -- all here
  |   |   |   +-- tet4.py
  |   |   |   +-- quad4.py
  |   |   |   +-- tri3.py
  |   |   |   +-- ...
  |   |   +-- cosserat/
  |   |   +-- micromorphic/
  |   |   +-- biot_cauchy/
  |   |   +-- biot_cosserat/
  |   |   +-- biot_micromorphic/
  |   +-- shell/
  |   +-- beam/
  |   +-- zl/
  |   +-- contact/
  |
  +-- kinematics/              <- SIBLING of element, not child
  |   +-- continuum/
  |   |   +-- cauchy/          <- linear, total_lagrangian, updated_lagrangian, corot
  |   |   +-- cosserat/
  |   |   +-- micromorphic/
  |   |   +-- biot_cauchy/
  |   |   +-- biot_cosserat/
  |   |   +-- biot_micromorphic/
  |   +-- shell/
  |   +-- beam/
  |   +-- zl/
  |   +-- contact/
  |
  +-- material/
      +-- continuum/
      |   +-- cauchy/
      |   +-- cosserat/
      |   +-- ...
      +-- shell/
      +-- ...
```

**Critical:** `kinematics/` is a top-level sibling of `element/` and
`material/` under `model/`. It is not a subdirectory of element. This
physically enforces that kinematics does not own element internals.

There is no `isoparametric.py` shared utility file. See Section 5.1.

---

## 3. PhysicsFamily Enum

The physics family is a type-safe enum, not a raw string. Raw strings fail
silently on typos. Enum attributes raise `AttributeError` at import time.

```python
# src/oneFEM/model/physics_family.py

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

All compatibility checks use the enum:

```python
assert kinematics.physics_family == PhysicsFamily.CONTINUUM_CAUCHY
```

A typo like `PhysicsFamily.CONTINUUM_CAUCY` raises `AttributeError` immediately
at import time, not a silent wrong-answer bug deep in the Newton loop.

Elements, kinematics, and materials declare their family as a class attribute:

```python
class Hex8(ContinuumCauchyElement):
    physics_family = PhysicsFamily.CONTINUUM_CAUCHY

class TotalLagrangian(CauchyContinuumKinematics):
    physics_family = PhysicsFamily.CONTINUUM_CAUCHY

class LinearElastic(CauchyContinuumMaterial):
    physics_family = PhysicsFamily.CONTINUUM_CAUCHY
```

---

## 4. Node

The node is a pure vessel. It carries:
- Coordinates
- DOF slots (by count only -- no physical meaning assigned)
- Fixities
- Masses/loads

The node has no concept of physics family, continuum theory, or field names.
It does not know whether its 6 DOFs are (u, phi) for Cosserat or (u, theta)
for a shell. That is the element's concern.

**Naming convention:** `node_{nDim}_{nDOF}.py`
- `node_3_6` is used by both `CosseratHex8` and shell elements -- correctly
- The DOF count is sufficient for node-element compatibility checking
- One node file serves every physics family with matching nDim and nDOF

---

## 5. Element

The element declares its physics family and owns all geometry and
interpolation. It exposes a well-defined API that kinematics uses exclusively.

### 5.1 One File Per Element -- No Shared isoparametric.py

**Each element file is self-contained.** Shape functions, natural derivatives,
Gauss quadrature points, Jacobian computation, B-matrix construction -- all
live in the element's own file.

There is no shared `isoparametric.py` utility file. The old `isoparametric.py`
contained element-specific functions (`quad4_shape_functions()`,
`quad4_shape_derivatives()`, `quad4_gauss_points()`, `hex8_shape_functions()`,
`hex8_shape_derivatives()`, `hex8_gauss_points()`) mixed with generic utilities
(`compute_jacobian()`, `compute_physical_derivatives()`). This is wrong on
both counts:

- Element-specific functions belong in their own element file. Adding a new
  element must not require editing any shared file.
- Generic utilities (`J = dN_dxi.T @ X`, `dN_dX = dN_dxi @ inv(J)`) are
  trivial two-liners that each element inlines or inherits from a thin
  `ContinuumElementBase`.

**Rule:** `hex8_shape_functions()` lives in `hex8.py`. `quad4_gauss_points()`
lives in `quad4.py`. No exceptions. Adding a new element type means creating
one new file with zero changes to any existing file.

### 5.2 Physics Family Declaration

```python
class Hex8(ContinuumCauchyElement):
    physics_family = PhysicsFamily.CONTINUUM_CAUCHY
    nNodes = 8
    nDim   = 3
    nDOF   = 3
```

The implementor writes one file. To add Hex8, write
`element/continuum/cauchy/hex8.py`. All existing kinematics in
`kinematics/continuum/cauchy/` work immediately. Zero changes elsewhere.

### 5.3 Cauchy Continuum Element API

All kinematics call only these methods. Kinematics never access element
private attributes directly.

```
get_N(xi)              -> shape functions at natural coords             (nNodes,)
get_dN_dxi(xi)         -> shape function derivatives                    (nNodes, nDim)
get_jacobian(xi)       -> Jacobian matrix J = dX/dxi                   (nDim, nDim)
get_dN_dX(xi)          -> dN/dX = J^-1 dN/dxi  (reference config)     (nNodes, nDim)
get_dN_dx(xi)          -> dN/dx (current config, for UL)               (nNodes, nDim)
get_B(xi)              -> strain-displacement matrix                    (nVoigt, nDOF*nNodes)
get_B_NL(xi)           -> nonlinear B for geometric stiffness          (nDim*nDim, nDOF*nNodes)
get_F(xi)              -> deformation gradient F = I + H               (nDim, nDim)
get_H(xi)              -> displacement gradient H = u^T dN/dX          (nDim, nDim)
get_gauss_points()     -> [(xi, weight), ...]  quadrature rule
get_coords_ref()       -> reference nodal coordinates                  (nNodes, nDim)
get_coords()           -> current nodal coordinates                    (nNodes, nDim)
get_disp()             -> current nodal displacements                  (nNodes*nDOF,)
get_disp_committed()   -> committed nodal displacements                (nNodes*nDOF,)
get_stress(gp)         -> stress vector at Gauss point (from material) (nVoigt,)
get_tangent(gp)        -> constitutive tangent at Gauss point          (nVoigt, nVoigt)
get_material(gp)       -> material object at Gauss point
get_dof_indices()      -> global DOF indices for assembly              (nNodes*nDOF,)
get_nDim()             -> spatial dimension                            int
get_nNodes()           -> number of nodes                              int
get_nDOF_total()       -> total element DOFs = nNodes * nDOF           int
get_nVoigt()           -> Voigt stress/strain dimension                int
get_nGP()              -> number of Gauss points                       int
```

### 5.4 Optional Enrichment API

Only for elements with `incompatible=True` or `bbar=True`. Default
implementations in the base class return `None` and `0` respectively.

```
get_G(xi)              -> incompatible mode B-matrix                   (nVoigt, nAlpha)
                          returns None if incompatible=False
get_nAlpha()           -> number of incompatible modes                 int
                          returns 0 if incompatible=False
get_B_bar(xi)          -> B-bar for volumetric locking                 (nVoigt, nDOF*nNodes)
                          returns None if bbar=False
```

### 5.5 get_H(xi) as Enrichment Injection Point

When `incompatible=True`, the element includes the alpha contribution in H:

```
H_standard  = u^T @ dN_dX
H_enriched  = u^T @ dN_dX + alpha^T @ dM_dX
```

Kinematics calls `element.get_H(xi)` and never knows whether enrichment is
active. This makes incompatible modes formulation-agnostic -- Linear, TL, UL,
and Corot all work automatically without modification.

### 5.6 Element Flags

```python
Hex8(nodes, kinematics, material, incompatible=False, bbar=False)
```

Both flags are element-level concerns. The condensation of internal alpha DOFs
happens inside the element's `getStiffness()` and `getForce()`. The
condensed stiffness is:

```
K* = K_uu - K_ua @ inv(K_aa) @ K_au
f* = f_u  - K_ua @ inv(K_aa) @ f_a
```

Kinematics, Analysis, Numberer, and System see nothing different -- just a
standard 24-DOF element returning a 24x24 stiffness matrix.

Internal alpha state is stored on the element:
- `self._alpha`           trial values (9,)
- `self._alpha_commit`    committed values (9,)
- Updated inside `_update()` per Newton iteration
- Committed and reverted in `commit()` / `revert()` alongside all other state

### 5.7 Compatibility Checks and Warnings at Construction

Every element constructor performs hard checks and physics warnings. Hard
checks raise `AssertionError` immediately. Warnings use `warnings.warn()`
and do not stop execution.

```python
def __init__(self, nodes, kinematics, material, incompatible=False, bbar=False):

    # --- HARD CHECKS ---
    assert len(nodes) == 8, \
        "Hex8 requires exactly 8 nodes"
    assert all(n.nDim == 3 and n.nDOF == 3 for n in nodes), \
        "Hex8 requires node_3_3 nodes (nDim=3, nDOF=3)"
    assert kinematics.physics_family == PhysicsFamily.CONTINUUM_CAUCHY, \
        f"Hex8 requires CONTINUUM_CAUCHY kinematics, got {kinematics.physics_family}"
    assert material.physics_family == PhysicsFamily.CONTINUUM_CAUCHY, \
        f"Hex8 requires CONTINUUM_CAUCHY material, got {material.physics_family}"

    # --- PHYSICS WARNINGS ---
    if hasattr(material, 'nu') and material.nu > 0.45 and not bbar:
        warnings.warn(
            f"Hex8: nu={material.nu:.3f} is nearly incompressible but bbar=False. "
            f"Volumetric locking likely. Consider Hex8(..., bbar=True).",
            UserWarning, stacklevel=2
        )
    if not incompatible:
        warnings.warn(
            "Hex8: incompatible=False. Shear locking will occur in "
            "bending-dominated problems. Consider Hex8(..., incompatible=True).",
            UserWarning, stacklevel=2
        )
    if kinematics.__class__.__name__ == 'Linear':
        warnings.warn(
            "Hex8: Linear kinematics. Suitable for small deformation only. "
            "For large deformation use TotalLagrangian or UpdatedLagrangian.",
            UserWarning, stacklevel=2
        )
    if incompatible and bbar:
        warnings.warn(
            "Hex8: Both incompatible=True and bbar=True active. Valid for "
            "nearly incompressible bending problems but not fully validated "
            "in oneFEM. Verify against reference solutions.",
            UserWarning, stacklevel=2
        )
```

Warnings are suppressed in benchmark scripts via `warnings.filterwarnings`.
They are active by default in user model scripts.

### 5.8 Different Families Have Different APIs

Shell, beam, ZL, and contact elements expose different APIs appropriate to
their physics. The API in Section 5.3 applies only to continuum Cauchy
elements. There is no universal element base API spanning all families -- that
would be a false abstraction. Each family defines its own base class with its
own API contract.

---

## 6. Kinematics

Kinematics is a first-class library concept, equal peer to element and
material. It owns the formulation -- which strain measure, which stress
measure, how K_mat and K_geo are assembled. It uses the element API
exclusively and never stores element geometry data.

```python
class TotalLagrangian(CauchyContinuumKinematics):
    physics_family = PhysicsFamily.CONTINUUM_CAUCHY

    def getK(self, element):
        nDOF = element.get_nDOF_total()
        K = np.zeros((nDOF, nDOF))
        for gp, (xi, w) in enumerate(element.get_gauss_points()):
            B    = element.get_B(xi)
            B_NL = element.get_B_NL(xi)
            J    = np.linalg.det(element.get_jacobian(xi))
            C    = element.get_tangent(gp)
            S    = element.get_stress(gp)
            S_mat = self._voigt_to_matrix(S)
            K   += w * J * (B.T @ C @ B + B_NL.T @ S_mat @ B_NL)
        return K

    def get_f_int(self, element):
        nDOF = element.get_nDOF_total()
        f = np.zeros(nDOF)
        for gp, (xi, w) in enumerate(element.get_gauss_points()):
            B = element.get_B(xi)
            J = np.linalg.det(element.get_jacobian(xi))
            S = element.get_stress(gp)
            f += w * J * B.T @ S
        return f
```

Kinematics never stores: dN_dX, dN_dxi, nNodes, nDim, nGP, X_ref, or any
geometry from the element. If it needs geometry, it calls the element API.

---

## 7. Material

Material is a pure constitutive responder. It declares its physics family and
implements stress update and tangent computation. It has no knowledge of
element geometry or kinematic formulation.

```python
class LinearElastic(CauchyContinuumMaterial):
    physics_family = PhysicsFamily.CONTINUUM_CAUCHY

    def setStrain(self, eps): ...
    def getStress(self): ...
    def getTangent(self): ...
    def commit(self): ...
    def revert(self): ...
```

---

## 8. Branching Rules

These rules are absolute. Violations are architecture bugs, not style issues.

| What you branch on | Where | Allowed |
|--------------------|-------|---------|
| `n.nDim`, `n.nDOF` | element constructor (node check) | YES |
| `nNodes`, `nDim` | inside element (Gauss loop bounds, etc.) | YES |
| field string `"u"`, `"p"` | Biot element API only | YES |
| `PhysicsFamily` enum value | construction-time compatibility check | YES |
| `element.get_nAlpha() > 0` | kinematics (enrichment check) | YES |
| `isinstance(kin, TotalLagrangian)` | anywhere | **NEVER** |
| `isinstance(el, Hex8)` | kinematics internals | **NEVER** |
| `isinstance(mat, J2Plasticity)` | element or kinematics | **NEVER** |
| kinematics class name as string | anywhere except init warnings | **NEVER** |

---

## 9. Physics Families

| PhysicsFamily enum | Element examples | Kinematics examples |
|--------------------|------------------|---------------------|
| CONTINUUM_CAUCHY | Hex8, Tet4, Quad4, Tri3 | Linear, TL, UL, Corot |
| CONTINUUM_COSSERAT | CosseratHex8 | CosseratLinear, CosseratTL |
| CONTINUUM_MICROMORPHIC | MicromorphicHex8 | MicromorphicTL |
| CONTINUUM_BIOT_CAUCHY | Hex8UP | BiotTL |
| CONTINUUM_BIOT_COSSERAT | CosseratHex8UP | CosseratBiotTL |
| SHELL | ShellQ4, ShellT3 | MindlinReissner, KirchhoffLove |
| BEAM | EulerBernoulliBeam | ... |
| ZL | ZeroLength | ... |
| CONTACT | ContactElement | ... |

**Biot elements** are separate element classes -- not a flag on Cauchy elements.
They carry two interpolation fields to satisfy LBB inf-sup stability (e.g.
Taylor-Hood: quadratic u, linear p). The element API extends with a `field`
parameter for these elements only:

```python
element.get_B(xi, field="u")    # displacement interpolation
element.get_B(xi, field="p")    # pressure interpolation (lower order)
```

---

## 10. Coexistence of Multiple Continuum Theories

Different element types coexist freely in the same domain:

```python
model.add(Hex8(regionA_nodes,         kinematics=TotalLagrangian(), material=mat1))
model.add(CosseratHex8(regionB_nodes, kinematics=CosseratTL(),     material=mat2))
model.add(Hex8UP(regionC_nodes,       kinematics=BiotTL(),         material=mat3))
```

Each element assembles only into its own DOF indices via
`element.get_dof_indices()`. The assembler never assumes uniform DOF layout.

**Interface nodes between regions:** a node shared between Cauchy (nDOF=3) and
Cosserat (nDOF=6) regions must be `node_3_6`. The Cauchy element drives only
the first 3 DOFs. The phi DOFs are driven solely by the Cosserat element.
Interface compatibility is the model builder's responsibility.

---

## 11. What Each Layer Owns -- Summary

```
Node:
  coordinates, DOF count, fixities, masses
  NO: physics meaning, field names, continuum theory

Element (one file per element type):
  shape functions, natural derivatives, Gauss points -- ALL self-contained
  Jacobian, B-matrix, F, H computation
  physics_family declaration (PhysicsFamily enum)
  node/kinematics/material compatibility checks at construction
  physics warnings at construction (locking, large deformation)
  enrichment flags (incompatible, bbar) -- condensation handled here
  internal alpha state (incompatible modes) -- commit/revert here
  NO: strain measures, stress measures, constitutive tangent
  NO: functions or data shared with other element types

Kinematics (one file per formulation):
  K_mat assembly, K_geo assembly, f_int assembly
  strain measure choice (Green-Lagrange, Almansi, linear)
  stress measure choice (PK2, Cauchy)
  physics_family declaration (PhysicsFamily enum)
  NO: shape functions, Jacobian, dN_dX, element geometry of any kind
  NO: stored geometry state from the element

Material (one file per material):
  stress update, tangent computation
  physics_family declaration (PhysicsFamily enum)
  NO: element geometry, kinematic formulation, DOF layout
```

---

## 12. The Newton Loop Invariant

The Analysis/Algorithm/Integrator layer sees only:
- `element.getStiffness()` -> condensed stiffness (nDOF*nNodes x nDOF*nNodes)
- `element.getForce()`     -> condensed internal force (nDOF*nNodes,)
- `element.get_dof_indices()` -> global assembly indices

It never knows whether enrichment is active, which kinematics is attached, or
which material is used. The Newton loop does not change when new element types,
kinematics, or materials are added.

---

## 13. Zero-Change Guarantee

| Task | Files touched |
|------|---------------|
| New Cauchy element (e.g. Tet10) | `element/continuum/cauchy/tet10.py` only |
| New kinematics (e.g. Corot UL hybrid) | `kinematics/continuum/cauchy/corot_ul.py` only |
| New material (e.g. NeoHookean) | `material/continuum/cauchy/neo_hookean.py` only |
| New physics family (e.g. Cosserat) | `element/continuum/cosserat/`, `kinematics/continuum/cosserat/`, `material/continuum/cosserat/` only |
| Incompatible modes on existing Hex8 | `element/continuum/cauchy/hex8.py` only |
| B-bar on existing Hex8 | `element/continuum/cauchy/hex8.py` only |

No changes to Analysis, Algorithm, Integrator, Numberer, System, or Assembler.
No changes to any sibling element, kinematics, or material file.

---

## 14. Audit of Current Codebase (2026-03-07)

### 14.1 Structural Violations

**Kinematics lives under element.** Currently at
`model/element/kinematics/continuum/`. Must move to `model/kinematics/continuum/cauchy/`.

**No physics_family declarations.** Elements, kinematics, and materials have
no `physics_family` attribute. No compatibility checking at construction time.

**No element API.** The element has no `get_B()`, `get_H()`, `get_F()`,
`get_dN_dX()`, etc. Instead, the element dumps geometry data into kinematics
via `initialize(nGP, nDim, nNodes, dN_dX_list, **kwargs)`.

**Shared `isoparametric.py` holds element-specific code.** Currently at
`element/continuum/isoparametric.py`. Contains `quad4_shape_functions()`,
`quad4_shape_derivatives()`, `quad4_gauss_points()`, `hex8_shape_functions()`,
`hex8_shape_derivatives()`, `hex8_gauss_points()` -- all element-specific code
that must live in each element's own file. Also contains generic
`compute_jacobian()` and `compute_physical_derivatives()` which are trivial
two-liners. The UL kinematics imports from this file directly (line 113),
creating a cross-layer dependency. This file must be eliminated entirely.

### 14.2 Coupling Violations

| File | Line(s) | Violation | Severity |
|------|---------|-----------|----------|
| `element/.../isoparametric.py` | 22-125 | Element-specific shape functions (quad4_*, hex8_*) in shared file -- adding new element requires editing shared file | Blocks new element |
| `element/.../base.py` | 55-56 | `_domain()` imports from `isoparametric.py` instead of calling element's own methods | Workaround |
| `kinematics/.../base.py` | 46-51 | `initialize()` stores nGP, nDim, nNodes, nVoigt, nDOF, dN_dX -- element geometry duplicated in kinematics | Blocks new element |
| `kinematics/.../nonlinear_base.py` | 45-51 | `_computeH()` builds displacement gradient from stored `self._dN_dX[gp]`, `self._nNodes`, `self._nDim` | Blocks new element |
| `kinematics/.../nonlinear_base.py` | 73-108 | `_buildBNL()` builds nonlinear B matrix from stored geometry | Blocks new element |
| `kinematics/.../nonlinear_base.py` | 110-145 | `_buildKgeo()` builds geometric stiffness from stored geometry | Blocks new element |
| `kinematics/.../linear.py` | 55-95 | `initialize()` pre-builds B and B-bar from dN_dX | Blocks new element |
| `kinematics/.../linear.py` | 128-185 | `_buildBMatrix()` static method builds linear B inside kinematics | Workaround |
| `kinematics/.../corot.py` | 55-76 | `initialize()` stores X_ref, C0, builds _B_local | Blocks new element |
| `kinematics/.../corot.py` | 86-128 | `_extract_R_2d()` hardcodes 4 nodes -- Quad4 only | Blocks new element |
| `kinematics/.../corot.py` | 265-287 | `transformToGlobal()` iterates stored `self._nNodes` | Workaround |
| `kinematics/.../updated_lagrangian.py` | 51-68 | `initialize()` stores dN_dxi_list, X_ref, duplicates dN_dX | Blocks new element |
| `kinematics/.../updated_lagrangian.py` | 107-136 | `commitState()` reimports `compute_physical_derivatives` from `isoparametric.py` -- cross-layer dependency | Blocks new element |
| `element/.../base.py` | 135 | `if self._kinematics.needs_incremental_u:` -- formulation-aware branch in element | Workaround |
| `element/.../base.py` | 148 | Calls `self._kinematics._setMaterialStrain()` -- protected method | Workaround |
| `element/.../base.py` | 93-96 | Passes formulation-specific kwargs -- element must know what each kinematics needs | Blocks new element |

**Total: 16 violations. 10 "blocks new element", 6 "workaround possible".**

---

## 15. Sequenced Refactor Plan

Each phase ends with a mandatory benchmark gate. No phase begins until the
previous gate is green. If any benchmark regresses, stop immediately.

**Benchmark gate (must pass after every phase and sub-phase):**
- `hex8_benchmarks.py`         -- 16/16
- `hex8_diagnostic_checks.py`  -- 24/24
- `quad4_benchmarks.py`        -- 8/8
- `corot_benchmarks.py`        -- 5/5
- `dynamic_truss.py`           -- PASS
- `eigen_truss.py`             -- PASS

---

### Phase 0: Eliminate isoparametric.py + add element API

**Step 0a -- Move element-specific functions into element files:**
- `quad4_shape_functions`, `quad4_shape_derivatives`, `quad4_gauss_points`
  -> `element/continuum/cauchy/quad4.py`
- `hex8_shape_functions`, `hex8_shape_derivatives`, `hex8_gauss_points`
  -> `element/continuum/cauchy/hex8.py`
- `compute_jacobian`, `compute_physical_derivatives` -> 2-line protected
  methods on `ContinuumElementBase`, or inlined in each element's `_domain()`
- Delete `isoparametric.py`
- Fix UL kinematics import (line 113) with a temporary local copy until
  Phase 3c removes the dependency entirely

**Step 0b -- Add element API to `ContinuumElement` base class:**
```
get_N(xi), get_dN_dxi(xi), get_jacobian(xi)
get_dN_dX(xi), get_dN_dx(xi)
get_B(xi), get_B_NL(xi), get_F(xi), get_H(xi)
get_gauss_points()
get_coords_ref(), get_coords(), get_disp(), get_disp_committed()
get_stress(gp), get_tangent(gp), get_material(gp)
get_nDim(), get_nNodes(), get_nDOF_total(), get_nVoigt(), get_nGP()
get_dof_indices()
get_G(xi) -> None, get_nAlpha() -> 0, get_B_bar(xi) -> None
```

Implement by delegating to precomputed data already stored on the element.
This phase is purely additive -- no existing logic changes.

**Files changed:** `quad4.py`, `hex8.py`, `element/continuum/base.py`.
`isoparametric.py` deleted.
**Kinematics files:** zero logic changes (UL import patched only).
**Benchmark gate:** all pass.

---

### Phase 1: PhysicsFamily enum + move kinematics directory

**Step 1a -- Create `src/oneFEM/model/physics_family.py`** with enum from
Section 3. No other files changed. Benchmark gate: all pass.

**Step 1b -- Move kinematics directory.**
Move `model/element/kinematics/` to `model/kinematics/continuum/cauchy/`.
Update all import paths:

    grep -r "kinematics" src/ --include="*.py" -l

Logic changes: none. Pure reorganization.
Benchmark gate: all pass.

---

### Phase 2: physics_family declarations and construction checks

Add `physics_family = PhysicsFamily.X` class attribute to all elements,
kinematics, and materials. Add construction-time assertions and warnings to
each element constructor per Section 5.7.

Add `warnings.filterwarnings("ignore")` to all benchmark scripts.

**Files changed:** all element, kinematics, and material files (class
attribute + constructor assertions only).
**Benchmark gate:** all pass.

---

### Phase 3: Refactor kinematics to use element API

One kinematics class at a time. Full benchmark gate after each sub-phase.

**Invariant:** after refactoring, each kinematics class stores ZERO geometry.
No `self._dN_dX`, no `self._nNodes`, no `self._X_ref`.

New `initialize()` signature for all kinematics:
```python
def initialize(self, element):
    self._element = element   # store reference only, not geometry
```

**Phase 3a -- Linear kinematics**
- `initialize(element)` replaces old signature
- Remove all stored geometry
- `_buildBMatrix()` moves to element as `get_B(xi)` (done in Phase 0)
- Kinematics calls `element.get_B(gp)`, `element.get_tangent(gp)` directly
- Benchmark gate: all pass.

**Phase 3b -- TotalLagrangian kinematics**
- `initialize(element)` replaces old signature, remove all stored geometry
- `_computeH(gp)` -> `element.get_H(gp)`
- `_buildBNL(gp)` -> `element.get_B_NL(gp)`
- `_buildKgeo(gp)` -> `element.get_dN_dX(gp)` + `element.get_stress(gp)`
- Remove `needs_incremental_u`: use `element.get_disp() - element.get_disp_committed()`
- Remove `_setMaterialStrain`: call `element.get_material(gp).setStrain()` directly
- Benchmark gate: all pass.

**Phase 3c -- UpdatedLagrangian kinematics**
- Same pattern as TL
- `commitState()` calls `element.update_reference()` -- removes the
  `isoparametric.py` cross-layer import entirely (temporary patch from Phase 0
  is now deleted)
- Uses `element.get_dN_dx(gp)` for current-config derivatives
- Benchmark gate: all pass.

**Phase 3d -- CorotContinuum kinematics**
- `initialize(element)` replaces old signature, remove stored `X_ref`, `C0`,
  `_B_local`
- `_extract_R()`: replace hardcoded 4-node unpacking with a loop over
  `element.get_nNodes()` -- makes Corot work for any element type
- `transformToGlobal()` uses `element.get_nNodes()` and `element.get_nDim()`
- Benchmark gate: all pass.

**Phase 3e -- Element base cleanup**
- Remove `initialize()` call and all kwargs passed to kinematics
- Remove `needs_incremental_u` branch
- Remove `_setMaterialStrain` call
- `element._update()` simplifies to:
  ```python
  def _update(self):
      self._kinematics.update(self)
  ```
- Benchmark gate: all pass.

---

### Phase 4: Kinematics owns the K/f integration loop

Move the GP integration loop from `element._buildStiffnessAndForce()` into
each kinematics class. One class at a time, benchmark gate after each.

Each kinematics class gets:
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
```
Benchmark gate: all pass.

**Phase 4b -- TotalLagrangian** (K_mat + K_geo, see Section 6).
Benchmark gate: all pass.

**Phase 4c -- UpdatedLagrangian** (current-config quantities).
Benchmark gate: all pass.

**Phase 4d -- CorotContinuum** (rotation via element API, local -> global).
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
No kinematics changes. No math yet.

```python
# Phase 5a -- flags and internal state
def __init__(self, nodes, kinematics, material, incompatible=False, bbar=False):
    super().__init__(nodes, kinematics, material)
    self._incompatible     = incompatible
    self._bbar             = bbar
    self._alpha            = np.zeros(9)
    self._alpha_commit     = np.zeros(9)

# Phase 5b -- commit/revert
def commit(self):
    super().commit()
    self._alpha_commit = self._alpha.copy()

def revert(self):
    super().revert()
    self._alpha = self._alpha_commit.copy()

# Phase 5c -- stub dispatch
def getStiffness(self):
    if self._incompatible: return self._stiffness_incompatible()
    if self._bbar:         return self._stiffness_bbar()
    return super().getStiffness()

def getForce(self):
    if self._incompatible: return self._force_incompatible()
    if self._bbar:         return self._force_bbar()
    return super().getForce()

def _stiffness_incompatible(self): raise NotImplementedError
def _force_incompatible(self):     raise NotImplementedError
def _stiffness_bbar(self):         raise NotImplementedError
def _force_bbar(self):             raise NotImplementedError

# Phase 5d -- enrichment API overrides
def get_G(self, xi):
    if self._incompatible:
        raise NotImplementedError("get_G: incompatible modes not yet implemented")
    return None

def get_nAlpha(self):
    return 9 if self._incompatible else 0
```

Benchmark gate: all pass (stubs not triggered with default flags).

---

### What does NOT change in any phase

- `Analysis`, `Algorithm`, `Integrator` -- completely untouched
- `Numberer`, `System`, `Assembler` -- completely untouched
- `Node` interface -- untouched
- All example scripts -- untouched (same user-facing API throughout)
- All existing benchmarks -- must pass after every single sub-phase
