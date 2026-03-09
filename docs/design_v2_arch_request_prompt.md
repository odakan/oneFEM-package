# oneFEM v2 Architecture Design Request

## First: Delete the old architecture document

Delete `docs/architecture.md` entirely. It was written with an incomplete understanding of the
architecture. Do not reference it. Start fresh.

---

## Task: Write `docs/oneFEM_v2_architecture.md`

This is the governing design document for oneFEM. It must capture the full architectural
philosophy before any refactor begins. Write it as a reference document — not a task list, not a
refactor plan. Future contributors should be able to read this and understand every design decision
and why it was made.

---

## The Full Architecture

### Philosophy

oneFEM is structured around three equal, independent pillars:

```
node          → pure geometric/DOF vessel
element       → geometry, interpolation, physics family declaration
kinematics    → formulation choice within a physics family
material      → constitutive response within a physics family
```

The user makes four explicit choices. The framework validates compatibility at construction time.
Nothing is inferred, nothing is magic, nothing fails silently inside a Newton loop.

---

### Directory Structure

```
src/oneFEM/model/
  ├── node/
  │   ├── node_2_2.py
  │   ├── node_2_3.py
  │   ├── node_2_4.py
  │   ├── node_3_3.py
  │   ├── node_3_4.py
  │   ├── node_3_6.py
  │   ├── node_3_7.py
  │   └── ...
  │
  ├── element/
  │   ├── continuum/
  │   │   ├── cauchy/          ← Hex8, Tet4, Quad4, Tri3, ...
  │   │   ├── cosserat/
  │   │   ├── micromorphic/
  │   │   ├── biot_cauchy/
  │   │   ├── biot_cosserat/
  │   │   └── biot_micromorphic/
  │   ├── shell/
  │   ├── beam/
  │   ├── zl/
  │   └── contact/
  │
  ├── kinematics/              ← SIBLING of element, not child
  │   ├── continuum/
  │   │   ├── cauchy/          ← linear, total_lagrangian, updated_lagrangian, corot
  │   │   ├── cosserat/
  │   │   ├── micromorphic/
  │   │   ├── biot_cauchy/
  │   │   ├── biot_cosserat/
  │   │   └── biot_micromorphic/
  │   ├── shell/
  │   ├── beam/
  │   ├── zl/
  │   └── contact/
  │
  └── material/
      ├── continuum/
      │   ├── cauchy/
      │   ├── cosserat/
      │   └── ...
      ├── shell/
      └── ...
```

**Critical:** `kinematics/` is a top-level sibling of `element/` and `material/`. It is not a
subdirectory of element. This physically enforces that kinematics does not own element internals.

---

### Node

The node is a pure vessel. It carries:
- Coordinates
- DOF slots (by count only — no physical meaning assigned)
- Fixities
- Masses/loads

The node has no concept of physics family, continuum theory, or field names. It does not know
whether its 6 DOFs are (u, φ) for Cosserat or (u, θ) for a shell. That is the element's concern.

**Naming convention:** `node_{nDim}_{nDOF}.py`
- `node_3_6` is used by both `CosseratHex8` and shell elements — correctly
- The DOF count is sufficient for node-element compatibility checking
- One node file serves every physics family with matching nDim and nDOF

**Compatibility check at node construction:**
```python
Node_3_3(x, y, z):
    assert len(coords) == 3
```

---

### Element

The element declares its physics family and owns all geometry and interpolation. It exposes a
well-defined API that all kinematics use exclusively.

**Physics family** is declared as a class attribute:
```python
class Hex8(ContinuumCauchyElement):
    physics_family = "continuum/cauchy"
    nNodes = 8
    nDim   = 3
    nDOF   = 3
```

**The implementor writes one file.** To add Hex8, write `element/continuum/cauchy/hex8.py`.
All existing kinematics in `kinematics/continuum/cauchy/` work immediately. Zero changes elsewhere.

**The Cauchy continuum element API** (all kinematics call only these methods):
```python
get_N(xi)              → shape functions at natural coords             (nNodes,)
get_dN_dxi(xi)         → shape function derivatives                    (nNodes, nDim)
get_jacobian(xi)       → Jacobian matrix J = dX/dξ                    (nDim, nDim)
get_dN_dX(xi)          → dN/dX = J⁻¹ dN/dξ  (reference config)       (nNodes, nDim)
get_dN_dx(xi)          → dN/dx (current config, for UL)               (nNodes, nDim)
get_B(xi)              → strain-displacement matrix                    (nVoigt, nDOF*nNodes)
get_B_NL(xi)           → nonlinear B for geometric stiffness          (nDim*nDim, nDOF*nNodes)
get_F(xi)              → deformation gradient F = I + H               (nDim, nDim)
get_H(xi)              → displacement gradient H = u^T dN/dX          (nDim, nDim)
get_gauss_points()     → [(xi, weight), ...]  quadrature rule
get_coords_ref()       → reference nodal coordinates                  (nNodes, nDim)
get_coords()           → current nodal coordinates                    (nNodes, nDim)
get_disp()             → current nodal displacements                  (nNodes*nDOF,)
get_stress(gp)         → stress vector at Gauss point (from material) (nVoigt,)
get_material(gp)       → material object at Gauss point
get_dof_indices()      → global DOF indices for assembly              (nNodes*nDOF,)
```

**Optional enrichment API** (only for elements with incompatible=True or bbar=True):
```python
get_G(xi)              → incompatible mode B-matrix                   (nVoigt, nAlpha)
                         returns None if incompatible=False
get_B_bar(xi)          → B-bar for volumetric locking
                         returns None if bbar=False
```

**Note on get_H(xi):** This is the enrichment injection point. When `incompatible=True`, the
element includes the alpha contribution in H:
```
H = u^T @ dN_dX + alpha^T @ dM_dX
```
Kinematics calls `element.get_H(xi)` and never knows whether enrichment is active. This makes
incompatible modes formulation-agnostic — Linear, TL, UL, and Corot all work automatically.

**Element flags (Hex8 example):**
```python
Hex8(nodes, kinematics, material, incompatible=False, bbar=False)
```
Both flags are element-level concerns. The condensation of internal alpha DOFs happens inside the
element's `getStiffness()` and `getForce()`. Kinematics, Analysis, Numberer, and System see nothing
different — just a 24-DOF element.

**Different element families have different API contracts.** Shell, beam, ZL, and contact elements
expose different APIs appropriate to their physics. The API above applies only to continuum Cauchy
elements. There is no universal element base API spanning all families — that would be a false
abstraction.

**Compatibility checks at element construction:**
```python
Hex8.__init__(nodes, kinematics, material):
    # Node compatibility
    assert all(n.nDim == 3 and n.nDOF == 3 for n in nodes), \
        "Hex8 requires node_3_3 nodes"
    assert len(nodes) == 8, \
        "Hex8 requires exactly 8 nodes"
    # Kinematics compatibility
    assert kinematics.physics_family == "continuum/cauchy", \
        f"Hex8 requires continuum/cauchy kinematics, got {kinematics.physics_family}"
    # Material compatibility
    assert material.physics_family == "continuum/cauchy", \
        f"Hex8 requires continuum/cauchy material, got {material.physics_family}"
    # Warnings
    if material.is_nearly_incompressible and not bbar:
        warnings.warn(
            "Nearly incompressible material without bbar=True — volumetric locking likely. "
            "Consider Hex8(..., bbar=True) or reduce Poisson ratio."
        )
    if isinstance(kinematics, Linear) and self._expects_large_strain():
        warnings.warn(
            "Linear kinematics with potentially large deformation problem."
        )
```

---

### Kinematics

Kinematics is a first-class library concept, equal peer to element and material. It owns the
formulation — which strain measure, which stress measure, how K_mat and K_geo are assembled.

**It uses the element API exclusively.** It never accesses element private attributes.

```python
class TotalLagrangian(CauchyContinuumKinematics):
    physics_family = "continuum/cauchy"

    def getK(self, element):
        K = zeros(nDOF*nNodes, nDOF*nNodes)
        for xi, w in element.get_gauss_points():
            B   = element.get_B(xi)         # element API
            F   = element.get_F(xi)         # element API
            B_NL = element.get_B_NL(xi)     # element API
            J   = det(element.get_jacobian(xi))
            C   = element.get_material(gp).getTangent()
            S   = element.get_stress(gp)
            K  += w * J * (B.T @ C @ B + B_NL.T @ S_mat @ B_NL)
        return K
```

Kinematics never stores geometry. It never stores dN_dX, nNodes, nDim, or any element data. Every
call passes through the element API.

---

### Material

Material is a pure constitutive responder. It declares its physics family and implements stress
update and tangent computation. It has no knowledge of element geometry or kinematic formulation.

```python
class LinearElastic(CauchyContinuumMaterial):
    physics_family = "continuum/cauchy"

    def setStrain(self, eps): ...
    def getStress(self): ...
    def getTangent(self): ...
    def commit(self): ...
    def revert(self): ...
```

---

### Branching Rules

These rules are absolute. Violations are architecture bugs, not style issues.

| What you branch on | Where | Allowed |
|---|---|---|
| `n.nDim`, `n.nDOF` | element constructor (node check) | YES |
| `nNodes`, `nDim` | inside element (e.g. Gauss loop bounds) | YES |
| field string `"u"`, `"p"` | Biot element API only | YES |
| `physics_family` string | construction-time compatibility check | YES |
| `isinstance(kin, TotalLagrangian)` | anywhere | **NEVER** |
| `isinstance(el, Hex8)` | kinematics internals | **NEVER** |
| `isinstance(mat, J2Plasticity)` | element or kinematics | **NEVER** |
| kinematics class name as string | anywhere | **NEVER** |

---

### Physics Families

The physics family string is the compatibility key. Element, kinematics, and material must match.

| Family string | Element examples | Kinematics examples |
|---|---|---|
| `continuum/cauchy` | Hex8, Tet4, Quad4 | Linear, TL, UL, Corot |
| `continuum/cosserat` | CosseratHex8 | CosseratLinear, CosseratTL |
| `continuum/micromorphic` | MicromorphicHex8 | MicromorphicTL |
| `continuum/biot_cauchy` | Hex8UP | BiotTL |
| `continuum/biot_cosserat` | CosseratHex8UP | CosseratBiotTL |
| `shell` | ShellQ4, ShellT3 | MindlinReissner, KirchhoffLove |
| `beam` | EulerBernoulliBeam | ... |
| `zl` | ZeroLength | ... |
| `contact` | ContactElement | ... |

**Biot elements** (`biot_cauchy`, `biot_cosserat`, etc.) are separate element classes — not a flag
on Cauchy elements. They carry two interpolation fields with different quadrature and shape function
orders to satisfy LBB inf-sup stability (e.g. Taylor-Hood: quadratic u, linear p). The element API
is extended with a `field` parameter for these elements only.

---

### Coexistence of Multiple Continuum Theories in One Domain

Different element types coexist freely in the same domain:
```python
model.add(Hex8(regionA_nodes,          kinematics=TotalLagrangian(), material=mat1))
model.add(CosseratHex8(regionB_nodes,  kinematics=CosseratTL(),      material=mat2))
model.add(Hex8UP(regionC_nodes,        kinematics=BiotTL(),          material=mat3))
```

Each element assembles only into its own DOF indices via `element.get_dof_indices()`. The assembler
never assumes uniform DOF layout — it scatters using the indices the element reports.

Interface nodes between regions: a node shared between Cauchy (nDOF=3) and Cosserat (nDOF=6)
regions must be `node_3_6`. The Cauchy element only drives the first 3 DOFs. The φ DOFs are driven
solely by the Cosserat element. Interface compatibility is the model builder's responsibility. The
framework validates nDim and nDOF at element construction — it does not enforce inter-region
physical compatibility.

---

### What Each Layer Owns — Summary

```
Node:
  coordinates, DOF count, fixities, masses
  NO: physics meaning, field names, continuum theory

Element (one file per element type):
  shape functions, Jacobian, B-matrix, F, H, gauss points
  physics_family declaration
  node/kinematics/material compatibility checks and warnings
  enrichment flags (incompatible, bbar) — condensation handled here
  internal alpha state (incompatible modes) — commit/revert here
  NO: strain measures, stress measures, constitutive tangent

Kinematics (one file per formulation):
  K_mat assembly, K_geo assembly
  strain measure choice (Green-Lagrange, Almansi, linear)
  stress measure choice (PK2, Cauchy)
  physics_family declaration
  NO: shape functions, Jacobian, dN_dX, element geometry of any kind

Material (one file per material):
  stress update, tangent computation
  physics_family declaration
  NO: element geometry, kinematic formulation, DOF layout
```

---

### The Newton Loop Invariant

The Analysis/Algorithm/Integrator layer sees only:
- `element.getStiffness()` → (nDOF*nNodes × nDOF*nNodes) condensed stiffness
- `element.getForce()` → (nDOF*nNodes,) condensed internal force
- `element.get_dof_indices()` → global assembly indices

It never knows whether enrichment is active, which kinematics is attached, or which material is
used. The Newton loop does not change when new element types, kinematics, or materials are added.

---

### Adding New Capabilities — Zero-Change Guarantee

| Task | Files touched |
|---|---|
| New Cauchy element (e.g. Tet10) | `element/continuum/cauchy/tet10.py` only |
| New kinematics (e.g. Corot UL hybrid) | `kinematics/continuum/cauchy/corot_ul.py` only |
| New material (e.g. NeoHookean) | `material/continuum/cauchy/neo_hookean.py` only |
| New physics family (e.g. Cosserat) | `element/continuum/cosserat/`, `kinematics/continuum/cosserat/`, `material/continuum/cosserat/` only |
| Incompatible modes on existing Hex8 | `element/continuum/cauchy/hex8.py` only |
| B-bar on existing Hex8 | `element/continuum/cauchy/hex8.py` only |

No changes to Analysis, Algorithm, Integrator, Numberer, System, or Assembler.
No changes to any sibling element, kinematics, or material file.

---

## After writing `docs/oneFEM_v2_architecture.md`

Do not implement anything. Do not refactor anything. Do not touch any source file.

Produce only:
1. `docs/oneFEM_v2_architecture.md` — the document above, written clearly and completely
2. A short audit table: for each violation of this architecture found in the current codebase,
   record: File | Line | Violation | Severity (blocks new element / workaround possible)
3. A sequenced refactor plan: phases, what changes in each phase, what benchmarks must pass
   after each phase before proceeding to the next

We review and approve all three before any code changes begin.
