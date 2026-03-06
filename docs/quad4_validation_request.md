# Quad4 — Full Kinematics Validation Request

## Purpose

This document specifies the complete validation suite for the Quad4 element across all
four kinematics formulations: Linear, Total Lagrangian, Updated Lagrangian, and Corotational.
It is the 2D companion to `docs/hex8_validation_request.md`. Claude Code must read both
before proceeding.

**Scope:** All eight benchmarks. Benchmarks 1–2 cover Linear. Benchmarks 3–5 cover TL
and UL. Benchmarks 6–8 cover Corot (beams only — `CorotContinuumKinematics` on Quad4 is
deferred to the continuum-corot WP). All benchmarks live in `examples/quad4_benchmarks.py`.

---

## Visualization Requirements

Every benchmark produces one or more plots saved to `docs/validation/quad4/`.
Create this directory if it does not exist. All plots use matplotlib. The exact
filename for each plot is specified per benchmark. Do not skip plots.

### General plot style

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection, LineCollection
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
})
SAVE_DIR = 'docs/validation/quad4'
import os; os.makedirs(SAVE_DIR, exist_ok=True)
```

### Mesh plotting utility (implement once, reuse in every benchmark)

```python
def plot_quad4_mesh(ax, nodes_xy, connectivity,
                   deformed_xy=None, field=None, field_label='',
                   alpha=0.6, cmap='viridis', scale=1.0,
                   show_nodes=False, title=''):
```

Behaviour:
1. **Reference wireframe:** Always draw all element edges as thin grey lines
   (linewidth=0.5, color='#999999').
2. **Deformed filled faces:** If `deformed_xy` is provided, draw each element as a
   filled quadrilateral using `matplotlib.patches.Polygon`. Deformed coordinates:
   `pos = nodes_xy + scale * (deformed_xy - nodes_xy)` so scale=1 gives actual
   deformation and scale>1 amplifies for visual clarity.
3. **Field colouring:** If `field` is provided (one scalar per element), colour each
   element face by the field value via `PatchCollection`. Use `cmap` (diverging
   'RdBu_r' for signed fields like stress, sequential 'viridis' for non-negative fields
   like strain magnitude). Add a colorbar with `field_label`.
4. If `show_nodes=True`, draw each node as a small black dot.
5. Equal aspect ratio, axis labels ('X', 'Y'), title.
6. Return the axes object.

---

## What Is Being Validated

| Kinematics | Status After This WP |
|------------|----------------------|
| `LinearContinuumKinematics` | Validated by B1–B2 |
| `TotalLagrangianContinuumKinematics` | Validated by B3–B5, B8 |
| `UpdatedLagrangianContinuumKinematics` | Validated by B3–B5, B8 |
| `CorotContinuumKinematics` on Quad4 | **Deferred** (continuum-corot WP) |
| Beam Corotational (post-refactor) | Validated by B6–B7 |

**Note on B6–B7:** These benchmarks validate the beam/truss corotational coordinate
transformation after the `coordTransformation/` → `kinematics/` refactor. They do not
test `CorotContinuumKinematics`. These are distinct things. Snap-through (B5) uses
TL and UL on Quad4.

---

## Constructor Usage

```python
from oneFEM.model.element.continuum import Quad4
from oneFEM.model.element.kinematics.continuum import (
    LinearContinuumKinematics,
    TotalLagrangianContinuumKinematics,
    UpdatedLagrangianContinuumKinematics,
)

elem_lin = Quad4(tag, nodes, material, kinematics=LinearContinuumKinematics())
elem_tl  = Quad4(tag, nodes, material, kinematics=TotalLagrangianContinuumKinematics())
elem_ul  = Quad4(tag, nodes, material, kinematics=UpdatedLagrangianContinuumKinematics())
```

---

## Benchmark 1 — Patch Test (Linear — GATE)

### Purpose

The patch test is the minimum necessary condition for convergence. It verifies that
Quad4 can represent a state of constant stress exactly on an irregular mesh.
If it fails, nothing else is worth running.

### Model Definition

**Geometry — 5-element irregular patch (plane stress):**
Outer boundary: unit square [0,1]×[0,1]. One deliberately irregular interior node.
5 Quad4 elements sharing the interior node.

| Node | X     | Y     | Role                   |
|------|-------|-------|------------------------|
| 1    | 0.000 | 0.000 | Corner                 |
| 2    | 1.000 | 0.000 | Corner                 |
| 3    | 1.000 | 1.000 | Corner                 |
| 4    | 0.000 | 1.000 | Corner                 |
| 5    | 0.500 | 0.000 | Edge midpoint          |
| 6    | 1.000 | 0.500 | Edge midpoint          |
| 7    | 0.500 | 1.000 | Edge midpoint          |
| 8    | 0.000 | 0.500 | Edge midpoint          |
| 9    | 0.240 | 0.220 | **Interior (irregular)** |

**Material:** Plane stress, E=1.0, ν=0.25, t=1.0. G = E/(2(1+ν)) = 0.4.

**Boundary conditions:** Prescribe nodal displacements at all 8 boundary nodes
(1–8) from the exact analytical field. Leave node 9 free.

**Three load cases — run all three:**

Load case 1 (σ_x=1, σ_y=0, τ_xy=0):
```
u = x,   v = -0.25*y
```

Load case 2 (σ_x=0, σ_y=1, τ_xy=0):
```
u = -0.25*x,   v = y
```

Load case 3 (σ_x=0, σ_y=0, τ_xy=1):
```
u = y*(1+ν)/E = 1.25*y,   v = x*(1+ν)/E = 1.25*x
```

### Pass Criteria

- Stress at ALL Gauss points (5 elements × 4 GP = 20 values) matches prescribed
  stress to < 1e-10 (relative).
- Interior node 9 displacement matches analytical field to < 1e-10.
- All three load cases must pass.

### Plots

**`quad4_b1_patch_test.png`** — 1×3 figure, one subplot per load case:

Each subplot:
- Use `plot_quad4_mesh`. Reference mesh as grey wireframe. Deformed mesh (scale=50)
  as filled elements coloured by the relevant stress component (σ_x, σ_y, or τ_xy).
  Since stress is constant across the patch, all elements should be the same colour —
  any colour variation means the test failed.
- Mark node 9 with a red star. Annotate: "u9={:.4f} (ref={:.4f}), v9={:.4f} (ref={:.4f})"
- Mark all 4 GPs in each element as small grey dots.
- Subplot title: load case name + "Max GP err: {:.2e}"
- show_nodes=True.
Overall title: "B1 — Patch Test | 5-element irregular patch | PASS / FAIL"

---

## Benchmark 2 — Cook's Membrane (Linear)

### Purpose

Cook's tapered panel is the standard shear-locking and distortion test for Quad4.
The reference fine-mesh tip displacement is 23.91 (dimensionless with E=1, t=1).

### Model Definition

**Geometry — tapered panel (plane stress):**

Vertices (counterclockwise):
- Bottom-left: (0, 0)
- Bottom-right: (48, 44)
- Top-right: (48, 60)
- Top-left: (0, 44)

**Material:** E=1.0, ν=1/3, t=1.0, plane stress.

**Boundary conditions:**
- Left edge (X=0): u_x = u_y = 0.
- Right edge (X=48): distributed shear V=1/unit-height.
  Interior nodes: F_y = V/N. Corner nodes: F_y = V/(2N).

**Mesh refinement study:** n=2, 4, 8, 16 (n×n elements).

### Pass Criteria

- 16×16 mesh tip y-displacement within 0.1 of reference 23.91.
- Monotonic convergence from coarse to fine mesh.

**Warning:** Tip displacement significantly above 23.91 on the fine mesh indicates a
factor-of-2 shear error in the B matrix. Cook's membrane is specifically designed to
catch Voigt shear convention bugs.

### Plots

**`quad4_b2_cooks_membrane.png`** — 1×2 figure:

Left (deformed mesh):
- Use `plot_quad4_mesh` for the 16×16 mesh.
- Reference wireframe (grey) and deformed mesh at actual scale, coloured by von Mises
  stress per element (σ_vm = √(σ_x²−σ_xσ_y+σ_y²+3τ_xy²), plane stress). Colormap: hot_r.
- Mark tip corner (48, 60) with a red star. Annotate: "Tip v={:.3f} (ref=23.91)"
- Title: "B2 — Cook's Membrane | 16×16 | von Mises Stress"

Right (mesh convergence):
- X-axis: n (2, 4, 8, 16), log scale. Y-axis: tip y-displacement.
- Solid blue line with circles. Horizontal dashed red line at 23.91.
- Annotate each point.
- Title: "B2 — Convergence | Tip y-Displacement"

---

## Benchmark 3 — Simple Shear Kinematic Unit Test (TL and UL)

### Purpose

Most important kinematic verification test. Verifies F, E (Green-Lagrange), and e
(Almansi) analytically on a single element. No solver. Add to automated test suite
at `tests/kinematics/test_simple_shear.py` and run at every commit.

### Model Definition

Single Quad4 element, unit square [0,1]×[0,1]. Simple shear:
```
u_x = γ·Y,   u_y = 0
```
Apply directly as nodal displacements (no solver). Run for γ = 0.1, 0.5, 1.0, 2.0.

### Analytical Reference (γ = 0.5)

**F (2×2):** `[[1.0, 0.5], [0.0, 1.0]]`

**TL Green-Lagrange, Voigt [E_11, E_22, Γ_12]:**
`[0.000, 0.125, 0.500]`

**UL Almansi, Voigt [e_11, e_22, γ_12]:**
`[0.000, -0.125, 0.500]`

E_22 = +0.125 but e_22 = −0.125 — correct and expected.
det(F) = 1.0 for all γ (isochoric).

**Warning on Voigt ordering:** E_12 (tensor shear) = 0.25. Γ_12 (engineering, Voigt) = 0.50.
If Γ_12 = 0.25, there is a factor-of-2 Voigt shear storage bug.

### Pass Criteria

F, E, e, det(F) at all 4 GPs to < 1e-14 for all γ. Error > 1e-12 is a bug.

### Plots

**`quad4_b3_simple_shear.png`** — 2×2 figure, one subplot per γ (0.1, 0.5, 1.0, 2.0):

Each subplot:
- Reference unit square as grey wireframe. Deformed element as filled blue quad
  (alpha=0.5) at actual scale using `plot_quad4_mesh`.
- At γ=1.0 and γ=2.0 the shear must be visually obvious (skewed parallelogram).
- Deformed GPs as red dots.
- Subplot title: "γ={:.1f} | max F err: {:.1e} | max E err: {:.1e}"
Overall title: "B3 — Simple Shear | TL and UL | Machine Precision"

---

## Benchmark 4 — Large-Deformation Cantilever Rollup (TL and UL)

### Purpose

Primary validation for TL and UL. Exact closed-form solution at every load level
(Reissner 1972). TL and UL must produce identical results — any discrepancy is a bug.

### Model Definition

**Geometry:** L=10.0, H=1.0. Mesh: 40×2 Quad4 elements. Plane stress.

**Material:** E=1.2×10⁶, ν=0.0 (required — any ν≠0 has no clean closed-form correction).

**BCs:** Left end fully fixed. 20 equal load steps.

**Loading:** M_full = EI·2π/L, I = H³/12. Applied as force couple on tip nodes.

### Analytical Reference (Reissner 1972)

For M = EI·θ/L: `x_tip = (L/θ)·sin(θ)`, `y_tip = (L/θ)·(1−cos(θ))`

| Step | θ     | x_tip   | y_tip  |
|------|-------|---------|--------|
| 5    | π/2   | 6.366   | 6.366  |
| 10   | π     | 0.000   | 6.366  |
| 15   | 3π/2  | −6.366  | 6.366  |
| 20   | 2π    | 0.000   | 0.000  |

### Pass Criteria

- Tip within 1% of analytical at all 4 check steps.
- TL and UL agree to 6 significant figures.
- Newton converges within 10 iterations per step.
- **Quadratic Newton-Raphson convergence is a formal pass criterion.** Print residual
  norms at steps 5, 10, 15, 20. Linear convergence = missing K_σ.

### Plots

**`quad4_b4_rollup_deformed.png`** — 1×4 figure, one subplot per check step:
- Use `plot_quad4_mesh`. Reference beam as grey wireframe. Deformed mesh at actual
  scale, coloured by GL strain magnitude ‖E‖ (viridis, shared colorbar).
- Tip marked with a red star. Annotation: "Step {i} | θ={:.2f}π | x={:.3f} | y={:.3f}"
- Equal aspect ratio.
Overall title: "B4 — Cantilever Rollup | TL (solid) = UL (dashed)"

**`quad4_b4_rollup_curve.png`** — 1×2 figure:

Left (tip trajectory):
- TL solid blue, UL dashed orange, analytical black dots. Equal aspect ratio.
- Mark 4 check steps with colored squares.
- Title: "Tip Trajectory | TL vs UL vs Analytical"

Right (Newton convergence):
- Semilogy. One curve per check step (5, 10, 15, 20).
- Reference dashed line for quadratic slope.
- Title: "Newton Residual Norm | Steps 5, 10, 15, 20"

---

## Benchmark 5 — Snap-Through Arch (TL and UL, pre-snap)

### Purpose

Tests TL and UL on a path-following problem. Validates the pre-snap regime and
K_σ quality. Arc-length control is deferred; full post-snap tracing is not in scope.

### Model Definition

**Geometry:** Rise H=0.5, half-span=5.0. R = 25.25. Mesh: 10×2 Quad4 along full arch.
Use exact cylindrical coordinates — snap-through load is sensitive to rise/span ratio.

**Material:** E=10⁴, ν=0.0, plane stress, t=1.0.

**BCs:** Both support nodes pinned (u_x=u_y=0).
Add small downward imperfection 0.001%·H at apex to trigger snap-through.

**Loading:** Displacement control at apex. Steps of δ=0.02 to δ_max=1.0.

### Pass Criteria

- Max apex reaction R_y within 5% of P_cr = 0.5878 (verify vs `corot_benchmarks.py`
  geometry before comparing).
- TL and UL agree on limit load to 4 significant figures.
- Graceful divergence past limit point.
- Quadratic Newton convergence for pre-snap steps.

### Plots

**`quad4_b5_snapthrough.png`** — 1×3 figure:

Left (load-displacement):
- TL solid blue, UL dashed orange. P_cr dashed red line.
- Red star at computed limit load + annotation.
- Pre-snap region shaded light green.
- Title: "B5 — Snap-Through Arch | P_cr={:.4f} (ref=0.5878)"

Centre (arch profile sequence):
- 2D X-Y plot. Arch profiles at 25%, 50%, 75% of P_cr and at limit. Blue→red progression.
- Reference arch as solid grey.
- Title: "Arch Profile During Loading"

Right (Newton iterations per step):
- Bar chart. Green=converged, red=diverged. Vertical dashed line at limit step.
- Title: "Newton Iterations Per Step"

---

## Benchmark 6 — Lee's Frame (Corotational Beams)

### Purpose

Standard large-rotation beam benchmark (Lee 1971, Simo & Vu-Quoc 1986). Rotations
exceed 90°. Validates the Corotational coordinate transformation after the
`coordTransformation/` → `kinematics/` refactor. Reference: Simo & Vu-Quoc (1986,
CMAME 58) Example 7.4 — the corrected parameter set confirmed in Phase 4b.

### Model Definition

**Geometry:** Right-angle frame. Horizontal member (L=10) and vertical member (L=10).
Base of vertical member: fully fixed. Load at free end of horizontal member.

**Material and section:** E=7.2×10⁶, b=h=0.3, A=0.09, I=6.75×10⁻⁴.
10 ElasticBeamColumn elements per member.

**Loading:** Vertical point load P downward at horizontal member free end.
Load control, steps from 0 to P_max.

### Reference Values

Use the tabulated reference values from Simo & Vu-Quoc (1986) Example 7.4, exactly
as used in Phase 4b. Confirm with the Phase 4b passing result — post-refactor must
match pre-refactor to machine precision (zero regression).

### Pass Criteria

- Tip displacement and rotation within 2% of Simo & Vu-Quoc tabulated values.
- **Zero regression vs Phase 4b** — post-refactor result must match to machine precision.

### Plots

**`quad4_b6_lees_frame.png`** — 1×3 figure:

Left (load-displacement curve):
- X-axis: tip horizontal displacement. Y-axis: P.
- Computed curve: solid blue. S&V-Q reference points: black circles.
- Title: "B6 — Lee's Frame | Load vs Tip Displacement"

Centre (deformed shape at P_max):
- 2D X-Y plot. Undeformed frame as grey lines (two straight members, right-angle corner).
- Deformed frame at final load as thick blue curve.
- Mark fixed base, corner, and free tip with distinct symbols (square, circle, star).
- Actual scale — rotations exceeding 90° should be visually dramatic.
- Title: "Deformed Shape at P_max"

Right (Newton iterations):
- Bar chart. Green=converged, red=diverged.
- Title: "Newton Iterations Per Step"

---

## Benchmark 7 — Column Buckling (Corotational Beams)

### Purpose

Zero-regression check after the beam refactor. Uses the existing `column_buckling.py`
model unchanged. Post-refactor result must be numerically identical to pre-refactor.

### Model Definition

Identical to `examples/column_buckling.py`. Do not change any parameters.

### Pass Criteria

- P_cr within the tolerance already established in `column_buckling.py`.
- **Zero regression:** post-refactor P_cr matches pre-refactor to ≥ 8 significant figures.

### Plots

**`quad4_b7_column_buckling.png`** — 1×2 figure:

Left (load-displacement curves):
- Pre-refactor result as dashed black, post-refactor as solid blue.
- Euler P_cr as horizontal dashed red line.
- Curves should be indistinguishable.
- Title: "B7 — Column Buckling | Pre vs Post Refactor"

Right (regression residual):
- X-axis: load step. Y-axis: |post − pre| tip displacement.
- Should be at machine precision throughout.
- Title: "Post-Refactor Regression Residual"

---

## Benchmark 8 — Thick-Walled Cylinder Under Internal Pressure (TL and UL)

### Purpose

Validates TL and UL under volumetric deformation and a non-trivial deformation gradient.
The Lamé solution provides reference radial displacement and hoop stress.
Also cross-validates Linear, TL, and UL: at small load all three agree; at large load
TL/UL diverge from Linear but must agree with each other.

### Model Definition

**Geometry:** Quarter-cylinder, 2D plane strain. a=1.0, b=3.0.
Mesh: 8×4 Quad4 elements (8 radial, 4 circumferential). Cylindrical node coordinates.

**Material:** E=1000, ν=0.3, plane strain.

**BCs:** θ=0 face: u_y=0. θ=π/2 face: u_x=0. Outer surface: free.
Inner surface: radial pressure P_in as nodal forces.

**Two load levels:** P_in=10 (linear regime) and P_in=100 (nonlinear), 10 steps each.

### Analytical Reference (Lamé, plane strain)

```
u_r(r) = P_in * a² / (E*(b²-a²)) * [(1-2ν)*r + (1+ν)*b²/r]
σ_θ(r) = P_in * a² / (b²-a²) * [1 + b²/r²]
```

For a=1, b=3, E=1000, ν=0.3:
```
u_r(1): P_in=10 → 0.015125,    P_in=100 → 0.15125
σ_θ(1): P_in=10 → 12.5,        P_in=100 → 125.0
```

### Pass Criteria

Small load: Linear, TL, UL all within 2% of Lamé. All three agree to 0.1%.
Large load: TL and UL agree to 1%. Linear may differ from TL/UL by up to 15% (expected).
Both TL and UL: quadratic Newton convergence.

### Plots

**`quad4_b8_cylinder_mesh.png`** — 1×3 figure:

Left: Reference mesh (grey wireframe only). Equal aspect ratio.
Title: "B8 — Quarter-Cylinder | 8×4 Quad4 | Plane Strain"

Centre: Deformed mesh at P_in=10 (TL), coloured by u_r (viridis). Scale ×20.
Title: "Deformed | P_in=10 (×20)"

Right: Deformed mesh at P_in=100 (TL), coloured by u_r. Actual scale.
Title: "Deformed | P_in=100 (TL, actual)"

**`quad4_b8_cylinder_profiles.png`** — 1×2 figure:

Left (u_r profile):
- X-axis: r (1 to 3). Y-axis: u_r.
- P_in=10: Lamé (solid black), Linear (dashed blue), TL (dashed green), UL (dashed orange).
- P_in=100: same scheme with circle markers, brighter colors.
- Title: "B8 — Radial Displacement u_r(r)"

Right (σ_θ profile):
- Same legend. Title: "B8 — Hoop Stress σ_θ(r)"

---

## Summary Figure

**Produce after all 8 benchmarks pass.**

**`quad4_summary.png`** — 2×4 grid of panels:

| Row\Col | 1 | 2 | 3 | 4 |
|---------|---|---|---|---|
| 1 | B1: Patch test (σ_x case) | B2: Cook's convergence | B3: Simple shear γ=1.0 | B4: Rollup trajectory |
| 2 | B5: Snap-through P-δ | B6: Lee's frame deformed | B7: Buckling P-δ | B8: Cylinder u_r profile |

Each panel simplified from per-benchmark figure. Green ✓ / red ✗ per panel.
Overall title: "Quad4 — Full Kinematics Validation | Linear · TL · UL · Corot (beams)"

---

## Full Regression

After all 8 benchmarks pass, run in order — zero regressions:

```
examples/truss.py
examples/beam.py
examples/column_buckling.py
examples/corot_benchmarks.py
examples/quad4_benchmarks.py    ← all 8 benchmarks + summary
examples/hex8_benchmarks.py    ← Hex8 must still pass after Quad4 refactor
```

---

## Benchmark Status Table

Update in-place as each benchmark passes.

| # | Benchmark | Kinematics | Status | Plots |
|---|-----------|------------|--------|-------|
| 1 | Patch Test (5-element irregular) | Linear | ⬜ Pending | quad4_b1_patch_test.png |
| 2 | Cook's Membrane | Linear | ⬜ Pending | quad4_b2_cooks_membrane.png |
| 3 | Simple Shear | TL + UL | ⬜ Pending | quad4_b3_simple_shear.png |
| 4 | Cantilever Rollup | TL + UL | ⬜ Pending | quad4_b4_rollup_deformed.png, quad4_b4_rollup_curve.png |
| 5 | Snap-Through Arch | TL + UL | ⬜ Pending | quad4_b5_snapthrough.png |
| 6 | Lee's Frame | Corot beams | ⬜ Pending | quad4_b6_lees_frame.png |
| 7 | Column Buckling | Corot beams | ⬜ Pending | quad4_b7_column_buckling.png |
| 8 | Thick-Walled Cylinder | TL + UL | ⬜ Pending | quad4_b8_cylinder_mesh.png, quad4_b8_cylinder_profiles.png |
| — | Summary | All | ⬜ Pending | quad4_summary.png |

---

## References

- Reissner (1972), ZAMP 23:795 — cantilever rollup closed-form
- Cook (1974), J. Struct. Div. ASCE 100:1851 — Cook's membrane geometry; ref. value 23.91
- Simo & Vu-Quoc (1986), CMAME 58:79 — Lee's frame reference values, Example 7.4
- Lee (1971) — original Lee's frame problem statement
- Timoshenko & Goodier (1970) "Theory of Elasticity" §130 — Lamé cylinder
- Crisfield (1991) Vol. 1 Ch. 9 — snap-through arch, P_cr reference
- Bathe (1996) Ch. 6 — TL/UL strain measures
- Felippa & Haugen (2005), CMAME 194:2285 — EICR for CorotContinuumKinematics (deferred)
- Phase 4b confirmed parameter set: `docs/phase4b_lees_frame_correction.md`
