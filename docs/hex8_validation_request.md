# Hex8 — Full Kinematics Validation Request

## Purpose

This document specifies the complete validation suite for the Hex8 element across all
four kinematics formulations: Linear (B-bar), Total Lagrangian, Updated Lagrangian, and
Corotational. It is a companion to `docs/hex8_design_request.md` and the fixed
implementation plan. Claude Code must read both before proceeding.

**Scope:** All nine benchmarks. Benchmarks 1–4 cover Linear B-bar. Benchmarks 5–9
cover TL, UL, and Corot. All benchmarks live in `examples/hex8_benchmarks.py`.

---

## Visualization Requirements

Every benchmark produces one or more plots saved to `docs/validation/hex8/`.
Create this directory if it does not exist. All plots use matplotlib. The exact
filename for each plot is specified in each benchmark section. Do not skip plots —
they are required outputs, not optional extras.

### General plot style

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
})
SAVE_DIR = 'docs/validation/hex8'
import os; os.makedirs(SAVE_DIR, exist_ok=True)
```

### Mesh plotting utility (implement once, reuse in every benchmark)

Implement a standalone function:

```python
def plot_hex8_mesh(ax, nodes_xyz, connectivity,
                   deformed_xyz=None, field=None, field_label='',
                   alpha=0.35, color='steelblue', title='',
                   azim=210, elev=25):
```

Behaviour:
1. **Reference wireframe:** Always draw all element edges using thin grey lines
   (linewidth=0.5, color='#888888'). The 12 edges of each Hex8 are the 4 bottom
   edges, 4 top edges, and 4 vertical edges connecting them.
2. **Deformed filled faces:** If `deformed_xyz` is provided, draw the 6 faces of
   each deformed element using `Poly3DCollection` with the given `alpha` and `color`.
   Face vertex ordering: bottom=(0,1,2,3), top=(4,5,6,7), front=(0,1,5,4),
   back=(3,2,6,7), left=(0,3,7,4), right=(1,2,6,5).
3. **Field colouring:** If `field` is provided (one scalar per element), colour each
   element's faces by the field value. Use viridis for non-negative fields (strain
   magnitude, displacement magnitude) and RdBu_r for signed fields (stress, u_r).
   Add a colorbar with `field_label`.
4. Set axis labels ('X', 'Y', 'Z'), equal aspect ratio, view angle via
   `ax.view_init(elev=elev, azim=azim)`, and the title.
5. Return the axes object.

---

## Prerequisites

All four Linear B-bar benchmarks must PASS before any nonlinear benchmark runs:

| # | Benchmark | Pass Criterion |
|---|-----------|----------------|
| 1 | 3D Patch Test | All 6 strain states exact to < 1e-12 |
| 2 | Thick-walled cylinder (B-bar proof) | bbar=True within 2% Lamé; bbar=False > 50% error |
| 3 | 3D Cantilever (Linear) | Tip deflection within 5% Euler-Bernoulli |
| 4 | Cook's Membrane 3D (Linear) | Apex disp within 1% of Quad4 PlaneStrain result |

If any of Benchmarks 1–4 fails, stop. Do not proceed to nonlinear kinematics.

---

## What Is Being Validated

| Kinematics | 2D Status | 3D Status After This WP |
|------------|-----------|--------------------------|
| `LinearContinuumKinematics(bbar=True)` | N/A (2D has no B-bar) | Validated by B1–B4 |
| `TotalLagrangianContinuumKinematics` | Validated (Quad4) | Validated by B5–B7 |
| `UpdatedLagrangianContinuumKinematics` | Validated (Quad4) | Validated by B5–B7 |
| `CorotContinuumKinematics` | Stub only | Validated by B8–B9 |

**Corot note:** `CorotContinuumKinematics` is currently a stub (all methods raise
NotImplementedError). B8–B9 require it to be fully implemented. If not yet implemented,
skip B8–B9 and mark them Pending. Do not implement Corot as part of this validation —
that is a separate work package.

---

## Constructor Usage

```python
from oneFEM.model.element.continuum import Hex8
from oneFEM.model.element.kinematics.continuum import (
    TotalLagrangianContinuumKinematics,
    UpdatedLagrangianContinuumKinematics,
    CorotContinuumKinematics,
)

elem = Hex8(tag, nodes, material, kinematics=TotalLagrangianContinuumKinematics())
elem = Hex8(tag, nodes, material, kinematics=UpdatedLagrangianContinuumKinematics())
elem = Hex8(tag, nodes, material, kinematics=CorotContinuumKinematics())  # when implemented
```

---

## Benchmark 1 — 3D Patch Test (Linear B-bar)

### Purpose

Verify shape functions, Jacobian, and B matrix for an arbitrary (non-rectangular)
Hex8 element. The patch test is the minimum necessary condition for convergence.

### Model Definition

Single distorted Hex8 element (perturb corners by ±0.15 in a deterministic pattern
while keeping det(J) > 0). Apply linear displacement BCs at all 8 nodes for each
of 6 independent strain states (a = 0.01):

1. ε_xx: u=a·x, v=0, w=0
2. ε_yy: u=0, v=a·y, w=0
3. ε_zz: u=0, v=0, w=a·z
4. γ_xy: u=a·y, v=a·x, w=0
5. γ_yz: u=0, v=a·z, w=a·y
6. γ_xz: u=a·z, v=0, w=a·x

**Note on B-bar:** For a linear displacement field, B_vol[gp] is identical at all
Gauss points, so B_bar == B_std identically. Run with both bbar=True and bbar=False
and assert the results are identical to < 1e-15. The patch test does NOT exercise the
B-bar modification — that is Benchmark 2.

### Pass Criteria

- Strain at all 8 GPs matches analytical value to < 1e-12 for all 6 load cases.
- bbar=True and bbar=False give identical results (difference < 1e-15).

### Plots

**`hex8_b1_patch_test.png`** — 2×3 figure, one subplot per strain state:
- Each subplot: 3D axes with the distorted Hex8 wireframe (grey) and all 8 Gauss
  points marked as red dots.
- Subplot title: strain state name (e.g. "ε_xx = 0.01") + "Max GP error: {:.2e}"
- Camera: azim=200°, elev=20° (constant).
- Overall title: "B1 — 3D Patch Test | 6 strain states | PASS / FAIL"

---

## Benchmark 2 — Thick-Walled Cylinder, B-bar Proof (Linear)

### Purpose

Demonstrate volumetric locking (bbar=False) and its fix (bbar=True) at ν=0.499.

### Model Definition

Quarter-cylinder: r_i=1.0, r_o=3.0, H=1.0. E=1000, ν=0.499. 4×4×1 Hex8 mesh using
proper cylindrical node coordinates. Internal pressure p_i=1.0.
Symmetry BCs on θ=0 and θ=π/2 cut faces. u_z=0 on both Z faces.

Exact Lamé solution (plane strain):
```
u_r(r) = [p_i*r_i²/(E*(r_o²-r_i²))] * [(1-2ν)*r + (1+ν)*r_o²/r]
```

### Pass Criteria

- bbar=True: u_r at r_i within 2% of Lamé.
- bbar=False: u_r error > 50% (locking confirmed).

### Plots

**`hex8_b2_bbar_proof.png`** — 1×2 figure:

Left: 3D view of the quarter-cylinder mesh using `plot_hex8_mesh`. Show reference
wireframe (grey) and deformed mesh (bbar=True, filled faces coloured by radial
displacement magnitude, viridis). Displacement scale ×20 for visibility.
Camera: azim=225°, elev=25° so the curved inner surface and both planar cut faces
are simultaneously visible. Annotate: "r_i=1, r_o=3, ν=0.499, p_i=1.0, ×20 scale".
Title: "B2 — Deformed Mesh (B-bar ON)"

Right: Grouped bar chart comparing u_r at r=r_i for [Lamé, bbar=True, bbar=False].
bbar=True bar: green. bbar=False bar: red. Annotate each bar with its value and
% error vs Lamé. Title: "B2 — B-bar Locking Fix | ν=0.499"

---

## Benchmark 3 — 3D Cantilever Under Tip Load (Linear)

### Purpose

Verify bending response and multi-element 3D assembly.

### Model Definition

L=10, b=1, h=1. Mesh: 10×1×1 Hex8 elements. E=1000, ν=0.3.
Base (X=0) fully fixed. Tip load P=1 distributed on X=L face.
Reference: δ = PL³/(3EI), I=bh³/12=1/12.

### Pass Criteria

Tip deflection within 5% of Euler-Bernoulli reference.

### Plots

**`hex8_b3_cantilever_linear.png`** — 1×2 figure, side-by-side 3D views:

Left: Reference mesh as grey wireframe only. Camera: azim=200°, elev=20°.
Title: "Reference Configuration"

Right: Deformed mesh using `plot_hex8_mesh`. field = von Mises stress per element
(σ_vm = √(3/2·s:s), deviatoric stress s = σ − (tr σ/3)·I, averaged from 8 GPs to
element centroid). Colormap: hot_r. Actual displacement scale (no exaggeration needed).
Same camera as left. Annotate: "Tip δ_y={:.4f}\nRef={:.4f}\nError={:.1f}%"
Title: "Deformed | von Mises Stress"

Overall title: "B3 — Linear Cantilever | 10×1×1 Hex8"

---

## Benchmark 4 — Cook's Membrane 3D (Linear)

### Purpose

Cross-validate 3D Hex8 against the existing Quad4 PlaneStrain result.

### Model Definition

Cook's membrane geometry (trapezoid: 0,0 → 48,44 → 48,60 → 0,44), extruded W=1.0
in z. 1 element thick, u_z=0 on both z-faces (plane strain). Left edge fixed.
Right edge: distributed shear V=1/unit-height.
In-plane mesh matches Quad4 benchmark density.

### Pass Criteria

Apex y-displacement within 1% of Quad4 PlaneStrain result.

### Plots

**`hex8_b4_cooks_membrane.png`** — 1×2 figure:

Left: 3D extruded mesh using `plot_hex8_mesh`. Show reference wireframe (grey) and
deformed mesh (colour = y-displacement, RdBu_r diverging colormap). Displacement
scale ×5. Camera: azim=250°, elev=30° so both the tapered 2D shape and the extrusion
depth are clearly visible — this is the key visual confirming this is a Hex8 model.
Title: "B4 — Cook's Membrane 3D | Deformed (×5 scale)"

Right: Mesh convergence plot. X-axis: in-plane elements per side (2, 4, 8, 16).
Y-axis: apex y-displacement. Two curves: Hex8 3D (blue circles, solid) and
Quad4 2D PlaneStrain (dashed black). Horizontal dashed line at the fine-mesh
reference value. Title: "B4 — Convergence vs Quad4 Reference"

---

## Benchmark 5 — 3D Simple Shear Kinematic Unit Test (TL and UL)

### Purpose

Verify F, E (TL Green-Lagrange), and e (UL Almansi) in 3D. Pure kinematics unit test —
no solver. Displacements applied directly to nodes. Pass criterion: machine precision.

### Model Definition

Single Hex8 element, unit cube [0,1]³. Simple shear in x-y plane:
```
u_x = γ·Y,   u_y = 0,   u_z = 0
```
Run for γ = 0.1, 0.5, 1.0, 2.0. Apply nodal displacements directly without a solver.

### Analytical Reference (γ = 0.5)

**F (3×3):**
```
[[1.0, 0.5, 0.0],
 [0.0, 1.0, 0.0],
 [0.0, 0.0, 1.0]]
```

**TL Green-Lagrange, Voigt [E_11, E_22, E_33, Γ_12, Γ_23, Γ_13]:**
```
[0.000, 0.125, 0.000, 0.500, 0.000, 0.000]
```

**UL Almansi, Voigt [e_11, e_22, e_33, γ_12, γ_23, γ_13]:**
```
[0.000, -0.125, 0.000, 0.500, 0.000, 0.000]
```

Note: E_22 = +0.125 but e_22 = −0.125. This sign difference is correct and expected.
det(F) = 1.0 (isochoric) for all γ.

### Pass Criteria

F, E, e, and det(F) at all 8 GPs match analytical values to < 1e-14 for all γ.
Any error > 1e-12 is a bug in 3D B_NL or F assembly. Fix before proceeding.

**Warning on Voigt ordering:** If Γ_12 = 0.25 instead of 0.5, there is a factor-of-2
error in the 3D shear Voigt storage. Confirm the ordering against
`ElasticIsotropic(type='3D')` before running this test.

### Plots

**`hex8_b5_simple_shear.png`** — 2×2 figure, one 3D subplot per γ value:

Each subplot: grey wireframe = reference cube. Filled semi-transparent faces
(alpha=0.4, color='cornflowerblue') = deformed element using `plot_hex8_mesh`.
At γ=1.0 and γ=2.0 the shear should be visually dramatic — originally rectangular
faces become strongly skewed parallelograms.
Mark the 8 Gauss points in the deformed configuration as red dots.
Subplot title: "γ={:.1f} | max F err: {:.1e} | max E err: {:.1e}"
Camera: azim=200°, elev=20° (constant across all subplots for easy comparison).
Overall title: "B5 — 3D Simple Shear | TL (GL) and UL (Almansi) | Machine Precision"

---

## Benchmark 6 — 3D Cantilever Rollup (TL and UL)

### Purpose

Primary validation for TL and UL. Closed-form solution at every load level (Reissner
1972). TL and UL must produce identical results — any discrepancy is a bug.

### Model Definition

**Geometry:** L=10.0, H=1.0, W=1.0. Mesh: 40×2×1 Hex8 elements.

**Material:** E=1.2×10⁶, ν=0.0, `type='3D'`. ν=0 is required to match the classical
solution — any nonzero ν introduces 3D Poisson coupling with no clean analytical correction.

**Boundary conditions:**
- Left face (X=0): all nodes fully fixed.
- Front (Z=0) and back (Z=W) faces: u_z=0 (plane strain symmetry).

**Loading:** M_full = EI·2π/L where I = WH³/12.
Apply as a force couple on right-face tip nodes (top edge: +F_y, bottom edge: −F_y,
distributed evenly with half-weight at corners). 20 equal load steps.

### Analytical Reference (Reissner 1972)

For M = EI·θ/L:
```python
x_tip = (L / theta) * sin(theta)
y_tip = (L / theta) * (1 - cos(theta))
```

| Step | θ (rad) | x_tip   | y_tip  |
|------|---------|---------|--------|
| 5    | π/2     | 6.366   | 6.366  |
| 10   | π       | 0.000   | 6.366  |
| 15   | 3π/2    | −6.366  | 6.366  |
| 20   | 2π      | 0.000   | 0.000  |

At step 20 the tip returns exactly to its starting position. Any accumulated
kinematic error shows as a nonzero tip displacement.

### Pass Criteria

- Tip position within 1% of analytical at all 4 check steps.
- TL and UL agree to 6 significant figures at every step.
- Newton-Raphson converges within 10 iterations per step.
- **Quadratic Newton-Raphson convergence is a formal pass criterion.** Print and plot
  residual norms at steps 5, 10, 15, 20. Linear convergence = missing K_σ in 3D.

### Plots

This benchmark produces two figures.

**`hex8_b6_rollup_deformed.png`** — 2×2 figure, one 3D subplot per check step (θ = π/2, π, 3π/2, 2π):

Each subplot:
- Reference straight beam as a thin grey wireframe behind the deformed mesh.
- Deformed mesh drawn using `plot_hex8_mesh`, coloured by Green-Lagrange strain
  magnitude ‖E‖ = √(E:E) (scalar per element, averaged from 8 GPs). Colormap viridis
  with shared vmin/vmax across all 4 subplots (use the global max across all steps).
- **The extrusion depth (W=1.0) must be clearly visible** — this confirms 3D Hex8
  and not a flat 2D mesh. Camera: azim=195°, elev=25° (constant).
- Tip node marked with a red star.
- Subplot annotation: "Step {i} | θ={:.2f}π | x_tip={:.3f} | y_tip={:.3f}"
- Subplot title: "TL (solid) = UL (dashed) — indistinguishable if correct"
Overall figure title: "B6 — Cantilever Rollup | TL and UL"

**`hex8_b6_rollup_curve.png`** — 1×2 figure:

Left (tip trajectory):
- X-Y plane plot. TL trajectory as solid blue line, UL as dashed orange line,
  analytical as black dots at each of the 20 load steps.
- Mark the 4 check steps with colored squares (π/2=green, π=red, 3π/2=purple, 2π=black).
- Equal aspect ratio. Labels: "x_tip", "y_tip".
- Title: "Tip Trajectory | TL vs UL vs Analytical"

Right (Newton convergence):
- Semilogy plot. One curve per check step (steps 5, 10, 15, 20).
- Draw a dashed reference slope for quadratic convergence (slope=2 on log scale).
- Title: "Newton Residual Norm | Steps 5, 10, 15, 20"

---

## Benchmark 7 — Thick-Walled Cylinder (TL and UL)

### Purpose

Validates TL and UL under volumetric deformation. Cross-validates Linear, TL, and UL:
at small load all three agree; at large load TL/UL diverge from Linear but agree with
each other.

### Model Definition

Quarter-cylinder: r_i=1.0, r_o=3.0, H=1.0. E=1000, ν=0.3, `type='3D'`.
4×4×1 Hex8 mesh, proper cylindrical node coordinates.
Symmetry BCs on θ=0 and θ=π/2 cut faces. u_z=0 on both Z faces.
Outer surface free. Inner surface: radial pressure P_in as nodal forces.

Two load cases: P_in=10 (linear regime) and P_in=100 (nonlinear regime).
10 load steps each.

Lamé solution (plane strain):
```
u_r(r) = P_in*a²/(E*(b²-a²)) * [(1-2ν)*r + (1+ν)*b²/r]
u_r(r_i): P_in=10 → 0.015125,   P_in=100 → 0.15125
```

### Pass Criteria

Small load: Linear, TL, UL all within 2% of Lamé. All three agree to 0.1%.
Large load: TL and UL agree to 1%. Linear may differ from TL/UL by up to 15% (expected).
Both TL and UL show quadratic Newton convergence.

### Plots

This benchmark produces two figures.

**`hex8_b7_cylinder_mesh.png`** — 1×3 figure, three 3D subplots:

Left: Reference mesh only, grey wireframe. Camera: azim=210°, elev=30° so the curved
inner surface, both planar cut faces, and the flat end caps are all visible.
Title: "B7 — Quarter-Cylinder Reference Mesh | 4×4×1 Hex8"

Centre: Deformed mesh at P_in=10 (bbar=True, Linear for visual clarity),
coloured by radial displacement u_r (viridis). Displacement scale ×10 (small load
needs exaggeration). Title: "Deformed | P_in=10 (×10 scale)"

Right: Deformed mesh at P_in=100 (TL result), coloured by radial displacement u_r.
Actual scale. Title: "Deformed | P_in=100 (TL, actual scale)"
Camera same as left for all three subplots.

**`hex8_b7_cylinder_radial.png`** — 1×2 figure:

Left (radial displacement profile):
- X-axis: radius r from r_i to r_o. Y-axis: radial displacement u_r.
- For P_in=10: Lamé (solid black), Linear (dashed blue), TL (dashed green), UL (dashed orange).
- For P_in=100: same color scheme but with circle markers and brighter lines.
- Extract u_r from nodes along the θ=0 cut face at each radial position.
- Title: "B7 — Radial Displacement Profile"

Right (formulation comparison bar chart):
- Grouped bars: [Lamé, Linear, TL, UL] for P_in=10 and P_in=100 (side by side).
- Annotate bars with % error vs Lamé.
- Title: "B7 — u_r at Inner Radius | Formulation Comparison"

---

## Benchmark 8 — 3D Column Buckling (Corotational)

**Prerequisite:** `CorotContinuumKinematics` must be fully implemented (not a stub).
If still a stub, skip and mark Pending.

### Purpose

Verify `CorotContinuumKinematics` in 3D. Solid-element analog of `column_buckling.py`.

### Model Definition

**Geometry:** L=10.0, b=1.0, h=1.0. Mesh: 10×2×2 Hex8 elements (80 elements).
A 1×1 cross-section mesh suppresses the buckling mode — 2×2 is the minimum.

**Material:** E=1000, ν=0.3, `type='3D'`.

**BCs:** Base (Z=0) fully fixed. Top (Z=L): u_x=u_y=0, u_z free (guided end).

**Loading:** Displacement control. Apply u_z at top face, increasing to δ_max=1.0
in 50 steps. Track axial reaction R_z at base.

**Euler critical load (fixed-free, effective length=2L):**
```
P_cr = π²EI/(2L)²   where I=bh³/12=1/12   →   P_cr = 2.056
```

### Pass Criteria

- Max R_z within 5% of P_cr = 2.056.
- Post-buckling softening visible on the load-displacement curve.
- Lateral buckling mode shape visible in deformed geometry.
- Newton converges for at least 80% of steps.

### Plots

**`hex8_b8_buckling.png`** — 1×3 figure:

Left (load-displacement curve):
- X-axis: applied δ_z (0 to δ_max). Y-axis: base reaction R_z.
- Solid blue line for the computed curve.
- Horizontal dashed red line at Euler P_cr = 2.056.
- Mark the computed maximum R_z with a red star and annotate its value.
- Shade the pre-buckling region (R_z < P_cr) in light green, post-buckling in light red.
- Title: "B8 — Column Buckling | P_cr={:.3f} (Euler={:.3f})"

Centre (buckled shape, 3D):
- Use `plot_hex8_mesh` to show the deformed column at the step of maximum R_z.
  Displacement scale ×5 for visual clarity.
- Reference configuration as thin grey wireframe behind the deformed mesh.
- Deformed mesh coloured by lateral displacement u_x (RdBu_r, centred at 0).
  This makes the bending mode clearly visible as a red-to-blue gradient.
- Camera: azim=180°, elev=10° (near side-view to maximise apparent lateral deflection).
- Title: "Buckled Shape (×5 scale) | Step={step_at_Pcr}"

Right (Newton iterations per step):
- Bar chart. X-axis: step number (0–50). Y-axis: iterations used.
- Green bars for converged steps, red bars for diverged steps.
- Vertical dashed line at the P_cr step.
- Title: "Newton Iterations Per Step"

---

## Benchmark 9 — 3D Snap-Through Arch (Corotational)

**Prerequisite:** Same as B8. Skip if Corot is not implemented.

### Purpose

Validate `CorotContinuumKinematics` on a path-following problem with a limit point.
Solid-element analog of the snap-through arch in `corot_benchmarks.py`.

### Model Definition

**Geometry:** Shallow circular arch: rise H=0.5, half-span=5.0.
R = (5²+0.5²)/(2×0.5) = 25.25. Extruded W=1.0 in z, 1 element thick.
Mesh: 10×2×1 Hex8 elements (10 along arc, 2 through arch depth, 1 through width).

**Material:** E=10⁴, ν=0.0, `type='3D'`.

**BCs:** Both supports pinned (u_x=u_y=u_z=0). Front/back faces: u_z=0 (plane strain).

**Loading:** Displacement control at apex. Steps of δ=0.05 until δ_max=0.75 or
snap-through. Track apex reaction R_y.

**Reference:** From `corot_benchmarks.py`, `P_CR_ARCH_REF = 0.5878`.
Verify this applies to the same geometry before comparing.

### Pass Criteria

- Max R_y within 5% of reference P_cr.
- Qualitative match to beam result from `corot_benchmarks.py`.
- Graceful divergence past limit point.
- Quadratic Newton convergence for all pre-snap steps.

### Plots

**`hex8_b9_snapthrough.png`** — 1×3 figure:

Left (load-displacement curve):
- X-axis: apex downward displacement δ. Y-axis: apex reaction R_y.
- Hex8 Corot result as solid blue line.
- Overlay beam result from `corot_benchmarks.py` as dashed black (same geometry only).
- P_cr marked with red star and horizontal dashed line.
- Title: "B9 — Snap-Through Arch | P_cr={:.4f} (ref={:.4f})"

Centre (arch shape sequence, 3D):
- 3D axes. Draw the arch front face outline (top-face mid-span nodes) at 4 load
  levels: 25%, 50%, 75% of P_cr, and at snap-through. Colour progresses blue→red.
- Reference arch shown as solid grey line.
- The extrusion depth W=1.0 must be visible — draw the deformed front-face outline
  and the deformed back-face outline as dashed lines, connected at the supports.
- Camera: azim=190°, elev=20°.
- Title: "Arch Profile During Loading"

Right (Newton iterations per step):
- Same style as B8 right panel. Mark the snap-through step with a vertical dashed line.
- Title: "Newton Iterations Per Step"

---

## Summary Figure

**Produce after all 9 benchmarks pass.**

**`hex8_summary.png`** — 3×3 grid of panels:

| Row\Col | 1 | 2 | 3 |
|---------|---|---|---|
| 1 | B1: Patch test GP error (bar chart) | B2: B-bar locking comparison | B3: Cantilever deformed (3D) |
| 2 | B4: Cook's membrane convergence | B5: Simple shear γ=1.0 (3D) | B6: Rollup tip trajectory |
| 3 | B7: Radial displacement profile | B8: Column buckling P-δ | B9: Arch snap-through P-δ |

Each panel is a simplified single-plot excerpt from the per-benchmark figures.
Add a green ✓ or red ✗ label on each panel per pass/fail status.
Overall title: "Hex8 — Full Kinematics Validation | Linear (B-bar) · TL · UL · Corot"

---

## Full Regression

After all 9 benchmarks pass, run in order — zero regressions:

```
examples/truss.py
examples/beam.py
examples/column_buckling.py
examples/corot_benchmarks.py
examples/quad4_patch_test.py
examples/quad4_cooks_membrane.py
examples/hex8_benchmarks.py    ← all 9 benchmarks + summary figure
```

---

## Benchmark Status Table

Update in-place as each benchmark passes.

| # | Benchmark | Kinematics | Status | Plots |
|---|-----------|------------|--------|-------|
| 1 | 3D Patch Test | Linear B-bar | ⬜ Pending | hex8_b1_patch_test.png |
| 2 | Thick-walled cylinder (B-bar proof) | Linear B-bar | ⬜ Pending | hex8_b2_bbar_proof.png |
| 3 | 3D Cantilever | Linear B-bar | ⬜ Pending | hex8_b3_cantilever_linear.png |
| 4 | Cook's Membrane 3D | Linear B-bar | ⬜ Pending | hex8_b4_cooks_membrane.png |
| 5 | Simple Shear 3D | TL + UL | ⬜ Pending | hex8_b5_simple_shear.png |
| 6 | Cantilever Rollup 3D | TL + UL | ⬜ Pending | hex8_b6_rollup_deformed.png, hex8_b6_rollup_curve.png |
| 7 | Thick-walled cylinder | TL + UL | ⬜ Pending | hex8_b7_cylinder_mesh.png, hex8_b7_cylinder_radial.png |
| 8 | Column Buckling 3D | Corot | ⬜ Pending | hex8_b8_buckling.png |
| 9 | Snap-Through Arch 3D | Corot | ⬜ Pending | hex8_b9_snapthrough.png |
| — | Summary | All | ⬜ Pending | hex8_summary.png |

---

## References

- Reissner (1972), "On one-dimensional finite-strain beam theory", ZAMP 23:795
- Lamé solution — Timoshenko & Goodier (1970) "Theory of Elasticity", §130
- Timoshenko & Gere (1961), "Theory of Elastic Stability" — snap-through arch
- Felippa & Haugen (2005), "A unified formulation of small-strain corotational finite
  elements: I. Theory", CMAME 194:2285 — reference for CorotContinuumKinematics
- Bathe (1996), "Finite Element Procedures", Ch. 6 — TL/UL strain measures
- Hughes (1980), "Generalization of selective integration procedures", IJNME 15:1413 — B-bar
- Existing oneFEM: `examples/quad4_patch_test.py`, `examples/column_buckling.py`,
  `examples/corot_benchmarks.py` — primary reference implementations to extend
