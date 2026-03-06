# oneFEM Unified Kinematics — Validation & Benchmark Schedule
**Version 1.0 — March 2026**

---

## 1. Overview and Purpose

This document defines the complete validation and benchmarking schedule for the oneFEM unified kinematics system. It covers all four geometric nonlinearity formulations — Linear, Corotational, Total Lagrangian (TL), and Updated Lagrangian (UL) — across all element types. Each benchmark is fully specified: geometry, material parameters, boundary conditions, loading, reference solution, pass/fail criteria, numerical modeling tips, and known pitfalls.

The schedule is organized in strict priority order. A formulation is not considered validated until all benchmarks in its tier pass. TL and UL are always validated together on the same problems — agreement between them is itself a primary validation criterion.

### Master Benchmark Table

| Priority | Benchmark | Formulation(s) | Element(s) | Status |
|----------|-----------|----------------|------------|--------|
| 1 | Patch Test | Linear | Quad4 only (Tri3 deferred to future WP) | Pending |
| 2 | Cook's Membrane | Linear | Quad4 | Pending |
| 3 | Simple Shear (kinematic unit test) | TL, UL | Quad4 | Pending |
| 4 | Large-Deformation Cantilever | TL, UL | Quad4 | Pending |
| 5 | Snap-Through Arch | TL, UL, Corot | Quad4 / Truss | Pending |
| 6 | Lee's Frame | Corot | Beam (CrdTransf) | Pending |
| 7 | Williams Toggle Frame | Corot | Beam (CrdTransf) | Pending |
| 8 | Thick-Walled Cylinder | TL, UL | Quad4 | Pending |

### Decision Gates

Each gate must pass before the next is attempted. A gate failure means stop, debug, fix, re-run.

| Gate | Action | Condition to proceed |
|------|--------|----------------------|
| G0 | Fix all broken nodes (Node22, Node33, etc.) | All node types instantiate, commit, revert correctly |
| G1 | Implement LinearContinuumKinematics + Quad4 | Patch test passes for all 3 load cases |
| G2 | Validate Quad4 element quality | Cook's membrane within 0.5% of reference on 16x16 mesh |
| G3 | Implement TLContinuumKinematics | Simple shear: F and E exact to machine precision |
| G4 | Implement ULContinuumKinematics | Simple shear: F and e exact; TL and UL agree on cantilever |
| G5 | Full large-deformation validation | Cantilever rollup within 1% at all load steps |
| G6 | Path-following validation | Snap-through arch: limit load within 5% of reference under displacement control; graceful divergence past limit point *(pre-snap only — full path deferred to arc-length WP)* |
| G7 | Beam Corotational (re-validate after refactor) | Lee's frame within 2% of Simo & Vu-Quoc |

---

## 2. Benchmark 1 — Patch Test (Linear)

### Purpose

The patch test is the minimum necessary condition for finite element convergence. It verifies
that the element can represent a state of constant stress exactly on an irregular mesh. If a
Quad4 fails the patch test, nothing else is worth testing. This must be the first test run
after any new element implementation or kinematics refactor.

> **Scope note:** This WP runs the patch test on **Quad4 only**. Tri3 is deferred to the
> Tri3 porting WP. The Tri3 tip at the end of this benchmark section is retained for
> reference when that WP begins.

### Model Definition

**Geometry — MacNeal-Harder 5-element patch:**

Unit square domain [0,1] × [0,1]. One internal node at an *irregular* location surrounded by 4 quadrilateral elements. The interior node must NOT be at the centroid — irregularity is the whole point.

| Node | x | y | Type |
|------|-------|-------|------|
| 1 | 0.000 | 0.000 | Corner |
| 2 | 1.000 | 0.000 | Corner |
| 3 | 1.000 | 1.000 | Corner |
| 4 | 0.000 | 1.000 | Corner |
| 5 | 0.500 | 0.000 | Edge midpoint |
| 6 | 1.000 | 0.500 | Edge midpoint |
| 7 | 0.500 | 1.000 | Edge midpoint |
| 8 | 0.000 | 0.500 | Edge midpoint |
| 9 | 0.240 | 0.220 | **Interior (irregular)** |

**Material:** Plane stress, linear elastic, E = 1.0, ν = 0.25, t = 1.0.

**Boundary conditions and loading:**

Prescribe nodal displacements on ALL 8 boundary nodes consistent with a known constant stress state. Do NOT apply forces — this is displacement-controlled.

Run three independent load cases. For each, prescribe the following nodal displacements (plane stress, E = 1.0, ν = 0.25, G = E / (2(1+ν)) = 0.4):

**Load case 1 — pure σ_x = 1, σ_y = 0, τ_xy = 0:**
```
u(x, y) = x / E          =  x
v(x, y) = -ν * y / E     = -0.25 * y
```

**Load case 2 — pure σ_y = 1, σ_x = 0, τ_xy = 0:**
```
u(x, y) = -ν * x / E     = -0.25 * x
v(x, y) = y / E           =  y
```

**Load case 3 — pure τ_xy = 1, σ_x = 0, σ_y = 0:**
```
u(x, y) = y / (2G)        =  1.25 * y
v(x, y) = x / (2G)        =  1.25 * x
```

Prescribe these at all 8 boundary nodes. Leave node 9 (interior) free.

### Pass Criteria

- Stress at ALL Gauss points (5 elements × 4 GP = 20 values) must match the prescribed constant stress state to machine precision (relative error < 1e-10).
- No spurious rotations or hourglass modes.
- Run three independent load cases: pure σ_x, pure σ_y, and pure τ_xy. All three must pass.

### Numerical Tips

- Always run all three load cases. A buggy B matrix can pass one and fail another.
- If the test fails, print the B matrix at each GP and verify it satisfies the integration identity `∫ B dV = boundary term`. This localizes the bug immediately.
- The interior node displacement should equal the analytical field exactly. If not, the stiffness assembly or DOF numbering is wrong.
- *Tri3 is out of scope for this WP. When ported in a future WP, run a 4-triangle patch sharing one interior node — Tri3 passes exactly due to its linear displacement field.*

> **Warning:** The patch test is sensitive to Jacobian sign errors. A sign error in det(J) produces incorrect stiffness but may still assemble without runtime error. Print det(J) for each GP during debugging and verify it is strictly positive.

---

## 3. Benchmark 2 — Cook's Membrane (Linear)

### Purpose

Cook's membrane (Cook 1974) is the standard benchmark for 2D quadrilateral element quality under bending-dominated loading on an irregular, distorted mesh. It simultaneously tests membrane and bending performance. The converged tip displacement reference value is well-established, making it a reliable quantitative convergence check.

### Model Definition

**Geometry — tapered panel (plane stress):**

| Corner | x (mm) | y (mm) | Location |
|--------|--------|--------|----------|
| 1 | 0 | 0 | Bottom-left (fixed) |
| 2 | 48 | 44 | Bottom-right (free) |
| 3 | 48 | 60 | Top-right (free, load point) |
| 4 | 0 | 44 | Top-left (fixed) |

**Material:** Plane stress, E = 1.0, ν = 1/3, t = 1.0.

**Boundary conditions and loading:**
- Left edge (nodes 1 and 4): fully fixed — u_x = u_y = 0.
- Right edge: distributed shear load V = 1.0 N total, applied as nodal forces proportional to tributary edge length.
- No body forces.

### Mesh Convergence Study

| Mesh | Elements | Expected tip disp. (u_y at top-right) | Tolerance |
|------|----------|----------------------------------------|-----------|
| 1×1 | 1 | ~11.8 | Coarse — **do not judge element quality from this result; a single distorted Quad4 under bending will lock noticeably** |
| 2×2 | 4 | ~22.9 | ±0.1 |
| 4×4 | 16 | ~23.5 | ±0.05 |
| 8×8 | 64 | ~23.8 | ±0.02 |
| 16×16 | 256 | **23.91 (converged)** | ±0.01 |

### Pass Criteria

- 16×16 mesh tip displacement within 0.1 of 23.91.
- Monotonic convergence from coarse to fine mesh.
- No locking — displacement must not be severely underestimated on coarse mesh.

### Numerical Tips

- Apply distributed shear load correctly: for N nodes on the right edge, interior nodes get V/N and corner nodes get V/(2N).
- The reference value 23.91 is the fine-mesh limit. Published coarse-mesh values vary significantly between element formulations — do not use coarse-mesh values as pass criteria.
- This benchmark stresses elements with poor aspect ratios and skew angles. Incorrect isoparametric mapping for non-rectangular elements will be revealed here.
- Always verify that the isoparametric Jacobian is positive definite at all Gauss points for distorted meshes.

> **Warning:** If tip displacement is significantly *above* 23.91 on a fine mesh, check your Voigt convention. Over-flexible results typically indicate a factor-of-2 error on shear terms in the B matrix.

---

## 4. Benchmark 3 — Simple Shear (TL and UL — Kinematic Unit Test)

### Purpose

Simple shear is not a standard engineering benchmark but is the most important kinematic verification test. The deformation gradient F, Green-Lagrange strain E, and Almansi strain e are all known analytically. This test verifies kinematic formulas directly, independently of any solver, algorithm, or other elements. It should be run as an automated unit test at every commit.

### Model Definition

**Geometry:** Single Quad4 element, unit square [0,1] × [0,1].

**Material:** Any isotropic elastic material. E = 1000, ν = 0.0 for simplicity.

**Prescribed deformation** — simple shear with parameter γ, applied as nodal displacements directly (no solver):
```
x = X + γ·Y    →    u_x = γ·Y
y = Y          →    u_y = 0
```

For γ = 0.5:

| Node | X | Y | u_x | u_y |
|------|---|---|-----|-----|
| 1 | 0 | 0 | 0.000 | 0 |
| 2 | 1 | 0 | 0.000 | 0 |
| 3 | 1 | 1 | 0.500 | 0 |
| 4 | 0 | 1 | 0.500 | 0 |

### Analytical Reference Values (γ = 0.5)

**Deformation gradient:**
```
F = [[1, 0.5],
     [0, 1.0]]
```

**Green-Lagrange strain (TL):** E = 0.5 * (F^T F - I)
```
E = [[0.00,  0.25],
     [0.25,  0.125]]

In Voigt (engineering shear):  [E_11, E_22, Gamma_12] = [0.0, 0.125, 0.5]
```

**Almansi strain (UL):** e = 0.5 * (I - F^{-T} F^{-1})
```
F^{-1} = [[1, -0.5],
           [0,  1.0]]

e = [[0.0,   0.25],
     [0.25, -0.125]]   (note E_22 ≠ e_22 — this is expected)

In Voigt (engineering shear):  [e_11, e_22, gamma_12] = [0.0, -0.125, 0.5]
```

**Volume:** det(F) = 1.0 (simple shear is isochoric).

### Pass Criteria

- F at the centroid Gauss point matches the prescribed deformation gradient to machine precision (< 1e-14).
- E (TL) at all Gauss points matches the analytical Green-Lagrange strain to machine precision.
- e (UL) at all Gauss points matches the analytical Almansi strain to machine precision.
- det(F) = 1.0 to machine precision.
- Run for γ = 0.1, 0.5, 1.0, 2.0. Errors should be at machine precision for all values.

### Numerical Tips

- This test bypasses the solver entirely — apply displacements directly to nodes and call `update()` on the kinematics manually.
- If E is correct but e is wrong, the issue is in the Almansi term (F^{-T} F^{-1}), not in F itself. Compute and print F^{-1} separately to isolate.
- Add this test to the automated test suite (`tests/kinematics/test_simple_shear.py`) and run it at every commit.

> **Warning:** A common bug — using engineering shear convention (γ_12 = 2ε_12) inconsistently between TL and UL. The Green-Lagrange off-diagonal E_12 = 0.25 for γ=0.5, NOT 0.5. If your output shows 0.5, you have a factor-of-2 error in Voigt shear storage.

---

## 5. Benchmark 4 — Large-Deformation Cantilever Rollup (TL and UL)

### Purpose

A cantilever subjected to a tip moment large enough to produce a complete 360° rollup has a closed-form analytical solution for tip position at every load level. This is the primary validation problem for TL and UL continuum kinematics. TL and UL must produce identical results — any discrepancy is a bug, not a tolerance issue.

### Model Definition

**Geometry:**
- Length L = 10.0, height H = 1.0 (slender: H/L = 0.1)
- 2D plane stress, single layer of Quad4 through thickness
- Recommended mesh: 40×2 elements (40 along length, 2 through height)

**Material:** Linear elastic (geometric nonlinearity only), E = 1.2×10⁶, ν = 0.0 (use ν = 0 to match classical solution — no Poisson effect), plane stress.

**Boundary conditions and loading:**
- Left end: all nodes fully fixed (u_x = u_y = 0).
- Right end tip: applied moment M distributed as a force couple on top and bottom tip nodes.
- M_full = EI·2π/L (produces exactly 360° rotation at full load).
- Use 20 equal load steps, each = 5% of M_full.

### Analytical Reference Solution

For tip moment M = EI·θ/L where θ is total rotation in radians:
```python
x_tip = (L / theta) * sin(theta)
y_tip = (L / theta) * (1 - cos(theta))
```

| Load step | M / M_full | θ (rad) | x_tip | y_tip |
|-----------|------------|---------|-------|-------|
| 5 | 0.25 | π/2 | 6.366 | 6.366 |
| 10 | 0.50 | π | −6.366 | 0.000 |
| 15 | 0.75 | 3π/2 | −6.366 | −6.366 |
| 20 | 1.00 | 2π | **0.000** | **0.000** |

Note: at full load (θ = 2π), the tip returns exactly to its starting position.

### Pass Criteria

- Tip coordinates at each load step within 1% of analytical values.
- TL and UL results agree to at least 6 significant figures at every load step.
- Newton-Raphson converges within 10 iterations at each step.
- **Quadratic convergence of Newton-Raphson is a formal pass criterion.** At each load step,
  the residual norm must decrease quadratically (ratio of successive residuals ≈ residual²).
  Print the residual norm at each Newton iteration for steps 5, 10, 15, and 20. Linear or
  sub-linear convergence indicates incorrect or missing K_σ. This criterion catches wrong K_σ
  sign, missing K_σ, and incorrect material tangent simultaneously.
- No divergence before the full load is applied.

### Numerical Tips

- Use ν = 0. The classical rollup solution assumes pure bending with no Poisson coupling. A nonzero ν introduces axial stress that has no clean analytical correction.
- Apply moment as a force couple (force pair on top and bottom tip nodes), not as a nodal rotation — continuum elements have no rotational DOFs.
- 20 load steps is a minimum. If Newton-Raphson fails before θ = π/4, the geometric stiffness K_σ is wrong or missing. Bisect the step to localize the failure.
- For TL: at full rollup, print F at the fixed-end Gauss points. It should be a pure rotation matrix: det(F) = 1 and F^T·F = I to machine precision.
- For UL: print the reference coordinates at mid-span every 5 steps. They must track the deformed geometry, not remain at the original coordinates.

> **Warning:** Omitting K_σ from the tangent stiffness will produce apparent convergence at small loads (F ≈ I there) but divergence around θ ≈ π/4. If convergence fails at a specific load level and not before it, add K_σ before debugging anything else.

> **Warning:** Do not use reduced integration (1-point) for this problem. Hourglass modes may suppress locking but give incorrect large-deformation kinematics. Use 2×2 full integration.

---

## 6. Benchmark 5 — Snap-Through Arch (TL, UL, Corotational)

### Purpose

The snap-through arch under central point load tests the ability to trace load-displacement paths through limit points (local maxima of load-carrying capacity) and snap-back behavior. It validates both the kinematics formulation and the nonlinear solution algorithm. Arc-length control (Riks method) is required. Reference in Crisfield (1991, Vol. 1, Ch. 9).

### Model Definition

**Geometry — shallow circular arch:**
- Rise H = 0.5, half-span L = 5.0
- Radius R = (L² + H²) / (2H) = 25.25
- Half-angle α = arcsin(L/R) ≈ 11.31°
- Mesh: 10×2 Quad4 elements along arc (or equivalent)
- Use full arch, not half-model — snap-through may involve asymmetric modes

**Material:** E = 10⁴, ν = 0.0, plane stress, unit thickness, cross-section area A = 0.1.

**Boundary conditions and loading:**
- Both ends pinned: u_x = u_y = 0 (no rotational DOF for continuum elements)
- Central point load P applied downward at apex
- Arc-length control required; load control cannot pass the limit point

### Pass Criteria

**In this WP (arc-length control not yet available — pre-snap validation only):**
- Limit point load P_cr within 5% of reference under displacement control.
- Solver diverges gracefully past the limit point (no false equilibrium, no crash).
- TL and UL agree on limit load to 4 significant figures.
- **Quadratic convergence of Newton-Raphson is a formal pass criterion** for load steps
  below the limit point. Print residual norms at each Newton iteration for 3 pre-snap steps.
  Linear or sub-linear convergence indicates missing or incorrect K_σ.

**Deferred to analysis pipeline WP (arc-length required):**
- Full load-displacement curve tracing through and beyond the limit point.
- Recovery of the post-snap unstable branch.
- Matching Crisfield (1991) Fig. 9.17 qualitatively.

### Numerical Tips

- If arc-length control is not yet implemented: use displacement control on the apex node and look for the snap-through in the reaction force vs. apex displacement curve. This will miss the unstable branch but validates the kinematics.
- Use exact arc coordinates (not a polygonal approximation) — snap-through load is sensitive to the rise/span ratio.
- Monitor det(K_T) along the load path. det(K_T) = 0 at the limit point; plotting it vs. load helps locate the bifurcation.
- Add a tiny imperfection (0.001% of rise) at the apex to trigger asymmetric snap-through if the symmetric mode locks the solver at the limit point.
- Arc-length step size: δs ≈ 1% of P_cr for reliable tracing through the limit point.

> **Warning:** If TL and UL give significantly different limit loads, the UL reference configuration update at `commitState()` is buggy. The reference update must happen after convergence, not during iteration.

---

## 7. Benchmark 6 — Lee's Frame (Corotational, Beams)

### Purpose

Lee's right-angle frame (Lee 1971, Simo & Vu-Quoc 1986) is the standard large-rotation beam benchmark. Rotations exceed 90°, making it a severe test of the Corotational transformation. Reference values are tabulated in Simo & Vu-Quoc (1986, CMAME 58), which is the definitive source.

### Model Definition

**Geometry:** Right-angle frame — horizontal member and vertical member, each L = 10.

**Material and section:**
- E = 7.2×10⁶, ν = 0.3
- Rectangular cross-section: b = h = 0.3 → I = bh³/12 = 6.75×10⁻⁴
- 10 ElasticBeamColumn elements per member

**Boundary conditions and loading:**
- Base of vertical member: fully fixed
- Free end of horizontal member: point load P downward
- P_max = 50.0, applied in 50 equal steps of ΔP = 1.0

### Reference Solution (Simo & Vu-Quoc 1986)

| P | u_x (horizontal tip) | u_y (vertical tip) |
|---|----------------------|---------------------|
| 10 | 4.91 | −0.98 |
| 20 | 11.00 | −3.51 |
| 30 | 15.67 | −7.68 |
| 40 | 17.37 | −12.84 |
| 50 | **23.48** | **−13.89** |

### Pass Criteria

- Tip displacements at P = 50 within 2% of Simo & Vu-Quoc reference.
- Monotonic Newton-Raphson convergence (< 10 iterations per step).
- Frame deformation qualitatively shows large rotation of the horizontal member.

### Numerical Tips

- This is a 2D problem. Use `ElasticBeamColumn2d` with `CorotCrdTransf2d`.
- 10 elements per member is sufficient.
- If Newton-Raphson fails before P = 20, `CrdTransf.update()` is not being called with current nodal positions.
- The horizontal member undergoes approximately 90° of rigid-body rotation by P = 50. Small-deformation or PDelta formulations will grossly underpredict tip displacement.
- Verification check: at large deformation, extract the local basic forces in the element frame — they should remain moderate in magnitude. Artificial growth in basic forces indicates incorrect separation of rigid-body motion.

> **Warning:** Reference values above are from Simo & Vu-Quoc (1986, CMAME 58). Some other papers use slightly different cross-sections or load levels. Always identify which paper your reference values come from before comparing.

---

## 8. Benchmark 7 — Williams Toggle Frame (Corotational, Snap-Through)

### Purpose

The Williams toggle frame (Williams 1964) is a two-bar frame that exhibits snap-through under a central point load. It combines Corotational kinematics with path-following at a bifurcation point. Reference in Crisfield (1991, Vol. 1) and Battini (2002, PhD Thesis, KTH).

### Model Definition

**Geometry:**
- Two inclined bars forming a shallow V-shape, symmetric about the vertical axis
- Half-span = 2540 mm, rise H = 254 mm (≈ 5.7° from horizontal — very shallow)
- Bar cross-section area A = 6452 mm²

**Material:** E = 68950 N/mm² (aluminum-like), linear elastic.

**Boundary conditions and loading:**
- Both base nodes pinned (u_x = u_y = 0)
- Apex: vertical point load P downward
- Arc-length control required

### Pass Criteria

- Snap-through load P_cr within 3% of reference value ≈ 26.7 kN.
- Post-snap-through equilibrium path recovered.
- Load-displacement curve matches Crisfield (1991) Fig. 9.16.

### Numerical Tips

- This is a pure truss problem. Until the truss kinematics refactor is complete, model with `ElasticBeamColumn` + `CorotCrdTransf` and a very small moment of inertia (I ≈ A²/1000) to approximate pin-jointed behavior.
- Use exact node coordinates — even 0.1% error in rise will shift P_cr noticeably.
- Arc-length step: ≈ 1% of P_cr.
- An anti-symmetric bifurcation mode exists at a slightly lower load in some configurations — monitor the eigenvalue of K_T for early warning.

> **Warning:** This problem has a very shallow rise (5.7°). Linear analysis overpredicts stiffness by approximately 3×. If your nonlinear result is close to the linear result, geometric stiffness is not being included.

---

## 9. Benchmark 8 — Thick-Walled Cylinder Under Internal Pressure (TL, UL)

### Purpose

The thick-walled cylinder has a closed-form solution (Lamé) and validates TL and UL in the presence of significant volumetric deformation, large hoop strains, and a non-trivial deformation gradient that is not a pure rotation.

### Model Definition

**Geometry:**
- Inner radius a = 1.0, outer radius b = 2.0
- **2D plane strain** — use `ElasticIsotropic(type='PlaneStrain')`
- Mesh: 8 Quad4 elements radially, quarter model with symmetry BCs

**Material:** E = 210000 MPa, ν = 0.3, linear elastic.

**Boundary conditions and loading:**
- Inner surface (r = a): uniform internal pressure P_in as radial traction
- Outer surface (r = b): traction-free
- Quarter model symmetry: u_x = 0 on left edge, u_y = 0 on bottom edge

### Analytical Reference (Lamé solution — plane strain)

The **plane strain** radial displacement is:

```
u_r(r) = P_in * a² / (E * (b² - a²)) * ((1 - 2ν) * r + b² / r)
```

> **Formula consistency warning:** The plane stress Lamé formula has `(1+ν)·b²/r` in place of `b²/r`.
> Using the wrong formula produces a wrong reference value and a false failure against a correct
> plane-strain implementation. This benchmark is **plane strain throughout** — model definition
> and formula are consistent. Do not mix them.

For E = 210000, ν = 0.3, a = 1, b = 2, P_in = 0.01·E = 2100:
```
u_r(a) = 2100·1 / (210000·3) · (0.4·1 + 4/1) ≈ 1.429×10⁻² m   (inner surface sanity check)
```

Use this formula as reference for small pressure (P_in < 0.01·E). For larger pressure, TL and UL must agree with each other to at least 4 significant figures — this is the primary pass criterion in the large-deformation regime.

### Pass Criteria

- Radial displacement u_r(r) at P_in = 0.01·E within 1% of **plane strain** Lamé solution at all radial positions.
- TL and UL agree to 4 significant figures at all pressure levels.
- Hoop stress distribution σ_θ(r) within 2% of plane strain Lamé solution for small pressure.

### Numerical Tips

- For TL: apply pressure as traction on the reference (undeformed) area. For UL: apply on the current deformed area. This distinction matters at large deformation.
- Print F = [[F_rr, 0], [0, F_θθ]] at mid-radius Gauss points. F_rr = dr/dR, F_θθ = r/R — both should be positive and near 1 for small pressure.
- Near the inner radius, elements may have large Jacobian ratios. Verify det(J) > 0 at all Gauss points after meshing.

> **Warning:** If TL gives larger displacements than UL, the UL reference configuration is not updating at `commitState()`. TL and UL must converge to the same solution. Any persistent discrepancy is a reference-configuration management bug.

---

## 10. General Numerical Modeling Tips and Common Pitfalls

### Load Stepping Strategy

- Start with 10–20 equal load steps regardless of problem. Increase only if convergence fails.
- Newton-Raphson should converge in 3–6 iterations for well-conditioned problems. More than 10 iterations per step usually means the tangent stiffness is wrong.
- If divergence occurs, halve the step and retry. Divergence at 1% of total load means the stiffness matrix assembly is incorrect — the problem is not the step size.
- For snap-through: arc-length control is required to trace the full load-displacement path including the post-snap branch. In this WP, load control is used — the benchmark is passed when the pre-snap limit load is within 5% of reference and the solver diverges gracefully. Post-snap path recovery is deferred to the analysis pipeline WP.

### Debugging Kinematic Formulations

- Always test kinematics in isolation first (simple shear unit test) before coupling to a solver.
- Print F, J = det(F), E (TL) or e (UL) at a single Gauss point for a prescribed deformation. If these are wrong, everything downstream is wrong.
- Consistency check at zero displacement: F = I, E = 0, e = 0, B = B_linear. This must hold exactly.
- For UL: after `commitState()`, the stored reference coordinates must equal the current deformed coordinates. Add an assertion during debugging.

### TL vs UL Agreement

- TL and UL describe the same physics. They must produce identical results to at least 4 significant figures on any well-behaved problem.
- Most likely causes of disagreement: (a) UL reference update not executing correctly, (b) incorrect Jacobian (TL uses J_0, UL uses J_current), (c) traction boundary condition applied to the wrong configuration.
- Resolve all TL/UL disagreements at the kinematic unit test level (Benchmark 3 / Simple Shear) before running coupled problems.

### Voigt Convention and Factor-of-2 Errors

The most common source of implementation errors is inconsistent treatment of shear strain in Voigt notation.

- Engineering convention (B matrix output): γ_12 = ∂u/∂y + ∂v/∂x = 2·ε_12
- Tensorial convention (CTensor internal): ε_12 = 0.5·(∂u/∂y + ∂v/∂x)
- The B matrix in oneFEM produces engineering strain. CTensor with COV representation applies the 1/2 factor internally. These must be consistent — never mix conventions within the same integration loop.
- Quick check: for pure shear loading (τ_xy only), the shear stiffness should be G = E / (2(1+ν)). If you get 2G or G/2, there is a factor-of-2 error in the Voigt chain.

### Geometric Stiffness K_σ

- K_σ is required for quadratic Newton-Raphson convergence in large-deformation problems. Without it, the tangent is only the material part and convergence degrades to linear or worse.
- TL: K_σ involves the 2nd Piola-Kirchhoff stress S and shape function gradients in the reference configuration.
- UL: K_σ involves the Cauchy stress σ and shape function gradients in the current configuration.
- Corotational: K_σ is handled implicitly by the corotational transformation of the element stiffness.
  **Note: this implicit handling applies to beam CrdTransf only.** When `CorotContinuumKinematics`
  is eventually implemented (future WP), K_σ must be made explicit — the EICR framework
  does not absorb it the same way as the beam corotational transformation.
- Quick test: compare Newton-Raphson convergence with and without K_σ on the cantilever rollup. With K_σ: quadratic convergence (iteration count independent of step size). Without K_σ: linear convergence or divergence at moderate loads.

---

## 11. Reference Table

| Benchmark | Primary Reference | Secondary Reference |
|-----------|------------------|---------------------|
| Patch Test | MacNeal & Harder (1985), Comput. Struct. 20(1-3) | Zienkiewicz & Taylor, FEM Vol. 1, Ch. 10 |
| Cook's Membrane | Cook (1974), J. Struct. Div. ASCE 100(9) | Simo & Rifai (1990), IJNME 29 |
| Simple Shear | Any nonlinear continuum mechanics textbook | Bonet & Wood (2008), Nonlinear Continuum Mechanics, Ch. 3 |
| Large-Deformation Cantilever | Bathe & Bolourchi (1979), Comput. Struct. 10 | Bonet & Wood (2008), Ch. 7 |
| Snap-Through Arch | Crisfield (1981), Comput. Struct. 13 | Crisfield, Nonlinear FEA Vol. 1 (1991), Ch. 9 |
| Lee's Frame | Simo & Vu-Quoc (1986), CMAME 58 | Lee (1971), ASCE J. Struct. Div. |
| Williams Toggle | Williams (1964) | Crisfield, Nonlinear FEA Vol. 1 (1991), Ch. 9 |
| Thick-Walled Cylinder | Bathe (1996), FEM Procedures, Ch. 6 | Holzapfel (2000), Nonlinear Solid Mechanics, Ch. 4 |

---

*oneFEM — Unified Kinematics Validation Schedule — v1.0 — March 2026*
