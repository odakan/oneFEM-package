# Known Issues

## UL Kinematics: Total vs Incremental Displacement

**Status**: Fixed (2026-03-06)

**Summary**: Two bugs in Updated Lagrangian (UL) kinematics.

**Bug 1 — Wrong displacement**: `ContinuumElement._update()` was passing the total
displacement from the original coordinates to all kinematics formulations. For UL,
this was incorrect: UL updates `dN_dX` to the last committed configuration via
`commitState()`, so it expects the incremental displacement from that committed
state, not the total displacement from origin.

**Bug 2 — Strain accumulation error**: After fixing Bug 1, UL computed incremental
Green-Lagrange strain from the updated reference and added it to the committed strain
via `_setTrialStrainIncr()`. This is mathematically wrong: GL strains from different
reference configurations are NOT additive. The correct relationship is
E_total = E_1 + F_1^T E_2 F_1 (not E_1 + E_2). This caused TL and UL to diverge
by 5-20% on multi-row meshes at large deformation.

**Fix**: Two-layer approach:
1. `ContinuumElement` tracks committed displacement (`_committed_u_e`), computes
   incremental displacement for UL kinematics. Backup/restore in commit/revert.
2. UL kinematics tracks `F_commit` (total deformation gradient from original config)
   at each Gauss point. On each `update()`:
   - Computes `F_incr` from incremental displacement and updated `dN_dX`
   - Recovers `F_total = F_incr @ F_commit`
   - Computes strain, B matrix, and geometric stiffness from `F_total` and
     `dN_dX_original` (same as TL)
   - Uses `material._setTrialStrain(E_total)` (not `_setTrialStrainIncr`)
   - Returns `None` from `getDetJ()` so element uses original `dV`
3. On `commitState()`, updates `F_commit = F_total` and recomputes `dN_dX` for the
   next step's `F_incr` computation.
4. Fixed shallow-copy bugs in TL and Linear `copy()` methods.

This guarantees TL == UL to machine precision on any mesh at any deformation level.
The UL advantage (better Newton conditioning from smaller incremental displacements)
is preserved.

**Verification**:
- B3 (simple shear): TL == UL exact for gamma = 0.1, 0.5, 1.0, 2.0
- B4 (cantilever 20x1): TL == UL = 2.66e-15 at large deformation
- B8 (thick-walled cylinder 8x4 annular): TL == UL = 8.16e-16 with 10 load steps
- Validation (cantilever 20x2): TL == UL = 4.44e-15 at 77.6 deg tip rotation

**Previous symptoms** (before fix):
- Bug 1: TL and UL produced identical results on rectangular meshes but diverged by
  50-400% on non-rectangular meshes (arches, annular cylinders). Root cause was
  `H = u_total @ dN_dX_updated` — physically meaningless.
- Bug 2: After Bug 1 fix, TL-UL difference was ~5% on 20x2 cantilever and ~5% on
  thick-walled cylinder. Difference did NOT decrease with more steps (not a step-size
  effect). Root cause was additive GL strain accumulation across references.
