# Known Issues

## UL Kinematics: Total vs Incremental Displacement

**Status**: Open — tracked as future WP

**Summary**: `ContinuumElement._update()` passes the total displacement from
the original coordinates to all kinematics formulations. For Updated Lagrangian
(UL), this is incorrect: UL updates `dN_dX` to the last committed configuration
via `commitState()`, so it expects the incremental displacement from that
committed state, not the total displacement from origin.

**Symptoms**:
- TL and UL produce identical results on rectangular meshes (cantilever, simple
  shear on unit square) because `dN_dX_updated` is numerically close to
  `dN_dX_original` for rectangular elements.
- TL and UL diverge on non-rectangular meshes (circular arch, annular cylinder)
  where the updated shape function derivatives differ significantly from the
  original. The error accumulates over load steps.
- Verified: TL==UL to machine precision on 20x1 straight cantilever (B4), but
  TL!=UL by 50-400% on circular arches at all rise heights including H=0.05
  (nearly flat but trapezoidal elements).

**Root cause**: `_update()` in `continuum/base.py` extracts `nd._getTrialDisp()`
which is total displacement from origin. UL kinematics computes
`H = u_total^T @ dN_dX_updated`, which is physically meaningless — it is
neither the total gradient (TL uses `dN_dX_original`) nor the incremental
gradient (would need `u_incremental = u_trial - u_committed`).

**Fix requirements**:
1. Pass incremental displacement `u_trial - u_committed` to UL kinematics.
2. Since UL would then receive only incremental strain, the material must
   accumulate total stress across steps (`sigma_total = sigma_committed + C :
   d_epsilon`). Currently `ElasticIsotropic` computes `sigma = C : epsilon`
   with no history, so it would return only the incremental stress.
3. This requires a material interface change (stress accumulation), which is a
   Stop-and-Ask item per CLAUDE.md.

**Affected benchmarks**: B5 snap-through arch runs TL only. B8 thick-walled
cylinder uses single-step linear solve where UL==TL (no accumulated error).
B3 simple shear and B4 cantilever use rectangular elements where the bug is
invisible.
