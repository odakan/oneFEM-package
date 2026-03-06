"""CorotContinuumKinematics unit tests.

Tests:
  1. Pure rotation at 10, 30, 90, 180 degrees -> zero strain, correct R
  2. Commit/revert preserves and restores rotation state
  3. transformToGlobal rotates K block-by-block

All tests must PASS before running element-level benchmarks.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import numpy as np
from oneFEM.model.element.kinematics.continuum.corot import CorotContinuumKinematics
from oneFEM.model.element.kinematics.continuum.linear import LinearContinuumKinematics
from oneFEM.model.element.continuum.isoparametric import (
    quad4_shape_derivatives, quad4_gauss_points, compute_physical_derivatives)
from oneFEM._systools.data import Vector, Matrix


def setup_quad4_kin():
    """Set up a CorotContinuumKinematics for a unit-square Quad4.

    Node ordering (CCW):
        3 --- 2
        |     |
        0 --- 1

    Returns: (kin, ref_coords, nGP, dN_dX_list)
    """
    ref_coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ])

    gps = quad4_gauss_points()
    nGP = len(gps)
    dN_dX_list = []
    for xi, eta, w in gps:
        dN_dxi = quad4_shape_derivatives(xi, eta)
        dN_dX, detJ = compute_physical_derivatives(dN_dxi, ref_coords)
        dN_dX_list.append(Matrix(init=dN_dX))

    kin = CorotContinuumKinematics()
    kin.initialize(nGP, 2, 4, dN_dX_list, X_ref=ref_coords)
    return kin, ref_coords, nGP, dN_dX_list


def make_rigid_rotation_disp(ref_coords, angle_deg):
    """Compute nodal displacements for a rigid rotation about the centroid."""
    theta = np.radians(angle_deg)
    R_test = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta),  np.cos(theta)]])
    centroid = ref_coords.mean(axis=0)
    u = np.zeros(8)
    for i, (X, Y) in enumerate(ref_coords):
        pos = np.array([X, Y]) - centroid
        new_pos = R_test @ pos + centroid
        u[2*i]   = new_pos[0] - X
        u[2*i+1] = new_pos[1] - Y
    return Vector(u), R_test


# ----------------------------------------------------------------
# Test 1: Pure rotation -> zero strain
# ----------------------------------------------------------------
def test_pure_rotation():
    results = []
    for angle in [10, 30, 90, 180]:
        kin, ref_coords, nGP, _ = setup_quad4_kin()
        u_rot, R_test = make_rigid_rotation_disp(ref_coords, angle)

        for gp in range(nGP):
            kin.update(gp, u_rot)
        kin.applyCorotFrame(u_rot)

        # Check strain = 0
        max_strain = 0.0
        for gp in range(nGP):
            eps = kin.getStrain(gp).make_vector()
            max_strain = max(max_strain, np.max(np.abs(eps)))

        # Check R = R_test
        R_err = np.max(np.abs(kin._R - R_test))

        ok = max_strain < 1e-10 and R_err < 1e-10
        status = "PASS" if ok else "FAIL"
        print(f"  Pure rotation {angle:3d} deg: strain={max_strain:.2e}, "
              f"R_err={R_err:.2e}  {status}")
        results.append(ok)
    return all(results)


# ----------------------------------------------------------------
# Test 2: Commit / revert
# ----------------------------------------------------------------
def test_revert():
    kin, ref_coords, nGP, _ = setup_quad4_kin()
    u_zero = Vector(np.zeros(8))

    # Step 1: commit at zero
    for gp in range(nGP):
        kin.update(gp, u_zero)
    kin.applyCorotFrame(u_zero)
    kin.commitState()
    R_committed = kin._R_committed.copy()

    # Step 2: apply 30-degree rotation (trial, not committed)
    u_rot, _ = make_rigid_rotation_disp(ref_coords, 30.0)
    for gp in range(nGP):
        kin.update(gp, u_rot)
    kin.applyCorotFrame(u_rot)

    changed = not np.allclose(kin._R, R_committed)

    # Step 3: revert
    kin.revertToLastCommit()
    restored   = np.allclose(kin._R, R_committed, atol=1e-14)
    no_alias   = kin._R is not kin._R_committed
    commit_eye = np.allclose(kin._R_committed, np.eye(2), atol=1e-14)

    ok = changed and restored and no_alias and commit_eye
    status = "PASS" if ok else "FAIL"
    print(f"  Revert test: changed={changed}, restored={restored}, "
          f"no_alias={no_alias}, commit_eye={commit_eye}  {status}")
    return ok


# ----------------------------------------------------------------
# Test 3: transformToGlobal rotates K blocks correctly
# ----------------------------------------------------------------
def test_transform_to_global():
    kin, ref_coords, nGP, _ = setup_quad4_kin()
    u_45, _ = make_rigid_rotation_disp(ref_coords, 45.0)

    for gp in range(nGP):
        kin.update(gp, u_45)
    kin.applyCorotFrame(u_45)

    # Use a non-trivial K_local (random symmetric)
    rng = np.random.default_rng(42)
    K_raw = rng.standard_normal((8, 8))
    K_raw = K_raw + K_raw.T
    K_local = Matrix(init=K_raw)
    f_local = Vector(rng.standard_normal(8))

    K_global, f_global = kin.transformToGlobal(K_local, f_local)

    R = kin._R
    ok = True
    for I in range(4):
        # Check f block
        sl = slice(2*I, 2*I+2)
        f_expected = R @ f_local.data[sl]
        if not np.allclose(f_global.data[sl], f_expected, atol=1e-12):
            ok = False

        # Check K blocks
        for J in range(4):
            sj = slice(2*J, 2*J+2)
            K_expected = R @ K_raw[sl, sj] @ R.T
            if not np.allclose(K_global.data[sl, sj], K_expected, atol=1e-12):
                ok = False

    status = "PASS" if ok else "FAIL"
    print(f"  Transform-to-global test:  {status}")
    return ok


# ----------------------------------------------------------------
# Test 4: Pure translation -> zero strain
# ----------------------------------------------------------------
def test_pure_translation():
    kin, ref_coords, nGP, _ = setup_quad4_kin()
    # Translate all nodes by (5.0, -3.0)
    u = np.zeros(8)
    for i in range(4):
        u[2*i]   = 5.0
        u[2*i+1] = -3.0
    u_vec = Vector(u)

    for gp in range(nGP):
        kin.update(gp, u_vec)
    kin.applyCorotFrame(u_vec)

    max_strain = 0.0
    for gp in range(nGP):
        eps = kin.getStrain(gp).make_vector()
        max_strain = max(max_strain, np.max(np.abs(eps)))

    ok = max_strain < 1e-14
    status = "PASS" if ok else "FAIL"
    print(f"  Pure translation: strain={max_strain:.2e}  {status}")
    return ok


# ----------------------------------------------------------------
# Test 5: Translation + rotation -> zero strain
# ----------------------------------------------------------------
def test_translation_plus_rotation():
    kin, ref_coords, nGP, _ = setup_quad4_kin()
    theta = np.radians(60.0)
    R_test = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta),  np.cos(theta)]])
    centroid = ref_coords.mean(axis=0)
    translation = np.array([10.0, -7.5])

    u = np.zeros(8)
    for i, (X, Y) in enumerate(ref_coords):
        pos = np.array([X, Y]) - centroid
        new_pos = R_test @ pos + centroid + translation
        u[2*i]   = new_pos[0] - X
        u[2*i+1] = new_pos[1] - Y
    u_vec = Vector(u)

    for gp in range(nGP):
        kin.update(gp, u_vec)
    kin.applyCorotFrame(u_vec)

    max_strain = 0.0
    for gp in range(nGP):
        eps = kin.getStrain(gp).make_vector()
        max_strain = max(max_strain, np.max(np.abs(eps)))

    R_err = np.max(np.abs(kin._R - R_test))
    ok = max_strain < 1e-10 and R_err < 1e-10
    status = "PASS" if ok else "FAIL"
    print(f"  Translation+rotation (60 deg): strain={max_strain:.2e}, "
          f"R_err={R_err:.2e}  {status}")
    return ok


if __name__ == "__main__":
    print("=" * 60)
    print("CorotContinuumKinematics Unit Tests")
    print("=" * 60)

    results = []
    print("\nTest 1 — Pure rotation (zero strain):")
    results.append(test_pure_rotation())

    print("\nTest 2 — Commit/revert:")
    results.append(test_revert())

    print("\nTest 3 — transformToGlobal:")
    results.append(test_transform_to_global())

    print("\nTest 4 — Pure translation (zero strain):")
    results.append(test_pure_translation())

    print("\nTest 5 — Translation + rotation (zero strain):")
    results.append(test_translation_plus_rotation())

    n_pass = sum(results)
    n_total = len(results)
    print(f"\n{'=' * 60}")
    print(f"  {n_pass}/{n_total} PASS")
    if all(results):
        print("  ALL PASS")
    print(f"{'=' * 60}")
