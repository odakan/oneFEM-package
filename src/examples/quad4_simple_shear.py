##-----------------------------------------------------------------------##
#  Benchmark: Quad4 Simple Shear — TL / UL validation
#
#  Single unit Quad4 element, PlaneStrain, E=1000, nu=0.3
#  Bottom edge fixed, top edge displaced horizontally by gamma.
#
#  Tests:
#    1. TL strain vs analytical Green-Lagrange for F=[[1,gamma],[0,1]]
#       E11 = 0, E22 = gamma^2/2, E12 = gamma/2
#    2. TL == UL (single load step, no history dependence)
#    3. Small gamma: TL/UL ≈ linear (within O(gamma^2))
#
#  Pass criterion: rel error < 1e-8 (for Green-Lagrange components)
##-----------------------------------------------------------------------##

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from oneFEM.model import Domain
from oneFEM.model.node import Node22
from oneFEM.model.element.continuum.quad4 import Quad4
from oneFEM.model.material.nD.elastic_isotropic import ElasticIsotropic
from oneFEM.model.element.kinematics.continuum.linear import LinearContinuumKinematics
from oneFEM.model.element.kinematics.continuum.total_lagrangian import TotalLagrangianContinuumKinematics
from oneFEM.model.element.kinematics.continuum.updated_lagrangian import UpdatedLagrangianContinuumKinematics
from oneFEM.model.pattern import Plain as PlainPattern
from oneFEM.model.tseries import Constant
from oneFEM._systools.data import Vector, Matrix
from oneFEM._systools.data.ctensor import CTensor

E_val = 1000.0
nu_val = 0.3


def build_single_quad(kinematics):
    """Build a single unit Quad4 with given kinematics. Bottom fixed, top free."""
    model = Domain(nD=2)
    nd1 = Node22(1, coord=[0.0, 0.0])
    nd2 = Node22(2, coord=[1.0, 0.0])
    nd3 = Node22(3, coord=[1.0, 1.0])
    nd4 = Node22(4, coord=[0.0, 1.0])

    for nd in [nd1, nd2, nd3, nd4]:
        model.add(nd)

    # Fix bottom edge
    nd1.setFix([True, True])
    nd2.setFix([True, True])

    mat = ElasticIsotropic(1, E_val, nu_val, type='PlaneStrain')
    elem = Quad4(1, [nd1, nd2, nd3, nd4], mat, kinematics=kinematics, thickness=1.0)
    model.add(elem)

    ts = Constant(1, factor=1.0)
    pat = PlainPattern(1, ts)
    model.add(pat)

    model._domain()
    return model, [nd1, nd2, nd3, nd4], elem


def apply_simple_shear(nodes, gamma):
    """Apply pure simple shear: top nodes displaced by gamma in x, bottom fixed.
    F = [[1, gamma], [0, 1]] everywhere (homogeneous).
    Bottom nodes: u=0, v=0. Top nodes: u=gamma*y, v=0 (y=1 for top).
    """
    nd1, nd2, nd3, nd4 = nodes
    # Bottom already fixed at zero
    nd1._update(Vector([0.0, 0.0]), Vector([0.0, 0.0]))
    nd2._update(Vector([0.0, 0.0]), Vector([0.0, 0.0]))
    # Top: u = gamma, v = 0 (y = 1)
    nd3._update(Vector([0.0, 0.0]), Vector([gamma, 0.0]))
    nd4._update(Vector([0.0, 0.0]), Vector([gamma, 0.0]))
    nd1._commitState()
    nd2._commitState()
    nd3._commitState()
    nd4._commitState()


def analytical_GL(gamma):
    """Exact Green-Lagrange strain for F = [[1, gamma], [0, 1]].
    E = 0.5*(F^T F - I)
    F^T F = [[1, gamma],[gamma, 1+gamma^2]]
    E = [[0, gamma/2],[gamma/2, gamma^2/2]]
    Voigt COV input: [E11, E22, 2*E12] = [0, gamma^2/2, gamma]
    """
    return np.array([0.0, gamma**2 / 2.0, gamma])


def test_tl_strain(gamma):
    """Test 1: TL strain vs analytical Green-Lagrange."""
    kin = TotalLagrangianContinuumKinematics()
    _, nodes, elem = build_single_quad(kin)
    apply_simple_shear(nodes, gamma)
    elem._update()

    # Check strain at all GPs (should be constant for this problem)
    E_exact = analytical_GL(gamma)
    max_err = 0.0
    for gp in range(len(elem._gp_data)):
        strain = elem._kinematics.getStrain(gp)
        E_voigt = strain.make_vector()
        for i in range(3):
            ref = abs(E_exact[i])
            if ref > 1e-15:
                err = abs(E_voigt[i] - E_exact[i]) / ref
            else:
                err = abs(E_voigt[i])
            max_err = max(max_err, err)

    passed = max_err < 1e-8
    return passed, max_err


def test_tl_equals_ul(gamma):
    """Test 2: TL and UL produce same strain in single step (no committed history).
    Both use initial config as reference -> same result."""
    kin_tl = TotalLagrangianContinuumKinematics()
    _, nodes_tl, elem_tl = build_single_quad(kin_tl)
    apply_simple_shear(nodes_tl, gamma)
    elem_tl._update()

    kin_ul = UpdatedLagrangianContinuumKinematics()
    _, nodes_ul, elem_ul = build_single_quad(kin_ul)
    apply_simple_shear(nodes_ul, gamma)
    elem_ul._update()

    max_err = 0.0
    for gp in range(len(elem_tl._gp_data)):
        E_tl = elem_tl._kinematics.getStrain(gp).make_vector()
        E_ul = elem_ul._kinematics.getStrain(gp).make_vector()
        for i in range(3):
            ref = max(abs(E_tl[i]), 1e-15)
            err = abs(E_tl[i] - E_ul[i]) / ref
            max_err = max(max_err, err)

    passed = max_err < 1e-10
    return passed, max_err


def test_small_strain_convergence():
    """Test 3: For small gamma, TL ~ linear within O(gamma^2).
    Compare TL and linear at gamma=0.001. Difference should be O(gamma^2) ~ 1e-6."""
    gamma = 0.001

    kin_lin = LinearContinuumKinematics()
    _, nodes_lin, elem_lin = build_single_quad(kin_lin)
    apply_simple_shear(nodes_lin, gamma)
    elem_lin._update()

    kin_tl = TotalLagrangianContinuumKinematics()
    _, nodes_tl, elem_tl = build_single_quad(kin_tl)
    apply_simple_shear(nodes_tl, gamma)
    elem_tl._update()

    # Compare shear strain component (dominant term)
    max_diff = 0.0
    for gp in range(len(elem_tl._gp_data)):
        eps_lin = elem_lin._kinematics.getStrain(gp).make_vector()
        E_tl = elem_tl._kinematics.getStrain(gp).make_vector()
        # shear: linear gives gamma (engineering), TL gives gamma (same to first order)
        # normal: linear E11=0, E22=0. TL: E11=0, E22=gamma^2/2 ~ 5e-7
        for i in range(3):
            diff = abs(eps_lin[i] - E_tl[i])
            max_diff = max(max_diff, diff)

    # The max difference should be O(gamma^2) = 1e-6
    passed = max_diff < 10.0 * gamma**2
    return passed, max_diff


def test_tl_kgeo_nonzero(gamma):
    """Test 4: TL geometric stiffness is non-zero for non-trivial stress."""
    kin = TotalLagrangianContinuumKinematics()
    _, nodes, elem = build_single_quad(kin)
    apply_simple_shear(nodes, gamma)
    elem._update()

    kgeo_norm = 0.0
    for gp in range(len(elem._gp_data)):
        stress = elem._materials[gp].getStress()
        K_geo = elem._kinematics.getGeometricStiffness(gp, stress)
        kgeo_norm += np.linalg.norm(K_geo.data)

    passed = kgeo_norm > 1e-10
    return passed, kgeo_norm


def test_tl_f_identity_at_zero():
    """Test 5: At zero displacement, F = I, E = 0, B_NL = B_linear."""
    kin_tl = TotalLagrangianContinuumKinematics()
    _, nodes, elem = build_single_quad(kin_tl)
    # Apply zero displacement
    apply_simple_shear(nodes, 0.0)
    elem._update()

    max_err_F = 0.0
    max_err_E = 0.0
    for gp in range(len(elem._gp_data)):
        F = elem._kinematics.getF(gp)
        I = np.eye(2)
        max_err_F = max(max_err_F, np.max(np.abs(F.data - I)))

        E = elem._kinematics.getStrain(gp).make_vector()
        max_err_E = max(max_err_E, np.max(np.abs(E)))

    passed = max_err_F < 1e-15 and max_err_E < 1e-15
    return passed, max(max_err_F, max_err_E)


def test_deformation_gradient(gamma):
    """Test 6: Verify F matches analytical for simple shear."""
    kin = TotalLagrangianContinuumKinematics()
    _, nodes, elem = build_single_quad(kin)
    apply_simple_shear(nodes, gamma)
    elem._update()

    F_exact = np.array([[1.0, gamma], [0.0, 1.0]])
    max_err = 0.0
    for gp in range(len(elem._gp_data)):
        F = elem._kinematics.getF(gp)
        err = np.max(np.abs(F.data - F_exact))
        max_err = max(max_err, err)

    passed = max_err < 1e-10
    return passed, max_err


if __name__ == "__main__":
    print("=" * 60)
    print("Quad4 Simple Shear — TL / UL Validation")
    print("  E = {}, nu = {}, PlaneStrain, t = 1.0".format(E_val, nu_val))
    print("  Single unit Quad4, bottom fixed, top sheared")
    print("=" * 60)

    results = []

    # Test 5: Zero displacement sanity check
    p, err = test_tl_f_identity_at_zero()
    print("\n  Test 1 (TL: F=I, E=0 at zero disp):     {}  (err={:.2e})".format(
        "PASS" if p else "FAIL", err))
    results.append(p)

    # Test 6: F matches analytical
    for gamma in [0.1, 0.3, 0.5]:
        p, err = test_deformation_gradient(gamma)
        print("  Test 2 (TL: F analytical, gamma={:.1f}):   {}  (err={:.2e})".format(
            gamma, "PASS" if p else "FAIL", err))
        results.append(p)

    # Test 1: TL strain vs analytical
    for gamma in [0.1, 0.3, 0.5]:
        p, err = test_tl_strain(gamma)
        print("  Test 3 (TL: E vs analytical, gamma={:.1f}): {}  (err={:.2e})".format(
            gamma, "PASS" if p else "FAIL", err))
        results.append(p)

    # Test 2: TL == UL
    for gamma in [0.1, 0.3, 0.5]:
        p, err = test_tl_equals_ul(gamma)
        print("  Test 4 (TL == UL, gamma={:.1f}):           {}  (err={:.2e})".format(
            gamma, "PASS" if p else "FAIL", err))
        results.append(p)

    # Test 3: Small strain convergence
    p, diff = test_small_strain_convergence()
    print("  Test 5 (small gamma: TL ~ linear):       {}  (diff={:.2e})".format(
        "PASS" if p else "FAIL", diff))
    results.append(p)

    # Test 4: K_geo non-zero
    p, norm_val = test_tl_kgeo_nonzero(0.3)
    print("  Test 6 (TL K_geo non-zero, gamma=0.3):   {}  (||K_geo||={:.2e})".format(
        "PASS" if p else "FAIL", norm_val))
    results.append(p)

    n_pass = sum(results)
    n_total = len(results)
    print("\n" + "=" * 60)
    print("  {}/{} PASS".format(n_pass, n_total))
    if n_pass == n_total:
        print("  ALL PASS")
    print("=" * 60)
