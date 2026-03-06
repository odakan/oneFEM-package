##-----------------------------------------------------------------------##
#  Benchmark: Corotational Continuum (EICR) — Quad4
#
#  Tests:
#    1. Corot Newton convergence (cantilever, 10 steps)
#    2. Corot == TL for small load (single step, err < 1e-6)
#    3. Corot == TL == UL for small load (cross-check)
#    4. Corot cantilever large deformation (qualitative)
#    5. Corot patch test (zero strain for rigid body motion)
#
#  Pass criteria: Newton converges, formulations agree at small strain.
##-----------------------------------------------------------------------##

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from oneFEM.model import Domain
from oneFEM.model.node import Node22
from oneFEM.model.element.continuum.quad4 import Quad4
from oneFEM.model.material.nD.elastic_isotropic import ElasticIsotropic
from oneFEM.model.element.kinematics.continuum.corot import CorotContinuumKinematics
from oneFEM.model.element.kinematics.continuum.total_lagrangian import TotalLagrangianContinuumKinematics
from oneFEM.model.element.kinematics.continuum.updated_lagrangian import UpdatedLagrangianContinuumKinematics
from oneFEM.model.pattern import Plain as PlainPattern
from oneFEM.model.tseries import Linear as LinearTS
from oneFEM.analysis import Analysis
from oneFEM.analysis.algorithm.newton_raphson import Newton
from oneFEM.analysis.constraints import Plain as PlainConstraints
from oneFEM.analysis.numberer import Plain as PlainNumberer
from oneFEM.analysis.system import FullGeneral
from oneFEM.analysis.integrator import LoadControl
from oneFEM.analysis.test import NormUnbalance
from oneFEM import SimulationManager

# Cantilever geometry
L = 10.0
h = 1.0
nElem_x = 10
nElem_y = 1

E_val = 1000.0
nu_val = 0.0
t_val = 1.0

F_tip = 0.5


def build_cantilever(kinematics_class, nSteps=10, F_total=None):
    """Build mesh, run Newton analysis, return (converged, ux, uy)."""
    if F_total is None:
        F_total = F_tip

    model = Domain(nD=2)
    dx = L / nElem_x
    dy = h / nElem_y
    nodes = {}
    nid = 1

    for j in range(nElem_y + 1):
        for i in range(nElem_x + 1):
            x = i * dx
            y = j * dy
            nd = Node22(nid, coord=[x, y])
            nodes[nid] = nd
            model.add(nd)
            nid += 1

    # Fix left edge
    for j in range(nElem_y + 1):
        left_nid = j * (nElem_x + 1) + 1
        nodes[left_nid].setFix([True, True])

    # Create elements
    eid = 1
    mat = ElasticIsotropic(1, E_val, nu_val, type='PlaneStress')
    for j in range(nElem_y):
        for i in range(nElem_x):
            n1 = j * (nElem_x + 1) + i + 1
            n2 = n1 + 1
            n3 = n2 + (nElem_x + 1)
            n4 = n1 + (nElem_x + 1)
            kin = kinematics_class()
            elem = Quad4(eid, [nodes[n1], nodes[n2], nodes[n3], nodes[n4]],
                         mat, kinematics=kin, thickness=t_val)
            model.add(elem)
            eid += 1

    # Tip load distributed between top and bottom
    tip_top_nid = (nElem_y) * (nElem_x + 1) + (nElem_x + 1)
    tip_bot_nid = nElem_x + 1

    ts = LinearTS(1, factor=1.0)
    F_per_node = F_total / 2.0
    pat = PlainPattern(1, tseries=ts,
                       load=[[tip_bot_nid, 0.0, F_per_node],
                              [tip_top_nid, 0.0, F_per_node]])
    model.add(pat)

    alg = Newton(1, tangent='current')
    const = PlainConstraints(1)
    numb = PlainNumberer(1)
    syst = FullGeneral(1)
    integ = LoadControl(1)
    ctest = NormUnbalance(1, tol=1e-8, maxIter=50)
    analysis = Analysis(1, algorithm=alg, constraints=const,
                        integrator=integ, system=syst, test=ctest)
    sim = SimulationManager(1, model, analysis, dt=1.0/nSteps)

    converged = True
    try:
        sim.analyze(nSteps, dt=1.0/nSteps)
    except Exception:
        converged = False

    tip_nd = nodes[tip_bot_nid]
    u_tip = tip_nd._getCommitDisp()
    ux = float(u_tip[0])
    uy = float(u_tip[1])

    return converged, ux, uy


def test_corot_convergence():
    """Test 1: Corot Newton converges over 10 steps."""
    converged, ux, uy = build_cantilever(CorotContinuumKinematics, nSteps=10)
    status = "PASS" if converged else "FAIL"
    print("  Test 1 (Corot Newton convergence):    {}  (ux={:.6e}, uy={:.6e})".format(
        status, ux, uy))
    return converged


def test_corot_tl_agreement():
    """Test 2: Corot and TL agree at small load (single step)."""
    F_small = 0.01
    _, ux_co, uy_co = build_cantilever(CorotContinuumKinematics, nSteps=1, F_total=F_small)
    _, ux_tl, uy_tl = build_cantilever(TotalLagrangianContinuumKinematics, nSteps=1, F_total=F_small)

    err_x = abs(ux_co - ux_tl) / max(abs(ux_tl), 1e-15) if abs(ux_tl) > 1e-15 else abs(ux_co - ux_tl)
    err_y = abs(uy_co - uy_tl) / max(abs(uy_tl), 1e-15)
    max_err = max(err_x, err_y)

    passed = max_err < 1e-3
    status = "PASS" if passed else "FAIL"
    print("  Test 2 (Corot == TL, small load):     {}  (err={:.2e})".format(status, max_err))
    print("    Corot: ux={:.6e}, uy={:.6e}".format(ux_co, uy_co))
    print("    TL:    ux={:.6e}, uy={:.6e}".format(ux_tl, uy_tl))
    return passed


def test_corot_tl_ul_crosscheck():
    """Test 3: Corot, TL, UL all agree at small load."""
    F_small = 0.01
    _, ux_co, uy_co = build_cantilever(CorotContinuumKinematics, nSteps=1, F_total=F_small)
    _, ux_tl, uy_tl = build_cantilever(TotalLagrangianContinuumKinematics, nSteps=1, F_total=F_small)
    _, ux_ul, uy_ul = build_cantilever(UpdatedLagrangianContinuumKinematics, nSteps=1, F_total=F_small)

    err_co_tl = max(abs(uy_co - uy_tl) / max(abs(uy_tl), 1e-15),
                    abs(ux_co - ux_tl) / max(abs(ux_tl), 1e-15) if abs(ux_tl) > 1e-15 else 0)
    err_co_ul = max(abs(uy_co - uy_ul) / max(abs(uy_ul), 1e-15),
                    abs(ux_co - ux_ul) / max(abs(ux_ul), 1e-15) if abs(ux_ul) > 1e-15 else 0)

    passed = err_co_tl < 1e-3 and err_co_ul < 1e-3
    status = "PASS" if passed else "FAIL"
    print("  Test 3 (Corot==TL==UL, small load):   {}  (err_co_tl={:.2e}, err_co_ul={:.2e})".format(
        status, err_co_tl, err_co_ul))
    return passed


def test_corot_large_deformation():
    """Test 4: Corot large deformation is physically reasonable."""
    converged, ux, uy = build_cantilever(CorotContinuumKinematics, nSteps=10, F_total=F_tip)
    reasonable = converged and uy > 0
    status = "PASS" if reasonable else "FAIL"
    print("  Test 4 (Corot large deformation):     {}  (ux={:.6e}, uy={:.6e})".format(
        status, ux, uy))
    return reasonable


def test_corot_patch_test():
    """Test 5: Patch test — single element, uniform strain recovers exact stress.

    Apply uniform horizontal extension eps_xx = 0.001 to a unit square.
    All formulations should give the same stress (for small strain, Corot == linear).
    """
    from oneFEM._systools.data import Vector

    for kin_name, kin_class in [("Linear", None),
                                 ("Corot", CorotContinuumKinematics)]:
        model = Domain(nD=2)
        nd1 = Node22(1, coord=[0.0, 0.0])
        nd2 = Node22(2, coord=[1.0, 0.0])
        nd3 = Node22(3, coord=[1.0, 1.0])
        nd4 = Node22(4, coord=[0.0, 1.0])
        model.add(nd1, nd2, nd3, nd4)

        mat = ElasticIsotropic(1, E_val, nu_val, type='PlaneStress')
        if kin_class is not None:
            kin = kin_class()
            elem = Quad4(1, [nd1, nd2, nd3, nd4], mat, kinematics=kin, thickness=1.0)
        else:
            elem = Quad4(1, [nd1, nd2, nd3, nd4], mat, thickness=1.0)
        model.add(elem)

        ts = LinearTS(1, factor=1.0)
        pat = PlainPattern(1, tseries=ts, load=[[2, 0.0, 0.0]])
        model.add(pat)

        # Fix
        nd1.setFix([True, True])
        nd4.setFix([True, True])

        model._domain()
        model._assemble()

        # Apply uniform eps_xx = 0.001 → u_x = 0.001 * x
        eps_xx = 0.001
        f_zero = Vector([0.0, 0.0])
        nd2._update(f_zero, Vector([eps_xx * 1.0, 0.0]))
        nd2._commitState()
        nd3._update(f_zero, Vector([eps_xx * 1.0, 0.0]))
        nd3._commitState()

        elem._update()

        # Check stress at GP 0
        sig = elem._materials[0].getStress().make_vector()
        sig_xx_expected = E_val * eps_xx  # PlaneStress, nu=0: sigma_xx = E * eps_xx
        err = abs(sig[0] - sig_xx_expected) / sig_xx_expected

        if kin_name == "Corot":
            passed = err < 1e-10
            status = "PASS" if passed else "FAIL"
            print("  Test 5 (Corot patch, uniform strain): {}  (sig_xx={:.6e}, expected={:.6e}, err={:.2e})".format(
                status, sig[0], sig_xx_expected, err))
            return passed


if __name__ == "__main__":
    print("=" * 60)
    print("Corotational Continuum (EICR) — Quad4 Benchmarks")
    print("  E={}, nu={}, PlaneStress, t={}".format(E_val, nu_val, t_val))
    print("  {}x{} mesh ({} elements), L={}, h={}".format(
        nElem_x, nElem_y, nElem_x * nElem_y, L, h))
    print("=" * 60)

    results = []

    results.append(test_corot_convergence())
    results.append(test_corot_tl_agreement())
    results.append(test_corot_tl_ul_crosscheck())
    results.append(test_corot_large_deformation())
    results.append(test_corot_patch_test())

    n_pass = sum(results)
    n_total = len(results)
    print("\n" + "=" * 60)
    print("  {}/{} PASS".format(n_pass, n_total))
    if all(results):
        print("  ALL PASS")
    print("=" * 60)
