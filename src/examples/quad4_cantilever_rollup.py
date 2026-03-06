##-----------------------------------------------------------------------##
#  Benchmark: Quad4 Cantilever — Large Deformation (TL / UL)
#
#  Slender cantilever beam modeled with Quad4 elements.
#  Tip moment applied via force couple (or displacement control).
#  Validates Newton convergence and TL/UL agreement.
#
#  Tests:
#    1. Newton convergence with TL (10 steps)
#    2. Newton convergence with UL (10 steps)
#    3. TL and UL produce same tip displacement (single step, small load)
#    4. Large deformation: tip curves reasonably (qualitative)
#
#  Pass criterion: Newton converges within maxIter per step.
#                  TL == UL within 1e-6 for single step.
##-----------------------------------------------------------------------##

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from oneFEM.model import Domain
from oneFEM.model.node import Node22
from oneFEM.model.element.continuum.quad4 import Quad4
from oneFEM.model.material.nD.elastic_isotropic import ElasticIsotropic
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
from oneFEM._systools.data import Vector


# Cantilever geometry
L = 10.0     # length
h = 1.0      # height
nElem_x = 10 # elements along length
nElem_y = 1  # elements through height

E_val = 1000.0
nu_val = 0.0   # zero Poisson for beam-like behavior
t_val = 1.0    # thickness

# Tip force
F_tip = 0.5    # small enough for convergence with 10 steps


def build_cantilever(kinematics_class, nSteps=10, F_total=None):
    """Build mesh, run Newton analysis, return tip displacements."""
    if F_total is None:
        F_total = F_tip

    model = Domain(nD=2)
    dx = L / nElem_x
    dy = h / nElem_y
    nodes = {}
    nid = 1

    # Create nodes
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
    elements = {}
    for j in range(nElem_y):
        for i in range(nElem_x):
            n1 = j * (nElem_x + 1) + i + 1
            n2 = n1 + 1
            n3 = n2 + (nElem_x + 1)
            n4 = n1 + (nElem_x + 1)
            kin = kinematics_class()
            elem = Quad4(eid, [nodes[n1], nodes[n2], nodes[n3], nodes[n4]],
                         mat, kinematics=kin, thickness=t_val)
            elements[eid] = elem
            model.add(elem)
            eid += 1

    # Tip load as force couple for bending moment, or simple transverse load
    # Use simple transverse load at tip top node
    tip_top_nid = (nElem_y) * (nElem_x + 1) + (nElem_x + 1)
    tip_bot_nid = nElem_x + 1

    ts = LinearTS(1, factor=1.0)
    # Distribute force equally between top and bottom tip nodes
    F_per_node = F_total / 2.0
    pat = PlainPattern(1, tseries=ts,
                       load=[[tip_bot_nid, 0.0, F_per_node],
                              [tip_top_nid, 0.0, F_per_node]])
    model.add(pat)

    # Newton analysis
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
    except Exception as e:
        converged = False

    # Get tip displacement
    tip_nd = nodes[tip_bot_nid]
    u_tip = tip_nd._getCommitDisp()
    ux = float(u_tip[0])
    uy = float(u_tip[1])

    return converged, ux, uy


def test_tl_convergence():
    """Test 1: TL Newton converges over 10 steps."""
    converged, ux, uy = build_cantilever(TotalLagrangianContinuumKinematics, nSteps=10)
    return converged, ux, uy


def test_ul_convergence():
    """Test 2: UL Newton converges over 10 steps."""
    converged, ux, uy = build_cantilever(UpdatedLagrangianContinuumKinematics, nSteps=10)
    return converged, ux, uy


def test_tl_ul_agreement():
    """Test 3: TL and UL produce same result for single small step."""
    F_small = 0.01
    _, ux_tl, uy_tl = build_cantilever(TotalLagrangianContinuumKinematics,
                                            nSteps=1, F_total=F_small)
    _, ux_ul, uy_ul = build_cantilever(UpdatedLagrangianContinuumKinematics,
                                            nSteps=1, F_total=F_small)

    err_x = abs(ux_tl - ux_ul) / max(abs(ux_tl), 1e-15) if abs(ux_tl) > 1e-15 else abs(ux_tl - ux_ul)
    err_y = abs(uy_tl - uy_ul) / max(abs(uy_tl), 1e-15)
    max_err = max(err_x, err_y)

    passed = max_err < 1e-6
    return passed, max_err, (ux_tl, uy_tl), (ux_ul, uy_ul)


def test_large_deformation_reasonable():
    """Test 4: Large deformation is physically reasonable.
    Cantilever tip should deflect downward (uy > 0 for upward load, or uy < 0 for downward).
    We use an upward load, so uy should be positive. Also, x-displacement should be small."""
    converged, ux, uy = build_cantilever(TotalLagrangianContinuumKinematics,
                                              nSteps=10, F_total=F_tip)
    # For upward transverse load, tip should move up (uy > 0)
    # and slightly toward root (ux < 0 or small for nonlinear)
    reasonable = converged and uy > 0
    return reasonable, ux, uy


if __name__ == "__main__":
    print("=" * 60)
    print("Quad4 Cantilever -- Large Deformation (TL / UL)")
    print("  E={}, nu={}, PlaneStress, t={}".format(E_val, nu_val, t_val))
    print("  {}x{} mesh ({} elements), L={}, h={}".format(
        nElem_x, nElem_y, nElem_x * nElem_y, L, h))
    print("=" * 60)

    results = []

    # Test 1: TL convergence
    p, ux, uy = test_tl_convergence()
    print("\n  Test 1 (TL Newton convergence):       {}  (ux={:.6e}, uy={:.6e})".format(
        "PASS" if p else "FAIL", ux, uy))
    results.append(p)

    # Test 2: UL convergence
    p, ux, uy = test_ul_convergence()
    print("  Test 2 (UL Newton convergence):       {}  (ux={:.6e}, uy={:.6e})".format(
        "PASS" if p else "FAIL", ux, uy))
    results.append(p)

    # Test 3: TL == UL
    p, err, (ux_tl, uy_tl), (ux_ul, uy_ul) = test_tl_ul_agreement()
    print("  Test 3 (TL == UL, small load):        {}  (err={:.2e})".format(
        "PASS" if p else "FAIL", err))
    print("    TL: ux={:.6e}, uy={:.6e}".format(ux_tl, uy_tl))
    print("    UL: ux={:.6e}, uy={:.6e}".format(ux_ul, uy_ul))
    results.append(p)

    # Test 4: Reasonable large deformation
    p, ux, uy = test_large_deformation_reasonable()
    print("  Test 4 (reasonable deformation):      {}  (ux={:.6e}, uy={:.6e})".format(
        "PASS" if p else "FAIL", ux, uy))
    results.append(p)

    n_pass = sum(results)
    n_total = len(results)
    print("\n" + "=" * 60)
    print("  {}/{} PASS".format(n_pass, n_total))
    if n_pass == n_total:
        print("  ALL PASS")
    print("=" * 60)
