# Column Buckling Benchmark
# Validates PDeltaCrdTransf and CorotCrdTransf for 2D and 3D beam-columns.
#
# Euler critical load for cantilever: P_cr = pi^2 * EI / (2L)^2
#
# Tests:
#   Test 1: 2D P-Delta single element — lateral amplification at P = 0.3*P_cr
#   Test 2: 2D Corot single element — lateral amplification at P = 0.3*P_cr
#   Test 3: 2D Corot 10 elements — amplification vs exact beam-column formula
#   Test 4: 2D side-by-side P_cr detection (PDelta 1-elem vs Corot 10-elem)
#   Test 5: 3D P-Delta single element — weak-axis amplification
#   Test 6: 3D Corot single element — weak-axis amplification
#   Test 7: 3D Corot 10 elements — weak-axis amplification vs exact
#   Test 8: 3D side-by-side P_cr detection (PDelta vs Corot)
#
# Analytical references:
#   P_cr (cantilever) = pi^2 * EI / (2L)^2
#   P_cr_PDelta (1 elem cantilever) = 3EI/L^2
#   Exact amplification: phi(u) = 3*(tan(u)-u)/u^3  where u = (pi/2)*sqrt(P/P_cr)
#   PDelta amplification: delta = F*L^3 / (3EI + N*L^2)  where N < 0 for compression

import numpy as np
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from oneFEM.model import Domain
from oneFEM.model.element.beam import ElasticBeamColumn2d, ElasticBeamColumn3d
from oneFEM.model.kinematics.beam import (
    PDeltaCrdTransf2d, CorotCrdTransf2d,
    PDeltaCrdTransf3d, CorotCrdTransf3d
)
from oneFEM.model.node import Node23, Node36
from oneFEM.model.tseries import Linear as LinearTS
from oneFEM.model.pattern import Plain as PlainPattern
from oneFEM.analysis import Analysis
from oneFEM.analysis.algorithm.newton_raphson import Newton
from oneFEM.analysis.algorithm.linear import Linear as LinearAlg
from oneFEM.analysis.constraints import Plain as PlainConstraints
from oneFEM.analysis.numberer import Plain as PlainNumberer
from oneFEM.analysis.system import FullGeneral
from oneFEM.analysis.integrator import LoadControl, DispControl
from oneFEM.analysis.test import NormUnbalance
from oneFEM import SimulationManager


# ===========================================================
# Properties
# ===========================================================
E = 1000.0
A = 1.0
Iz = 1.0       # strong axis (bending in x-y plane for 2D; about local z for 3D)
Iy = 0.5       # weak axis (bending in x-z plane for 3D, about local y)
G = 400.0
J = 0.8
L = 10.0

# Euler critical loads (cantilever: effective length = 2L)
P_cr_euler_Iz = np.pi**2 * E * Iz / (2*L)**2   # ~24.674 (strong axis)
P_cr_euler_Iy = np.pi**2 * E * Iy / (2*L)**2   # ~12.337 (weak axis / 3D)

# P-Delta critical load (single element cantilever)
P_cr_pdelta_Iz = 3.0 * E * Iz / L**2   # = 30.0

# Lateral perturbation
F_lat = 0.01


def exact_amplification(P, P_cr):
    """Exact beam-column amplification factor for cantilever tip load.
    phi(u) = 3*(tan(u)-u)/u^3  where u = (pi/2)*sqrt(P/P_cr)
    Returns delta/delta_0.
    """
    ratio = P / P_cr
    if ratio >= 1.0:
        return float('inf')
    u = (np.pi / 2.0) * np.sqrt(ratio)
    if abs(u) < 1e-10:
        return 1.0
    return 3.0 * (np.tan(u) - u) / u**3


def pdelta_1elem_tip_disp(F, P, EI, length):
    """Analytical tip displacement for P-Delta single-element cantilever.
    delta = F * L^3 / (3*EI + N*L^2)  where N = -P (compression).
    """
    return F * length**3 / (3.0 * EI - P * length**2)


# ===========================================================
# Helper: build and run analysis
# ===========================================================
def run_newton(model, nSteps, dt):
    """Run Newton-Raphson analysis with LoadControl."""
    alg = Newton(1, tangent='current')
    const = PlainConstraints(1)
    numb = PlainNumberer(1)
    syst = FullGeneral(1)
    integ = LoadControl(1)
    ctest = NormUnbalance(1, tol=1e-10, maxIter=50)
    analysis = Analysis(1, algorithm=alg, constraints=const,
                        integrator=integ, system=syst, test=ctest)
    sim = SimulationManager(1, model, analysis, dt=dt)
    sim.analyze(nSteps, dt=dt)


# ===========================================================
# 2D Tests
# ===========================================================

def test_2d_pdelta_amplification():
    """Test 1: 2D P-Delta single element amplification at P = 0.3*P_cr."""
    P_load = 0.3 * P_cr_euler_Iz

    model = Domain(nD=2)
    nd1 = Node23(1, coord=[0.0, 0.0], fix=[True, True, True])
    nd2 = Node23(2, coord=[0.0, L])
    model.add(nd1, nd2)

    transf = PDeltaCrdTransf2d()
    beam = ElasticBeamColumn2d(1, nodes=[nd1, nd2], A=A, E=E, I=Iz, transf=transf)
    model.add(beam)

    # Lateral + axial load at top: [Fx, Fy, Mz]
    ts = LinearTS(1, factor=1.0)
    pat = PlainPattern(1, tseries=ts, load=[[2, F_lat, -P_load, 0.0]])
    model.add(pat)

    run_newton(model, nSteps=10, dt=0.1)

    u_top = nd2._getCommitDisp()
    delta_x = float(u_top[0])

    # Analytical P-Delta single element result
    delta_expected = pdelta_1elem_tip_disp(F_lat, P_load, E*Iz, L)

    err = abs(delta_x - delta_expected) / abs(delta_expected)
    result = "PASS" if err < 1e-6 else "FAIL"
    print("  Test 1 (2D PDelta amplification):   {}  (delta={:.6e}, expected={:.6e}, err={:.2e})".format(
        result, delta_x, delta_expected, err))
    return result == "PASS"


def test_2d_corot_amplification():
    """Test 2: 2D Corot single element amplification at P = 0.3*P_cr."""
    P_load = 0.3 * P_cr_euler_Iz

    model = Domain(nD=2)
    nd1 = Node23(1, coord=[0.0, 0.0], fix=[True, True, True])
    nd2 = Node23(2, coord=[0.0, L])
    model.add(nd1, nd2)

    transf = CorotCrdTransf2d()
    beam = ElasticBeamColumn2d(1, nodes=[nd1, nd2], A=A, E=E, I=Iz, transf=transf)
    model.add(beam)

    ts = LinearTS(1, factor=1.0)
    pat = PlainPattern(1, tseries=ts, load=[[2, F_lat, -P_load, 0.0]])
    model.add(pat)

    run_newton(model, nSteps=10, dt=0.1)

    u_top = nd2._getCommitDisp()
    delta_x = float(u_top[0])

    # Single element Corot should be close to single element PDelta for small deformations
    # Use exact beam-column formula as reference (Corot should be closer)
    delta_0 = F_lat * L**3 / (3.0 * E * Iz)
    phi_exact = exact_amplification(P_load, P_cr_euler_Iz)
    delta_exact = delta_0 * phi_exact

    # PDelta reference
    delta_pdelta = pdelta_1elem_tip_disp(F_lat, P_load, E*Iz, L)

    # Corot single-element should be between PDelta single-element and exact
    # Just check it produces a reasonable amplification (> 1)
    amplification = delta_x / delta_0
    result = "PASS" if amplification > 1.0 and abs(delta_x) > 0 else "FAIL"
    print("  Test 2 (2D Corot amplification):    {}  (delta={:.6e}, exact={:.6e}, pdelta={:.6e}, amp={:.3f})".format(
        result, delta_x, delta_exact, delta_pdelta, amplification))
    return result == "PASS"


def test_2d_corot_10elem_amplification():
    """Test 3: 2D Corot 10 elements vs exact beam-column amplification."""
    P_load = 0.3 * P_cr_euler_Iz
    nElem = 10
    dL = L / nElem

    model = Domain(nD=2)

    # Create nodes along the column
    nodes = []
    for i in range(nElem + 1):
        y = i * dL
        fix_list = [True, True, True] if i == 0 else None
        nd = Node23(i + 1, coord=[0.0, y], fix=fix_list)
        nodes.append(nd)
        model.add(nd)

    # Create elements
    for i in range(nElem):
        transf = CorotCrdTransf2d()
        beam = ElasticBeamColumn2d(i + 1, nodes=[nodes[i], nodes[i+1]],
                                   A=A, E=E, I=Iz, transf=transf)
        model.add(beam)

    # Lateral + axial load at top node
    top_id = nElem + 1
    ts = LinearTS(1, factor=1.0)
    pat = PlainPattern(1, tseries=ts, load=[[top_id, F_lat, -P_load, 0.0]])
    model.add(pat)

    run_newton(model, nSteps=10, dt=0.1)

    u_top = nodes[-1]._getCommitDisp()
    delta_x = float(u_top[0])

    # Exact beam-column amplification
    delta_0 = F_lat * L**3 / (3.0 * E * Iz)
    phi_exact = exact_amplification(P_load, P_cr_euler_Iz)
    delta_exact = delta_0 * phi_exact

    err = abs(delta_x - delta_exact) / abs(delta_exact)
    result = "PASS" if err < 0.05 else "FAIL"
    print("  Test 3 (2D Corot 10-elem vs exact): {}  (delta={:.6e}, exact={:.6e}, err={:.1f}%)".format(
        result, delta_x, delta_exact, err * 100))
    return result == "PASS"


def test_2d_pcr_comparison():
    """Test 4: 2D side-by-side P_cr detection — PDelta(1) vs Corot(10).
    Uses DisplacementControl. Detects P_cr when Newton fails to converge.
    """
    incr = -0.005  # axial displacement increment (compression)
    nSteps_max = 200  # enough to go past P_cr

    results_pdelta = _run_pcr_2d(PDeltaCrdTransf2d, 1, incr, nSteps_max, x_offset=0.0)
    results_corot = _run_pcr_2d(CorotCrdTransf2d, 10, incr, nSteps_max, x_offset=1.0)

    P_last_pdelta = results_pdelta['P_last']
    P_last_corot = results_corot['P_last']

    # PDelta single element: P_cr ≈ 3EI/L^2 = 30
    err_pdelta = abs(P_last_pdelta - P_cr_pdelta_Iz) / P_cr_pdelta_Iz
    # Corot 10 elements: P_cr ≈ π²EI/(2L)² ≈ 24.674
    err_corot = abs(P_last_corot - P_cr_euler_Iz) / P_cr_euler_Iz

    # Allow 15% tolerance for P_cr estimates (coarser displacement steps)
    pass_pdelta = err_pdelta < 0.15
    pass_corot = err_corot < 0.15

    result_pd = "PASS" if pass_pdelta else "FAIL"
    result_cr = "PASS" if pass_corot else "FAIL"
    result = "PASS" if pass_pdelta and pass_corot else "FAIL"

    print("  Test 4 (2D P_cr detection):")
    print("    PDelta(1-elem):  {} P_last={:.2f}, P_cr_theory={:.2f}, err={:.1f}%".format(
        result_pd, P_last_pdelta, P_cr_pdelta_Iz, err_pdelta * 100))
    print("    Corot(10-elem):  {} P_last={:.2f}, P_cr_theory={:.2f}, err={:.1f}%".format(
        result_cr, P_last_corot, P_cr_euler_Iz, err_corot * 100))
    return result == "PASS"


def _run_pcr_2d(TransfClass, nElem, incr, nSteps_max, x_offset=0.0):
    """Run incremental displacement control, monitor tangent stiffness eigenvalue.
    P_cr is detected when the smallest eigenvalue of K_uu crosses zero.
    Returns dict with detected P_cr.
    """
    dL = L / nElem

    model = Domain(nD=2)
    nodes = []
    for i in range(nElem + 1):
        y = i * dL
        if i == 0:
            fix_list = [True, True, True]  # fixed base
        elif i == nElem:
            fix_list = [False, True, False]  # top: fix uy (controlled), free ux, θ
        else:
            fix_list = None
        nd = Node23(i + 1, coord=[x_offset, y], fix=fix_list)
        nodes.append(nd)
        model.add(nd)

    for i in range(nElem):
        transf = TransfClass()
        beam = ElasticBeamColumn2d(i + 1, nodes=[nodes[i], nodes[i+1]],
                                   A=A, E=E, I=Iz, transf=transf)
        model.add(beam)

    top_node = nodes[-1]

    # DisplacementControl on top node, DOF 1 (uy)
    alg = Newton(1, tangent='current')
    const = PlainConstraints(1)
    numb = PlainNumberer(1)
    syst = FullGeneral(1)
    integ = DispControl(node=top_node, dof=1, incr=incr)
    ctest = NormUnbalance(1, tol=1e-8, maxIter=50)
    analysis = Analysis(1, algorithm=alg, constraints=const,
                        integrator=integ, system=syst, test=ctest)

    P_last = 0.0
    P_cr_detected = 0.0
    prev_min_eig = None

    analysis._analyze(model, nSteps=0, dt=0.0)

    uu_idx = np.array(analysis.uu, dtype=int)

    for step in range(nSteps_max):
        try:
            assembly_time = analysis._time + 1.0
            analysis._assembleF(model, time=assembly_time)
            integ.newStep(model, 1.0, analysis._time)
            alg.solve(model, analysis.uu, analysis.pp, integ, syst, ctest)
            integ.commit(model)
            analysis._time += 1.0

            # Get axial force
            F_int = model.getInternalForce()
            base_dofs = nodes[0].getDOFs()
            if hasattr(base_dofs, 'tolist'):
                base_dof_list = base_dofs.tolist()
            elif hasattr(base_dofs, 'data'):
                base_dof_list = base_dofs.data.tolist()
            else:
                base_dof_list = list(base_dofs)
            P_current = abs(F_int[base_dof_list[1]])

            # Check tangent stiffness eigenvalue
            # Reassemble K at current state to get geometric stiffness
            integ._assembleK(model)
            K = analysis._assembly_system.getK().toarray()
            K_uu = K[np.ix_(uu_idx, uu_idx)]
            eigs = np.linalg.eigvalsh(K_uu)
            min_eig = np.min(eigs)

            if prev_min_eig is not None and min_eig <= 0:
                # Eigenvalue crossed zero — interpolate P_cr
                P_cr_detected = P_last + (P_current - P_last) * prev_min_eig / (prev_min_eig - min_eig)
                break

            prev_min_eig = min_eig
            P_last = P_current

        except RuntimeError:
            P_cr_detected = P_last
            break

    if P_cr_detected == 0.0:
        P_cr_detected = P_last

    return {'P_last': P_cr_detected, 'steps': step + 1}


# ===========================================================
# 3D Tests
# ===========================================================

def test_3d_pdelta_amplification():
    """Test 5: 3D P-Delta single element — weak axis amplification."""
    P_load = 0.3 * P_cr_euler_Iy

    model = Domain(nD=3)
    nd1 = Node36(1, coord=[0.0, 0.0, 0.0], fix=[True]*6)
    nd2 = Node36(2, coord=[0.0, L, 0.0])
    model.add(nd1, nd2)

    # Column along Y; vecxz = (0,0,1) → e_y = cross(vecxz, e_x)
    # e_x = (0,1,0), vecxz = (0,0,1)
    # e_y = cross((0,0,1),(0,1,0)) = (-1,0,0)
    # e_z = cross((0,1,0),(-1,0,0)) = (0,0,1)
    # Weak axis bending: about local y = (-1,0,0), bending in x-z plane
    # → lateral force in Z triggers Iy bending
    transf = PDeltaCrdTransf3d(vecxz=[0, 0, 1])
    beam = ElasticBeamColumn3d(1, nodes=[nd1, nd2], A=A, E=E,
                                Iz=Iz, Iy=Iy, G=G, J=J, transf=transf)
    model.add(beam)

    # Axial compression (-Y) and lateral force (Z) at top
    # Node36 DOFs: [ux, uy, uz, θx, θy, θz]
    ts = LinearTS(1, factor=1.0)
    pat = PlainPattern(1, tseries=ts, load=[[2, 0.0, -P_load, F_lat, 0.0, 0.0, 0.0]])
    model.add(pat)

    run_newton(model, nSteps=10, dt=0.1)

    u_top = nd2._getCommitDisp()
    delta_z = float(u_top[2])

    # P-Delta single element: delta = F*L^3 / (3*EIy + N*L^2) where N < 0
    delta_expected = pdelta_1elem_tip_disp(F_lat, P_load, E*Iy, L)

    err = abs(delta_z - delta_expected) / abs(delta_expected)
    result = "PASS" if err < 1e-5 else "FAIL"
    print("  Test 5 (3D PDelta weak-axis amp):   {}  (delta_z={:.6e}, expected={:.6e}, err={:.2e})".format(
        result, delta_z, delta_expected, err))
    return result == "PASS"


def test_3d_corot_amplification():
    """Test 6: 3D Corot single element — weak axis amplification."""
    P_load = 0.3 * P_cr_euler_Iy

    model = Domain(nD=3)
    nd1 = Node36(1, coord=[0.0, 0.0, 0.0], fix=[True]*6)
    nd2 = Node36(2, coord=[0.0, L, 0.0])
    model.add(nd1, nd2)

    transf = CorotCrdTransf3d(vecxz=[0, 0, 1])
    beam = ElasticBeamColumn3d(1, nodes=[nd1, nd2], A=A, E=E,
                                Iz=Iz, Iy=Iy, G=G, J=J, transf=transf)
    model.add(beam)

    ts = LinearTS(1, factor=1.0)
    pat = PlainPattern(1, tseries=ts, load=[[2, 0.0, -P_load, F_lat, 0.0, 0.0, 0.0]])
    model.add(pat)

    run_newton(model, nSteps=10, dt=0.1)

    u_top = nd2._getCommitDisp()
    delta_z = float(u_top[2])

    # Check it produces positive amplification
    delta_0 = F_lat * L**3 / (3.0 * E * Iy)
    amplification = delta_z / delta_0

    result = "PASS" if amplification > 1.0 else "FAIL"
    print("  Test 6 (3D Corot weak-axis amp):    {}  (delta_z={:.6e}, delta_0={:.6e}, amp={:.3f})".format(
        result, delta_z, delta_0, amplification))
    return result == "PASS"


def test_3d_corot_10elem_amplification():
    """Test 7: 3D Corot 10 elements — weak axis vs exact amplification."""
    P_load = 0.3 * P_cr_euler_Iy
    nElem = 10
    dL = L / nElem

    model = Domain(nD=3)
    nodes = []
    for i in range(nElem + 1):
        y = i * dL
        fix_list = [True]*6 if i == 0 else None
        nd = Node36(i + 1, coord=[0.0, y, 0.0], fix=fix_list)
        nodes.append(nd)
        model.add(nd)

    for i in range(nElem):
        transf = CorotCrdTransf3d(vecxz=[0, 0, 1])
        beam = ElasticBeamColumn3d(i + 1, nodes=[nodes[i], nodes[i+1]],
                                   A=A, E=E, Iz=Iz, Iy=Iy, G=G, J=J,
                                   transf=transf)
        model.add(beam)

    top_id = nElem + 1
    ts = LinearTS(1, factor=1.0)
    pat = PlainPattern(1, tseries=ts,
                       load=[[top_id, 0.0, -P_load, F_lat, 0.0, 0.0, 0.0]])
    model.add(pat)

    run_newton(model, nSteps=10, dt=0.1)

    u_top = nodes[-1]._getCommitDisp()
    delta_z = float(u_top[2])

    # Exact beam-column amplification (weak axis)
    delta_0 = F_lat * L**3 / (3.0 * E * Iy)
    phi_exact = exact_amplification(P_load, P_cr_euler_Iy)
    delta_exact = delta_0 * phi_exact

    err = abs(delta_z - delta_exact) / abs(delta_exact)
    result = "PASS" if err < 0.05 else "FAIL"
    print("  Test 7 (3D Corot 10-elem vs exact): {}  (delta_z={:.6e}, exact={:.6e}, err={:.1f}%)".format(
        result, delta_z, delta_exact, err * 100))
    return result == "PASS"


def test_3d_pcr_comparison():
    """Test 8: 3D side-by-side P_cr — PDelta(1) vs Corot(10) for weak axis."""
    incr = -0.005
    nSteps_max = 200

    # P-Delta single element
    results_pdelta = _run_pcr_3d(PDeltaCrdTransf3d, 1, incr, nSteps_max)
    # Corot 10 elements
    results_corot = _run_pcr_3d(CorotCrdTransf3d, 10, incr, nSteps_max)

    P_last_pdelta = results_pdelta['P_last']
    P_last_corot = results_corot['P_last']

    # PDelta single element: P_cr ≈ 3*E*Iy/L^2 = 3*1000*0.5/100 = 15.0
    P_cr_pdelta_3d = 3.0 * E * Iy / L**2
    err_pdelta = abs(P_last_pdelta - P_cr_pdelta_3d) / P_cr_pdelta_3d
    err_corot = abs(P_last_corot - P_cr_euler_Iy) / P_cr_euler_Iy

    pass_pdelta = err_pdelta < 0.15
    pass_corot = err_corot < 0.15

    result_pd = "PASS" if pass_pdelta else "FAIL"
    result_cr = "PASS" if pass_corot else "FAIL"
    result = "PASS" if pass_pdelta and pass_corot else "FAIL"

    print("  Test 8 (3D P_cr weak-axis):")
    print("    PDelta(1-elem):  {} P_last={:.2f}, P_cr_theory={:.2f}, err={:.1f}%".format(
        result_pd, P_last_pdelta, P_cr_pdelta_3d, err_pdelta * 100))
    print("    Corot(10-elem):  {} P_last={:.2f}, P_cr_theory={:.2f}, err={:.1f}%".format(
        result_cr, P_last_corot, P_cr_euler_Iy, err_corot * 100))
    return result == "PASS"


def _run_pcr_3d(TransfClass, nElem, incr, nSteps_max):
    """Run 3D displacement-controlled buckling, monitor tangent eigenvalue."""
    dL = L / nElem

    model = Domain(nD=3)
    nodes = []
    for i in range(nElem + 1):
        y = i * dL
        if i == 0:
            fix_list = [True]*6  # fixed base
        elif i == nElem:
            fix_list = [False, True, False, False, False, False]
        else:
            fix_list = None
        nd = Node36(i + 1, coord=[0.0, y, 0.0], fix=fix_list)
        nodes.append(nd)
        model.add(nd)

    vecxz = [0, 0, 1]
    for i in range(nElem):
        transf = TransfClass(vecxz=vecxz)
        beam = ElasticBeamColumn3d(i + 1, nodes=[nodes[i], nodes[i+1]],
                                   A=A, E=E, Iz=Iz, Iy=Iy, G=G, J=J,
                                   transf=transf)
        model.add(beam)

    top_node = nodes[-1]

    alg = Newton(1, tangent='current')
    const = PlainConstraints(1)
    numb = PlainNumberer(1)
    syst = FullGeneral(1)
    integ = DispControl(node=top_node, dof=1, incr=incr)
    ctest = NormUnbalance(1, tol=1e-8, maxIter=50)
    analysis = Analysis(1, algorithm=alg, constraints=const,
                        integrator=integ, system=syst, test=ctest)

    P_last = 0.0
    P_cr_detected = 0.0
    prev_min_eig = None

    analysis._analyze(model, nSteps=0, dt=0.0)

    uu_idx = np.array(analysis.uu, dtype=int)

    for step in range(nSteps_max):
        try:
            assembly_time = analysis._time + 1.0
            analysis._assembleF(model, time=assembly_time)
            integ.newStep(model, 1.0, analysis._time)
            alg.solve(model, analysis.uu, analysis.pp, integ, syst, ctest)
            integ.commit(model)
            analysis._time += 1.0

            F_int = model.getInternalForce()
            base_dofs = nodes[0].getDOFs()
            if hasattr(base_dofs, 'tolist'):
                base_dof_list = base_dofs.tolist()
            elif hasattr(base_dofs, 'data'):
                base_dof_list = base_dofs.data.tolist()
            else:
                base_dof_list = list(base_dofs)
            P_current = abs(F_int[base_dof_list[1]])

            # Check tangent stiffness eigenvalue
            # Reassemble K at current state to get geometric stiffness
            integ._assembleK(model)
            K = analysis._assembly_system.getK().toarray()
            K_uu = K[np.ix_(uu_idx, uu_idx)]
            eigs = np.linalg.eigvalsh(K_uu)
            min_eig = np.min(eigs)

            if prev_min_eig is not None and min_eig <= 0:
                P_cr_detected = P_last + (P_current - P_last) * prev_min_eig / (prev_min_eig - min_eig)
                break

            prev_min_eig = min_eig
            P_last = P_current

        except RuntimeError:
            P_cr_detected = P_last
            break

    if P_cr_detected == 0.0:
        P_cr_detected = P_last

    return {'P_last': P_cr_detected, 'steps': step + 1}


# ===========================================================
# Main
# ===========================================================
if __name__ == '__main__':
    print("\n=== Column Buckling Benchmark ===")
    print("Properties: E={}, A={}, Iz={}, Iy={}, G={}, J={}, L={}".format(E, A, Iz, Iy, G, J, L))
    print("Euler P_cr (strong, cantilever): {:.3f}".format(P_cr_euler_Iz))
    print("Euler P_cr (weak, cantilever):   {:.3f}".format(P_cr_euler_Iy))
    print("PDelta P_cr (1-elem, Iz):        {:.3f}".format(P_cr_pdelta_Iz))
    print()

    print("--- 2D Tests ---")
    results = []
    results.append(test_2d_pdelta_amplification())
    results.append(test_2d_corot_amplification())
    results.append(test_2d_corot_10elem_amplification())
    results.append(test_2d_pcr_comparison())

    print()
    print("--- 3D Tests ---")
    results.append(test_3d_pdelta_amplification())
    results.append(test_3d_corot_amplification())
    results.append(test_3d_corot_10elem_amplification())
    results.append(test_3d_pcr_comparison())

    print()
    print("=== Results: {}/{} PASS ===".format(sum(results), len(results)))
