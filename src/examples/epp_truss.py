##-----------------------------------------------------------------------##
#  EPP Truss Benchmark
#
#  Two parallel trusses sharing nodes, different yield strengths.
#  Verifies Newton-Raphson with ElasticPerfectlyPlastic material.
#
#         F(t) -->  o nd2  (free in x only)
#                  ||
#      truss 1    ||  truss 2
#      (fy=200)   ||  (fy=400)  [MPa]
#                  ||
#                  o nd1  (fully fixed)
#
#  E  = 200e9 Pa,  A = 0.01 m^2,  L = 1.0 m
#  EA = 2e9 N per truss, total = 4e9 N
#  fy1 = 200e6 Pa -> Fy1 = 2e6 N,  eps_y1 = 1e-3
#  fy2 = 400e6 Pa -> Fy2 = 4e6 N,  eps_y2 = 2e-3
#
#  Load: F_base = 1e6 N, linear time series (F = F_base * time)
#  6 steps with dt=1.0:
#    Step 1-4: both elastic, u = F / (EA1+EA2)
#    Step 5:   truss 1 yielded, u = 1.5e-3 m
#    Step 6:   truss 1 yielded, truss 2 at yield, u = 2.0e-3 m
##-----------------------------------------------------------------------##

import numpy as np

# --- imports ---
from oneFEM.model import Domain
from oneFEM.model.element.truss import Truss
from oneFEM.model.node import Node36
from oneFEM.model.element.section import Rectangular
from oneFEM.model.material.uniaxial import Elastic, ElasticPerfectlyPlastic
from oneFEM.model.tseries import Linear as LinearSeries
from oneFEM.model.pattern import Plain as PlainPattern
from oneFEM.output.recorder import NodeRecorder, ElementRecorder
from oneFEM.analysis import Analysis
from oneFEM.analysis.algorithm import Linear, Newton
from oneFEM.analysis.constraints import Plain as PlainConstraints
from oneFEM.analysis.numberer import Plain as PlainNumberer
from oneFEM.analysis.system import FullGeneral
from oneFEM.analysis.integrator import LoadControl
from oneFEM.analysis.test import NormUnbalance
from oneFEM import SimulationManager


def run_test():
    all_pass = True

    # ====================================================================
    # TEST 0: Newton-Raphson with elastic material (regression test)
    #         Same triangle truss as truss.py but solved with Newton
    # ====================================================================
    print("\n" + "=" * 60)
    print("  TEST 0: Newton-Raphson regression (elastic triangle truss)")
    print("=" * 60)

    model0 = Domain(nD=3)
    nd1_0 = Node36(1, coord=[0.0, 0.0, 0.0], fix=[1, 1, 1, 1, 1, 1])
    nd2_0 = Node36(2, coord=[2.0, 0.0, 0.0], fix=[0, 1, 1, 1, 1, 1])
    nd3_0 = Node36(3, coord=[2.0, 0.0, 2.0], fix=[0, 1, 0, 1, 1, 1])
    model0.add(nd1_0, nd2_0, nd3_0)

    mat0 = Elastic(1, E=2e9)
    sec0 = Rectangular(1, mat=mat0, h=0.2, w=0.1)
    tr1_0 = Truss(1, nodes=[nd1_0, nd2_0], section=sec0)
    tr2_0 = Truss(2, nodes=[nd2_0, nd3_0], section=sec0)
    tr3_0 = Truss(3, nodes=[nd1_0, nd3_0], section=sec0)
    model0.add(tr1_0, tr2_0, tr3_0)

    from oneFEM.model.tseries import Constant
    t_const = Constant(1, factor=1.0)
    push0 = PlainPattern(1, tseries=t_const, load=[[3, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    model0.add(push0)

    alg0 = Newton(1, tangent='current')
    syst0 = FullGeneral(1)
    ctest0 = NormUnbalance(1, tol=1e-10, maxIter=10)
    ana0 = Analysis(1, algorithm=alg0, constraints=PlainConstraints(1),
                    integrator=LoadControl(1), system=syst0, test=ctest0,
                    numberer=PlainNumberer(1))
    sim0 = SimulationManager(1, model0, ana0, dt=0.0)
    sim0.analyze(1, dt=0.1)

    # Analytical (same as truss.py)
    EA = 2e9 * 0.02
    L3 = 2.0 * np.sqrt(2.0)
    c3 = 1.0 / np.sqrt(2.0)
    K_ff = np.zeros((3, 3))
    K_ff[0, 0] += EA / 2.0
    K_ff[2, 2] += EA / 2.0
    K_ff[1, 1] += (EA / L3) * c3 * c3
    K_ff[1, 2] += (EA / L3) * c3 * c3
    K_ff[2, 1] += (EA / L3) * c3 * c3
    K_ff[2, 2] += (EA / L3) * c3 * c3
    F_ff = np.array([0.0, 1.0, 0.0])
    u_analytical = np.linalg.solve(K_ff, F_ff)

    u_nd2_x = float(nd2_0._getCommitDisp()[0])
    u_nd3_x = float(nd3_0._getCommitDisp()[0])
    u_nd3_z = float(nd3_0._getCommitDisp()[2])

    tol = 1e-10
    p0a = abs(u_nd2_x - u_analytical[0]) < tol
    p0b = abs(u_nd3_x - u_analytical[1]) < tol
    p0c = abs(u_nd3_z - u_analytical[2]) < tol
    t0_pass = p0a and p0b and p0c

    print(f"  nd2 x: num={u_nd2_x:.10e}  ana={u_analytical[0]:.10e}  {'PASS' if p0a else 'FAIL'}")
    print(f"  nd3 x: num={u_nd3_x:.10e}  ana={u_analytical[1]:.10e}  {'PASS' if p0b else 'FAIL'}")
    print(f"  nd3 z: num={u_nd3_z:.10e}  ana={u_analytical[2]:.10e}  {'PASS' if p0c else 'FAIL'}")
    print(f"  Test 0: {'PASS' if t0_pass else 'FAIL'}")
    all_pass = all_pass and t0_pass

    # ====================================================================
    # TEST 1: EPP two-truss system — incremental loading with yielding
    # ====================================================================
    print("\n" + "=" * 60)
    print("  TEST 1: EPP two-truss system (nonlinear Newton-Raphson)")
    print("=" * 60)

    E = 200e9       # Pa
    A = 0.01        # m^2
    fy1 = 200e6     # Pa
    fy2 = 400e6     # Pa
    L = 1.0         # m
    F_base = 1.0e6  # N

    model = Domain(nD=3)

    nd1 = Node36(1, coord=[0.0, 0.0, 0.0], fix=[1, 1, 1, 1, 1, 1])
    nd2 = Node36(2, coord=[L, 0.0, 0.0], fix=[0, 1, 1, 1, 1, 1])
    model.add(nd1, nd2)

    # Truss 1: fy = 200 MPa
    mat1 = ElasticPerfectlyPlastic(1, E=E, fy=fy1)
    sec1 = Rectangular(1, mat=mat1, h=np.sqrt(A), w=np.sqrt(A))

    # Truss 2: fy = 400 MPa
    mat2 = ElasticPerfectlyPlastic(2, E=E, fy=fy2)
    sec2 = Rectangular(2, mat=mat2, h=np.sqrt(A), w=np.sqrt(A))

    tr1 = Truss(1, nodes=[nd1, nd2], section=sec1)
    tr2 = Truss(2, nodes=[nd1, nd2], section=sec2)
    model.add(tr1, tr2)

    # Linear time series: factor * time
    t_lin = LinearSeries(1, factor=1.0)
    load = [[2, F_base, 0.0, 0.0, 0.0, 0.0, 0.0]]
    pat = PlainPattern(1, tseries=t_lin, load=load)
    model.add(pat)

    # Recorders
    nd_rec = NodeRecorder(1, nd2, dofs=[1], results=['displacement'])
    el_rec1 = ElementRecorder(2, tr1, results=['strain', 'stress'])
    el_rec2 = ElementRecorder(3, tr2, results=['strain', 'stress'])
    model.add(nd_rec, el_rec1, el_rec2)

    # Newton-Raphson analysis
    alg = Newton(1, tangent='current')
    syst = FullGeneral(1)
    ctest = NormUnbalance(1, tol=1e-8, maxIter=20, printFlag=1)
    ana = Analysis(1, algorithm=alg, constraints=PlainConstraints(1),
                   integrator=LoadControl(1), system=syst, test=ctest,
                   numberer=PlainNumberer(1))
    sim = SimulationManager(1, model, ana, dt=0.0)

    # Run 6 load steps
    nSteps = 6
    sim.analyze(nSteps, dt=1.0)

    # Analytical solutions (total displacement at each step)
    EA1 = E * A
    EA2 = E * A
    K_total = (EA1 + EA2) / L  # 4e9

    u_analytical = np.zeros(nSteps)
    # Steps 1-4: both elastic
    for i in range(4):
        F = F_base * (i + 1)
        u_analytical[i] = F / K_total

    # Step 5: truss 1 yielded
    # F_ext = 5e6 = fy1*A + E*eps*A -> eps = (5e6 - 2e6) / (E*A) = 3e6/2e9 = 1.5e-3
    u_analytical[4] = (F_base * 5 - fy1 * A) / (EA2 / L)

    # Step 6: truss 1 yielded, truss 2 at yield boundary
    # F_ext = 6e6 = fy1*A + E*eps*A -> eps = (6e6 - 2e6) / 2e9 = 2e-3
    u_analytical[5] = (F_base * 6 - fy1 * A) / (EA2 / L)

    # Get numerical results
    u_numerical = np.array([float(v[0]) if hasattr(v, '__getitem__') else float(v)
                            for v in nd_rec.data['displacement']])

    print(f"\n  {'Step':>4s}  {'F_ext [N]':>12s}  {'u_num [m]':>14s}  {'u_ana [m]':>14s}  {'Rel Err':>10s}  {'Status':>6s}")
    print("  " + "-" * 70)

    for i in range(nSteps):
        F = F_base * (i + 1)
        rel_err = abs(u_numerical[i] - u_analytical[i]) / abs(u_analytical[i])
        passed = rel_err < 1e-8
        all_pass = all_pass and passed
        print(f"  {i+1:4d}  {F:12.2e}  {u_numerical[i]:14.6e}  {u_analytical[i]:14.6e}  {rel_err:10.2e}  {'PASS' if passed else 'FAIL'}")

    # Verify element stresses (section stress = material_stress * A)
    print("\n  Element section force verification (stress * A):")
    for i in range(nSteps):
        eps = u_numerical[i] / L
        s1_val = float(el_rec1.data['stress'][i])
        s2_val = float(el_rec2.data['stress'][i])

        if i < 4:
            s1_expected = E * eps * A  # elastic
            s2_expected = E * eps * A
        elif i == 4:
            s1_expected = fy1 * A  # yielded
            s2_expected = E * (u_analytical[4] / L) * A  # elastic
        else:
            s1_expected = fy1 * A
            s2_expected = fy2 * A  # at yield boundary

        p_s1 = abs(s1_val - s1_expected) < 1.0  # tolerance in N
        p_s2 = abs(s2_val - s2_expected) < 1.0
        all_pass = all_pass and p_s1 and p_s2
        print(f"    Step {i+1}: F1={s1_val:12.2f}  exp={s1_expected:12.2f} {'PASS' if p_s1 else 'FAIL'}"
              f"  |  F2={s2_val:12.2f}  exp={s2_expected:12.2f} {'PASS' if p_s2 else 'FAIL'}")

    # ====================================================================
    # TEST 2: Single truss elastic, Newton vs Linear (must match)
    # ====================================================================
    print("\n" + "=" * 60)
    print("  TEST 2: Newton elastic == Linear elastic (single truss)")
    print("=" * 60)

    # Run with Linear
    model_lin = Domain(nD=3)
    nd1_l = Node36(1, coord=[0.0, 0.0, 0.0], fix=[1, 1, 1, 1, 1, 1])
    nd2_l = Node36(2, coord=[1.0, 0.0, 0.0], fix=[0, 1, 1, 1, 1, 1])
    model_lin.add(nd1_l, nd2_l)
    mat_l = Elastic(1, E=200e9)
    sec_l = Rectangular(1, mat=mat_l, h=0.1, w=0.1)
    tr_l = Truss(1, nodes=[nd1_l, nd2_l], section=sec_l)
    model_lin.add(tr_l)
    t_const2 = Constant(2, factor=1.0)
    pat_l = PlainPattern(1, tseries=t_const2, load=[[2, 1e6, 0.0, 0.0, 0.0, 0.0, 0.0]])
    model_lin.add(pat_l)
    ana_l = Analysis(1, algorithm=Linear(1), constraints=PlainConstraints(1),
                     integrator=LoadControl(1), system=FullGeneral(1),
                     test=NormUnbalance(1), numberer=PlainNumberer(1))
    sim_l = SimulationManager(1, model_lin, ana_l, dt=0.0)
    sim_l.analyze(1, dt=0.1)
    u_linear = float(nd2_l._getCommitDisp()[0])

    # Run with Newton
    model_nw = Domain(nD=3)
    nd1_n = Node36(1, coord=[0.0, 0.0, 0.0], fix=[1, 1, 1, 1, 1, 1])
    nd2_n = Node36(2, coord=[1.0, 0.0, 0.0], fix=[0, 1, 1, 1, 1, 1])
    model_nw.add(nd1_n, nd2_n)
    mat_n = Elastic(1, E=200e9)
    sec_n = Rectangular(1, mat=mat_n, h=0.1, w=0.1)
    tr_n = Truss(1, nodes=[nd1_n, nd2_n], section=sec_n)
    model_nw.add(tr_n)
    pat_n = PlainPattern(1, tseries=Constant(2, factor=1.0),
                         load=[[2, 1e6, 0.0, 0.0, 0.0, 0.0, 0.0]])
    model_nw.add(pat_n)
    ana_n = Analysis(1, algorithm=Newton(1, tangent='current'),
                     constraints=PlainConstraints(1),
                     integrator=LoadControl(1), system=FullGeneral(1),
                     test=NormUnbalance(1, tol=1e-10, maxIter=10),
                     numberer=PlainNumberer(1))
    sim_n = SimulationManager(1, model_nw, ana_n, dt=0.0)
    sim_n.analyze(1, dt=0.1)
    u_newton = float(nd2_n._getCommitDisp()[0])

    t2_pass = abs(u_linear - u_newton) < 1e-15
    all_pass = all_pass and t2_pass
    print(f"  Linear: u = {u_linear:.15e}")
    print(f"  Newton: u = {u_newton:.15e}")
    print(f"  Match: {'PASS' if t2_pass else 'FAIL'}")

    # ====================================================================
    # SUMMARY
    # ====================================================================
    print("\n" + "=" * 60)
    print(f"  OVERALL: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 60 + "\n")

    return all_pass


if __name__ == '__main__':
    run_test()
