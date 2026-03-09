# Elastic Beam-Column 2D/3D Benchmark
# Validates ElasticBeamColumn2d/3d with LinearCrdTransf2d/3d
# against analytical solutions.
#
# 2D Tests:
#   Test 1: Stiffness matrix — horizontal beam K vs analytical 6x6
#   Test 2: Cantilever — tip point load
#   Test 3: Cantilever — tip moment
#   Test 4: Simply supported — center point load (2 elements)
#   Test 5: Inclined cantilever (45 deg) — tip load
#
# 3D Tests:
#   Test 6: 3D stiffness matrix — beam along X vs analytical 12x12
#   Test 7: 3D cantilever — tip load in Y (bending about z)
#   Test 8: 3D cantilever — tip load in Z (bending about y)
#   Test 9: 3D cantilever — tip torsion
#   Test 10: 3D inclined cantilever — tip load
#
# Expected pass criterion: rel error < 1e-10

import numpy as np

# import domain
from oneFEM.model import Domain
# modeling tools
from oneFEM.model.element.beam import ElasticBeamColumn2d, ElasticBeamColumn3d
from oneFEM.model.kinematics.beam import LinearCrdTransf2d, LinearCrdTransf3d
from oneFEM.model.node import Node23, Node36
from oneFEM.model.tseries import Constant
from oneFEM.model.pattern import Plain as PlainPattern
# import analysis
from oneFEM.analysis import Analysis
from oneFEM.analysis.algorithm import Linear
from oneFEM.analysis.constraints import Plain as PlainConstraints
from oneFEM.analysis.numberer import Plain as PlainNumberer
from oneFEM.analysis.system import FullGeneral
from oneFEM.analysis.integrator import LoadControl
from oneFEM.analysis.test import NormUnbalance
# import simulation manager
from oneFEM import SimulationManager


# Shared properties
E = 100.0
A = 10.0
Iz = 1.0      # bending in x-y plane (about z)
Iy = 0.5      # bending in x-z plane (about y)
G = 40.0      # shear modulus
J = 0.8       # torsional constant
L = 10.0
TOL = 1e-10


def run_analysis(model):
    """Helper: set up and run a single static load step."""
    alg = Linear(1)
    const = PlainConstraints(1)
    numb = PlainNumberer(1)
    syst = FullGeneral(1)
    integ = LoadControl(1)
    ctest = NormUnbalance(1)
    analysis = Analysis(1, algorithm=alg, constraints=const,
                        integrator=integ, system=syst, test=ctest)
    sim = SimulationManager(1, model, analysis, dt=0.0)
    sim.analyze(1, dt=0.1)


# ============================================================
#  2D Tests
# ============================================================

def test_stiffness_matrix_2d():
    """Test 1: 2D stiffness matrix verification (horizontal beam)."""
    model = Domain(nD=2)
    nd1 = Node23(1, coord=[0.0, 0.0])
    nd2 = Node23(2, coord=[L, 0.0])
    model.add(nd1, nd2)

    transf = LinearCrdTransf2d()
    beam = ElasticBeamColumn2d(1, nodes=[nd1, nd2], A=A, E=E, I=Iz, transf=transf)
    model.add(beam)
    beam._domain()

    K_num = np.asarray(beam.getStiffness())

    EA = E * A
    EI = E * Iz
    K_exact = np.array([
        [ EA/L,    0,           0,        -EA/L,    0,           0       ],
        [ 0,       12*EI/L**3,  6*EI/L**2, 0,      -12*EI/L**3, 6*EI/L**2],
        [ 0,       6*EI/L**2,   4*EI/L,    0,      -6*EI/L**2,  2*EI/L   ],
        [-EA/L,    0,           0,         EA/L,    0,           0        ],
        [ 0,      -12*EI/L**3, -6*EI/L**2, 0,       12*EI/L**3,-6*EI/L**2],
        [ 0,       6*EI/L**2,   2*EI/L,    0,      -6*EI/L**2,  4*EI/L   ]
    ])

    max_err = np.max(np.abs(K_num - K_exact))
    rel_err = max_err / np.max(np.abs(K_exact))
    passed = rel_err < TOL

    print("  Test 1: 2D stiffness matrix (horizontal beam)")
    print(f"    max abs error = {max_err:.2e},  rel_err = {rel_err:.2e}  {'PASS' if passed else 'FAIL'}")
    return passed


def test_cantilever_tip_load_2d():
    """Test 2: 2D cantilever with tip point load."""
    P = 1.0
    model = Domain(nD=2)
    nd1 = Node23(1, coord=[0.0, 0.0], fix=[1, 1, 1])
    nd2 = Node23(2, coord=[L, 0.0])
    model.add(nd1, nd2)

    transf = LinearCrdTransf2d()
    beam = ElasticBeamColumn2d(1, nodes=[nd1, nd2], A=A, E=E, I=Iz, transf=transf)
    model.add(beam)

    t_const = Constant(1, factor=1.0)
    pattern = PlainPattern(1, tseries=t_const, load=[[2, 0.0, -P, 0.0]])
    model.add(pattern)

    run_analysis(model)

    u_y = nd2._getCommitDisp()[1]
    theta_z = nd2._getCommitDisp()[2]

    EI = E * Iz
    delta_exact = -P * L**3 / (3 * EI)
    theta_exact = -P * L**2 / (2 * EI)

    err_d = abs(u_y - delta_exact) / abs(delta_exact)
    err_t = abs(theta_z - theta_exact) / abs(theta_exact)
    pass_d = err_d < TOL
    pass_t = err_t < TOL

    print("  Test 2: 2D cantilever — tip load")
    print(f"    delta_y:  num = {u_y:.10e},  exact = {delta_exact:.10e},  err = {err_d:.2e}  {'PASS' if pass_d else 'FAIL'}")
    print(f"    theta_z:  num = {theta_z:.10e},  exact = {theta_exact:.10e},  err = {err_t:.2e}  {'PASS' if pass_t else 'FAIL'}")
    return pass_d and pass_t


def test_cantilever_tip_moment_2d():
    """Test 3: 2D cantilever with tip moment."""
    M = 1.0
    model = Domain(nD=2)
    nd1 = Node23(1, coord=[0.0, 0.0], fix=[1, 1, 1])
    nd2 = Node23(2, coord=[L, 0.0])
    model.add(nd1, nd2)

    transf = LinearCrdTransf2d()
    beam = ElasticBeamColumn2d(1, nodes=[nd1, nd2], A=A, E=E, I=Iz, transf=transf)
    model.add(beam)

    t_const = Constant(1, factor=1.0)
    pattern = PlainPattern(1, tseries=t_const, load=[[2, 0.0, 0.0, M]])
    model.add(pattern)

    run_analysis(model)

    u_y = nd2._getCommitDisp()[1]
    theta_z = nd2._getCommitDisp()[2]

    EI = E * Iz
    delta_exact = M * L**2 / (2 * EI)
    theta_exact = M * L / EI

    err_d = abs(u_y - delta_exact) / abs(delta_exact)
    err_t = abs(theta_z - theta_exact) / abs(theta_exact)
    pass_d = err_d < TOL
    pass_t = err_t < TOL

    print("  Test 3: 2D cantilever — tip moment")
    print(f"    delta_y:  num = {u_y:.10e},  exact = {delta_exact:.10e},  err = {err_d:.2e}  {'PASS' if pass_d else 'FAIL'}")
    print(f"    theta_z:  num = {theta_z:.10e},  exact = {theta_exact:.10e},  err = {err_t:.2e}  {'PASS' if pass_t else 'FAIL'}")
    return pass_d and pass_t


def test_simply_supported_center_load_2d():
    """Test 4: 2D simply supported beam with center point load (2 elements)."""
    P = 1.0
    L_total = 2 * L

    model = Domain(nD=2)
    nd1 = Node23(1, coord=[0.0, 0.0], fix=[1, 1, 0])       # pin
    nd2 = Node23(2, coord=[L, 0.0])                          # center
    nd3 = Node23(3, coord=[2*L, 0.0], fix=[0, 1, 0])        # roller
    model.add(nd1, nd2, nd3)

    transf1 = LinearCrdTransf2d()
    transf2 = LinearCrdTransf2d()
    beam1 = ElasticBeamColumn2d(1, nodes=[nd1, nd2], A=A, E=E, I=Iz, transf=transf1)
    beam2 = ElasticBeamColumn2d(2, nodes=[nd2, nd3], A=A, E=E, I=Iz, transf=transf2)
    model.add(beam1, beam2)

    t_const = Constant(1, factor=1.0)
    pattern = PlainPattern(1, tseries=t_const, load=[[2, 0.0, -P, 0.0]])
    model.add(pattern)

    run_analysis(model)

    EI = E * Iz
    u_center = nd2._getCommitDisp()[1]
    theta_end = nd1._getCommitDisp()[2]
    delta_exact = -P * L_total**3 / (48 * EI)
    theta_exact = -P * L_total**2 / (16 * EI)

    err_d = abs(u_center - delta_exact) / abs(delta_exact)
    err_t = abs(theta_end - theta_exact) / abs(theta_exact)
    pass_d = err_d < TOL
    pass_t = err_t < TOL

    print("  Test 4: 2D simply supported — center load")
    print(f"    delta:    num = {u_center:.10e},  exact = {delta_exact:.10e},  err = {err_d:.2e}  {'PASS' if pass_d else 'FAIL'}")
    print(f"    theta:    num = {theta_end:.10e},  exact = {theta_exact:.10e},  err = {err_t:.2e}  {'PASS' if pass_t else 'FAIL'}")
    return pass_d and pass_t


def test_inclined_cantilever_2d():
    """Test 5: 2D inclined cantilever (45 deg) with tip load in global Y."""
    P = 1.0
    angle = np.pi / 4.0
    c45 = np.cos(angle)
    s45 = np.sin(angle)

    model = Domain(nD=2)
    nd1 = Node23(1, coord=[0.0, 0.0], fix=[1, 1, 1])
    nd2 = Node23(2, coord=[L*c45, L*s45])
    model.add(nd1, nd2)

    transf = LinearCrdTransf2d()
    beam = ElasticBeamColumn2d(1, nodes=[nd1, nd2], A=A, E=E, I=Iz, transf=transf)
    model.add(beam)

    t_const = Constant(1, factor=1.0)
    pattern = PlainPattern(1, tseries=t_const, load=[[2, 0.0, -P, 0.0]])
    model.add(pattern)

    run_analysis(model)

    u_x = nd2._getCommitDisp()[0]
    u_y = nd2._getCommitDisp()[1]
    theta = nd2._getCommitDisp()[2]

    EA = E * A
    EI = E * Iz
    P_t = P * c45
    P_a = P * s45
    v_local = P_t * L**3 / (3 * EI)
    u_local = -P_a * L / EA
    theta_local = P_t * L**2 / (2 * EI)

    u_x_exact = u_local * c45 + v_local * s45
    u_y_exact = u_local * s45 - v_local * c45
    theta_exact = -theta_local

    err_ux = abs(u_x - u_x_exact) / max(abs(u_x_exact), 1e-30)
    err_uy = abs(u_y - u_y_exact) / abs(u_y_exact)
    err_t = abs(theta - theta_exact) / abs(theta_exact)
    pass_ux = err_ux < TOL
    pass_uy = err_uy < TOL
    pass_t = err_t < TOL

    print("  Test 5: 2D inclined cantilever (45 deg)")
    print(f"    u_x:      num = {u_x:.10e},  exact = {u_x_exact:.10e},  err = {err_ux:.2e}  {'PASS' if pass_ux else 'FAIL'}")
    print(f"    u_y:      num = {u_y:.10e},  exact = {u_y_exact:.10e},  err = {err_uy:.2e}  {'PASS' if pass_uy else 'FAIL'}")
    print(f"    theta_z:  num = {theta:.10e},  exact = {theta_exact:.10e},  err = {err_t:.2e}  {'PASS' if pass_t else 'FAIL'}")
    return pass_ux and pass_uy and pass_t


# ============================================================
#  3D Tests
# ============================================================

def test_stiffness_matrix_3d():
    """Test 6: 3D stiffness matrix verification (beam along X-axis)."""
    model = Domain(nD=3)
    nd1 = Node36(1, coord=[0.0, 0.0, 0.0])
    nd2 = Node36(2, coord=[L, 0.0, 0.0])
    model.add(nd1, nd2)

    transf = LinearCrdTransf3d(vecxz=[0.0, 0.0, 1.0])
    beam = ElasticBeamColumn3d(1, nodes=[nd1, nd2], A=A, E=E,
                                Iz=Iz, Iy=Iy, G=G, J=J, transf=transf)
    model.add(beam)
    beam._domain()

    K_num = np.asarray(beam.getStiffness())

    # For a beam along X with vecxz=[0,0,1]:
    #   local x = global X, local y = global Y, local z = global Z
    # So K_local = K_global (R = I)
    EA = E * A
    EIz = E * Iz
    EIy = E * Iy
    GJ = G * J

    K_exact = np.zeros((12, 12))

    # Axial (u DOFs: 0, 6)
    K_exact[0, 0] = EA/L;    K_exact[0, 6] = -EA/L
    K_exact[6, 0] = -EA/L;   K_exact[6, 6] = EA/L

    # Bending x-y plane: v DOFs (1,7), θz DOFs (5,11) — uses EIz
    K_exact[1, 1] = 12*EIz/L**3;   K_exact[1, 5] = 6*EIz/L**2
    K_exact[1, 7] = -12*EIz/L**3;  K_exact[1, 11] = 6*EIz/L**2
    K_exact[5, 1] = 6*EIz/L**2;    K_exact[5, 5] = 4*EIz/L
    K_exact[5, 7] = -6*EIz/L**2;   K_exact[5, 11] = 2*EIz/L
    K_exact[7, 1] = -12*EIz/L**3;  K_exact[7, 5] = -6*EIz/L**2
    K_exact[7, 7] = 12*EIz/L**3;   K_exact[7, 11] = -6*EIz/L**2
    K_exact[11, 1] = 6*EIz/L**2;   K_exact[11, 5] = 2*EIz/L
    K_exact[11, 7] = -6*EIz/L**2;  K_exact[11, 11] = 4*EIz/L

    # Bending x-z plane: w DOFs (2,8), θy DOFs (4,10) — uses EIy
    K_exact[2, 2] = 12*EIy/L**3;    K_exact[2, 4] = -6*EIy/L**2
    K_exact[2, 8] = -12*EIy/L**3;   K_exact[2, 10] = -6*EIy/L**2
    K_exact[4, 2] = -6*EIy/L**2;    K_exact[4, 4] = 4*EIy/L
    K_exact[4, 8] = 6*EIy/L**2;     K_exact[4, 10] = 2*EIy/L
    K_exact[8, 2] = -12*EIy/L**3;   K_exact[8, 4] = 6*EIy/L**2
    K_exact[8, 8] = 12*EIy/L**3;    K_exact[8, 10] = 6*EIy/L**2
    K_exact[10, 2] = -6*EIy/L**2;   K_exact[10, 4] = 2*EIy/L
    K_exact[10, 8] = 6*EIy/L**2;    K_exact[10, 10] = 4*EIy/L

    # Torsion (θx DOFs: 3, 9)
    K_exact[3, 3] = GJ/L;    K_exact[3, 9] = -GJ/L
    K_exact[9, 3] = -GJ/L;   K_exact[9, 9] = GJ/L

    max_err = np.max(np.abs(K_num[:12, :12] - K_exact))
    ref = np.max(np.abs(K_exact))
    rel_err = max_err / ref if ref > 0 else max_err

    passed = rel_err < TOL

    print("  Test 6: 3D stiffness matrix (beam along X)")
    print(f"    max abs error = {max_err:.2e},  rel_err = {rel_err:.2e}  {'PASS' if passed else 'FAIL'}")
    return passed


def test_cantilever_tip_load_3d_Y():
    """Test 7: 3D cantilever — tip load in Y (bending about z, uses EIz)."""
    P = 1.0
    model = Domain(nD=3)
    nd1 = Node36(1, coord=[0.0, 0.0, 0.0], fix=[1,1,1,1,1,1])
    nd2 = Node36(2, coord=[L, 0.0, 0.0])
    model.add(nd1, nd2)

    transf = LinearCrdTransf3d(vecxz=[0.0, 0.0, 1.0])
    beam = ElasticBeamColumn3d(1, nodes=[nd1, nd2], A=A, E=E,
                                Iz=Iz, Iy=Iy, G=G, J=J, transf=transf)
    model.add(beam)

    t_const = Constant(1, factor=1.0)
    # [nodeID, fx, fy, fz, mx, my, mz]
    pattern = PlainPattern(1, tseries=t_const, load=[[2, 0.0, -P, 0.0, 0.0, 0.0, 0.0]])
    model.add(pattern)

    run_analysis(model)

    u_y = nd2._getCommitDisp()[1]
    theta_z = nd2._getCommitDisp()[5]

    EI = E * Iz
    delta_exact = -P * L**3 / (3 * EI)
    theta_exact = -P * L**2 / (2 * EI)

    err_d = abs(u_y - delta_exact) / abs(delta_exact)
    err_t = abs(theta_z - theta_exact) / abs(theta_exact)
    pass_d = err_d < TOL
    pass_t = err_t < TOL

    print("  Test 7: 3D cantilever — tip load Y (bending z)")
    print(f"    u_y:      num = {u_y:.10e},  exact = {delta_exact:.10e},  err = {err_d:.2e}  {'PASS' if pass_d else 'FAIL'}")
    print(f"    theta_z:  num = {theta_z:.10e},  exact = {theta_exact:.10e},  err = {err_t:.2e}  {'PASS' if pass_t else 'FAIL'}")
    return pass_d and pass_t


def test_cantilever_tip_load_3d_Z():
    """Test 8: 3D cantilever — tip load in Z (bending about y, uses EIy)."""
    P = 1.0
    model = Domain(nD=3)
    nd1 = Node36(1, coord=[0.0, 0.0, 0.0], fix=[1,1,1,1,1,1])
    nd2 = Node36(2, coord=[L, 0.0, 0.0])
    model.add(nd1, nd2)

    transf = LinearCrdTransf3d(vecxz=[0.0, 0.0, 1.0])
    beam = ElasticBeamColumn3d(1, nodes=[nd1, nd2], A=A, E=E,
                                Iz=Iz, Iy=Iy, G=G, J=J, transf=transf)
    model.add(beam)

    t_const = Constant(1, factor=1.0)
    pattern = PlainPattern(1, tseries=t_const, load=[[2, 0.0, 0.0, -P, 0.0, 0.0, 0.0]])
    model.add(pattern)

    run_analysis(model)

    u_z = nd2._getCommitDisp()[2]
    theta_y = nd2._getCommitDisp()[4]

    EI = E * Iy
    delta_exact = -P * L**3 / (3 * EI)
    # Sign: load in -Z causes positive rotation about Y (right-hand rule:
    # positive θy rotates +Z toward -X, which is opposite to deflection direction)
    # The basic deformation ub[3] = θy + (w_j-w_i)/L. For cantilever under -P_z:
    # θy at tip is positive (beam curves so that slope increases in +Y rotation sense)
    theta_exact = P * L**2 / (2 * EI)

    err_d = abs(u_z - delta_exact) / abs(delta_exact)
    err_t = abs(theta_y - theta_exact) / abs(theta_exact)
    pass_d = err_d < TOL
    pass_t = err_t < TOL

    print("  Test 8: 3D cantilever — tip load Z (bending y)")
    print(f"    u_z:      num = {u_z:.10e},  exact = {delta_exact:.10e},  err = {err_d:.2e}  {'PASS' if pass_d else 'FAIL'}")
    print(f"    theta_y:  num = {theta_y:.10e},  exact = {theta_exact:.10e},  err = {err_t:.2e}  {'PASS' if pass_t else 'FAIL'}")
    return pass_d and pass_t


def test_cantilever_torsion_3d():
    """Test 9: 3D cantilever — tip torsion about X."""
    T = 1.0
    model = Domain(nD=3)
    nd1 = Node36(1, coord=[0.0, 0.0, 0.0], fix=[1,1,1,1,1,1])
    nd2 = Node36(2, coord=[L, 0.0, 0.0])
    model.add(nd1, nd2)

    transf = LinearCrdTransf3d(vecxz=[0.0, 0.0, 1.0])
    beam = ElasticBeamColumn3d(1, nodes=[nd1, nd2], A=A, E=E,
                                Iz=Iz, Iy=Iy, G=G, J=J, transf=transf)
    model.add(beam)

    t_const = Constant(1, factor=1.0)
    # Apply torque about X at tip: mx = T
    pattern = PlainPattern(1, tseries=t_const, load=[[2, 0.0, 0.0, 0.0, T, 0.0, 0.0]])
    model.add(pattern)

    run_analysis(model)

    theta_x = nd2._getCommitDisp()[3]

    GJ_val = G * J
    theta_exact = T * L / GJ_val

    err = abs(theta_x - theta_exact) / abs(theta_exact)
    passed = err < TOL

    print("  Test 9: 3D cantilever — tip torsion")
    print(f"    theta_x:  num = {theta_x:.10e},  exact = {theta_exact:.10e},  err = {err:.2e}  {'PASS' if passed else 'FAIL'}")
    return passed


def test_inclined_cantilever_3d():
    """
    Test 10: 3D inclined cantilever — beam in X-Y plane at 45 deg, tip load in Z.
    This tests the 3D coordinate transformation for a non-axis-aligned beam.
    Load in Z is purely transverse (bending about local y).

    Beam from (0,0,0) to (L*cos45, L*sin45, 0).
    vecxz = [0,0,1] (Z is in the local x-z plane).
    Local z = global Z, local y = perpendicular in X-Y plane.
    Load in global Z = load in local z direction = transverse bending about local y.

    Analytical: delta_z = P*L^3/(3*EIy), theta_y_local = P*L^2/(2*EIy)
    """
    P = 1.0
    angle = np.pi / 4.0
    c45 = np.cos(angle)
    s45 = np.sin(angle)

    model = Domain(nD=3)
    nd1 = Node36(1, coord=[0.0, 0.0, 0.0], fix=[1,1,1,1,1,1])
    nd2 = Node36(2, coord=[L*c45, L*s45, 0.0])
    model.add(nd1, nd2)

    transf = LinearCrdTransf3d(vecxz=[0.0, 0.0, 1.0])
    beam = ElasticBeamColumn3d(1, nodes=[nd1, nd2], A=A, E=E,
                                Iz=Iz, Iy=Iy, G=G, J=J, transf=transf)
    model.add(beam)

    t_const = Constant(1, factor=1.0)
    pattern = PlainPattern(1, tseries=t_const, load=[[2, 0.0, 0.0, -P, 0.0, 0.0, 0.0]])
    model.add(pattern)

    run_analysis(model)

    u_z = nd2._getCommitDisp()[2]

    # Load in -Z is along local -z (since local z = global Z for this orientation).
    # This causes bending about local y-axis.
    EI = E * Iy
    delta_z_exact = -P * L**3 / (3 * EI)

    err = abs(u_z - delta_z_exact) / abs(delta_z_exact)
    passed = err < TOL

    print("  Test 10: 3D inclined cantilever (45 deg in XY) — load in Z")
    print(f"    u_z:      num = {u_z:.10e},  exact = {delta_z_exact:.10e},  err = {err:.2e}  {'PASS' if passed else 'FAIL'}")
    return passed


if __name__ == "__main__":
    print("\n========== ELASTIC BEAM-COLUMN BENCHMARK ==========")
    print(f"  E={E}, A={A}, Iz={Iz}, Iy={Iy}, G={G}, J={J}, L={L}")
    print(f"  Tolerance: rel error < {TOL:.0e}")
    print()

    results = []

    print("  --- 2D Tests ---")
    results.append(test_stiffness_matrix_2d())
    print()
    results.append(test_cantilever_tip_load_2d())
    print()
    results.append(test_cantilever_tip_moment_2d())
    print()
    results.append(test_simply_supported_center_load_2d())
    print()
    results.append(test_inclined_cantilever_2d())
    print()

    print("  --- 3D Tests ---")
    results.append(test_stiffness_matrix_3d())
    print()
    results.append(test_cantilever_tip_load_3d_Y())
    print()
    results.append(test_cantilever_tip_load_3d_Z())
    print()
    results.append(test_cantilever_torsion_3d())
    print()
    results.append(test_inclined_cantilever_3d())

    print()
    print(f"  {sum(results)}/{len(results)} PASS")
    print("=====================================================\n")
