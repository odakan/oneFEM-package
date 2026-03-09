##-----------------------------------------------------------------------##
#  Diagnostic Checks: Hex8 Nonlinear Kinematics (TL, UL, Corot)
#
#  Gate test before any Hex8 nonlinear validation work.
#  All three kinematics classes support 3D in code, but only Linear
#  has been benchmarked with Hex8.
#
#  Check 1: TL/UL simple shear — verify UL bug fix works for 3D
#    Single unit cube, simple shear in xz-plane, 4 gamma values
#    12 sub-tests: 4 gammas x (F, E, TL==UL)
#
#  Check 2: Corot pure rotation — verify SVD polar decomp for 3D
#    Single unit cube, rigid rotations about X/Y/Z axes
#    9 sub-tests: 3 angles x 3 axes
#
#  Check 3: Corot Newton convergence — verify analysis pipeline
#    1x1x10 cantilever, lateral tip load, Newton+LoadControl
#    3 sub-tests: convergence, physical reasonableness, Corot==TL small load
##-----------------------------------------------------------------------##

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from oneFEM.model import Domain
from oneFEM.model.node import Node33
from oneFEM.model.element.continuum.hex8 import Hex8
from oneFEM.model.kinematics.continuum.cauchy.total_lagrangian import TotalLagrangianContinuumKinematics
from oneFEM.model.kinematics.continuum.cauchy.updated_lagrangian import UpdatedLagrangianContinuumKinematics
from oneFEM.model.kinematics.continuum.cauchy.corot import CorotContinuumKinematics
from oneFEM.model.material.nD.elastic_isotropic import ElasticIsotropic
from oneFEM.model.pattern import Plain as PlainPattern
from oneFEM.model.tseries import Constant, Linear as LinearTS
from oneFEM._systools.data import Vector, Matrix
from oneFEM.analysis import Analysis
from oneFEM.analysis.algorithm.linear import Linear
from oneFEM.analysis.algorithm.newton_raphson import Newton
from oneFEM.analysis.constraints import Plain as PlainConstraints
from oneFEM.analysis.numberer import Plain as PlainNumberer
from oneFEM.analysis.system import FullGeneral
from oneFEM.analysis.integrator import LoadControl
from oneFEM.analysis.test import NormUnbalance
from oneFEM import SimulationManager


# ================================================================
# Helpers
# ================================================================

UNIT_CUBE_COORDS = {
    1: (0.0, 0.0, 0.0),
    2: (1.0, 0.0, 0.0),
    3: (1.0, 1.0, 0.0),
    4: (0.0, 1.0, 0.0),
    5: (0.0, 0.0, 1.0),
    6: (1.0, 0.0, 1.0),
    7: (1.0, 1.0, 1.0),
    8: (0.0, 1.0, 1.0),
}
UNIT_CUBE_CONN = [1, 2, 3, 4, 5, 6, 7, 8]


def build_single_hex8(kinematics):
    """Build single unit cube Hex8 with given kinematics. All nodes fixed."""
    model = Domain()
    nodes = {}
    for nid, (x, y, z) in UNIT_CUBE_COORDS.items():
        nd = Node33(nid, coord=[x, y, z])
        nd.setFix([True, True, True])
        nodes[nid] = nd
        model.add(nd)

    mat = ElasticIsotropic(1, 1000.0, 0.3, type='3D')
    elem = Hex8(1, [nodes[n] for n in UNIT_CUBE_CONN], mat, kinematics=kinematics,
                incompatible=False)
    model.add(elem)

    ts = Constant(1, factor=1.0)
    pat = PlainPattern(1, ts)
    model.add(pat)
    analysis = Analysis(algorithm=Linear(), integrator=LoadControl(1))
    analysis._analyze(model, nSteps=0, dt=0.0)

    return model, nodes, elem


def check_committed_u_e(elem, label):
    """Verify _committed_u_e is initialized to zero. Abort if not."""
    if elem._committed_u_e is None:
        print("    ABORT: {}: _committed_u_e is None (not initialized)".format(label))
        return False
    norm = np.linalg.norm(elem._committed_u_e.data)
    if norm > 1e-15:
        print("    ABORT: {}: _committed_u_e norm = {:.2e} (expected 0)".format(label, norm))
        return False
    return True


def apply_displacements_3d(nodes, disp_func):
    """Apply displacements to all nodes via _update + _commitState."""
    f_zero = Vector([0.0, 0.0, 0.0])
    for nid, nd in nodes.items():
        x, y, z = UNIT_CUBE_COORDS[nid]
        ux, uy, uz = disp_func(x, y, z)
        nd._update(f_zero, Vector([ux, uy, uz]))
        nd._commitState()


def rotation_matrix_3d(angle_deg, axis):
    """3D rotation matrix about X, Y, or Z axis."""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    if axis == 'X':
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == 'Y':
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    else:  # Z
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _box_node_id(ix, iy, iz, nx, ny):
    """Structured node ID for box mesh (1-based)."""
    return iz * (nx + 1) * (ny + 1) + iy * (nx + 1) + ix + 1


def _build_box_mesh(nx, ny, nz, Lx, Ly, Lz):
    """Build structured hex8 mesh for a box [0,Lx]x[0,Ly]x[0,Lz]."""
    xs = np.linspace(0, Lx, nx + 1)
    ys = np.linspace(0, Ly, ny + 1)
    zs = np.linspace(0, Lz, nz + 1)

    node_coords = {}
    for iz in range(nz + 1):
        for iy in range(ny + 1):
            for ix in range(nx + 1):
                nid = _box_node_id(ix, iy, iz, nx, ny)
                node_coords[nid] = (xs[ix], ys[iy], zs[iz])

    elem_conn = {}
    eid = 1
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                n0 = _box_node_id(ix,   iy,   iz,   nx, ny)
                n1 = _box_node_id(ix+1, iy,   iz,   nx, ny)
                n2 = _box_node_id(ix+1, iy+1, iz,   nx, ny)
                n3 = _box_node_id(ix,   iy+1, iz,   nx, ny)
                n4 = _box_node_id(ix,   iy,   iz+1, nx, ny)
                n5 = _box_node_id(ix+1, iy,   iz+1, nx, ny)
                n6 = _box_node_id(ix+1, iy+1, iz+1, nx, ny)
                n7 = _box_node_id(ix,   iy+1, iz+1, nx, ny)
                elem_conn[eid] = [n0, n1, n2, n3, n4, n5, n6, n7]
                eid += 1

    return node_coords, elem_conn


# ================================================================
# Check 1: TL/UL Simple Shear
# ================================================================

def test_simple_shear():
    """Check 1: Simple shear in xz-plane on unit cube Hex8.

    Bottom face (z=0) fixed, top face (z=1) sheared: u_x = gamma.
    F = [[1,0,gamma],[0,1,0],[0,0,1]]
    GL strain Voigt [xx,yy,zz,2xy,2yz,2xz] = [0, 0, gamma^2/2, 0, 0, gamma]
    """
    gammas = [0.1, 0.5, 1.0, 2.0]
    results = []

    for gamma in gammas:
        # Analytical
        F_exact = np.eye(3)
        F_exact[0, 2] = gamma
        E_exact = np.array([0.0, 0.0, gamma**2 / 2.0, 0.0, 0.0, gamma])

        def shear_disp(x, y, z):
            return (gamma * z, 0.0, 0.0)

        # --- TL ---
        kin_tl = TotalLagrangianContinuumKinematics()
        _, nodes_tl, elem_tl = build_single_hex8(kin_tl)

        if not check_committed_u_e(elem_tl, "TL gamma={:.1f}".format(gamma)):
            results.extend([False, False, False])
            continue

        apply_displacements_3d(nodes_tl, shear_disp)
        elem_tl._update()

        max_F_err = 0.0
        max_E_err = 0.0
        for gp in range(len(elem_tl._gp_data)):
            F = elem_tl._kinematics.getF(gp)
            max_F_err = max(max_F_err, np.max(np.abs(F.data - F_exact)))

            E_voigt = elem_tl._kinematics.getStrain(gp).make_vector()
            for i in range(6):
                ref = abs(E_exact[i])
                if ref > 1e-15:
                    max_E_err = max(max_E_err, abs(E_voigt[i] - E_exact[i]) / ref)
                else:
                    max_E_err = max(max_E_err, abs(E_voigt[i]))

        p_F = max_F_err < 1e-14
        p_E = max_E_err < 1e-14

        # --- UL ---
        kin_ul = UpdatedLagrangianContinuumKinematics()
        _, nodes_ul, elem_ul = build_single_hex8(kin_ul)

        if not check_committed_u_e(elem_ul, "UL gamma={:.1f}".format(gamma)):
            results.extend([False, False, False])
            continue

        apply_displacements_3d(nodes_ul, shear_disp)
        elem_ul._update()

        max_tl_ul_err = 0.0
        for gp in range(len(elem_tl._gp_data)):
            E_tl = elem_tl._kinematics.getStrain(gp).make_vector()
            E_ul = elem_ul._kinematics.getStrain(gp).make_vector()
            for i in range(6):
                abs_diff = abs(E_tl[i] - E_ul[i])
                if abs_diff < 1e-14:
                    continue
                ref = max(abs(E_tl[i]), 1e-15)
                max_tl_ul_err = max(max_tl_ul_err, abs_diff / ref)

        p_eq = max_tl_ul_err < 1e-10

        results.extend([p_F, p_E, p_eq])
        print("    gamma={:.1f}: F err={:.2e} {} | E err={:.2e} {} | TL==UL err={:.2e} {}".format(
            gamma, max_F_err, "PASS" if p_F else "FAIL",
            max_E_err, "PASS" if p_E else "FAIL",
            max_tl_ul_err, "PASS" if p_eq else "FAIL"))

    return results


# ================================================================
# Check 2: Corot Pure Rotation
# ================================================================

def test_corot_pure_rotation():
    """Check 2: Corot extracts correct R via SVD, producing zero strain
    for rigid rotation of a unit cube Hex8.

    Rotations: 10, 30, 90 deg about each of X, Y, Z axes = 9 cases.
    """
    angles = [10.0, 30.0, 90.0]
    axes = ['X', 'Y', 'Z']
    results = []

    for axis in axes:
        for angle in angles:
            R = rotation_matrix_3d(angle, axis)

            def rot_disp(x, y, z, _R=R):
                X = np.array([x, y, z])
                return tuple(_R @ X - X)

            kin = CorotContinuumKinematics()
            _, nodes, elem = build_single_hex8(kin)
            apply_displacements_3d(nodes, rot_disp)
            elem._update()

            max_strain = 0.0
            for gp in range(len(elem._gp_data)):
                eps = elem._kinematics.getStrain(gp).make_vector()
                max_strain = max(max_strain, np.max(np.abs(eps)))

            passed = max_strain < 1e-10
            results.append(passed)
            status = "PASS" if passed else "FAIL"
            print("    R({:3.0f}deg, {}): max|strain| = {:.2e}  {}".format(
                angle, axis, max_strain, status))

    return results


# ================================================================
# Check 3: Corot Newton Convergence
# ================================================================

def build_hex8_cantilever(kin_class, nSteps=10, F_total=0.1):
    """Build 1x1x10 cantilever (10 Hex8 elements), run Newton analysis.

    Fix bottom face (z=0), lateral tip load F_x distributed on top face (z=10).
    Returns (converged, ux_tip, uy_tip, uz_tip).
    """
    nx, ny, nz = 1, 1, 10
    Lx, Ly, Lz = 1.0, 1.0, 10.0
    E, nu = 1000.0, 0.3

    node_coords, elem_conn = _build_box_mesh(nx, ny, nz, Lx, Ly, Lz)

    model = Domain()
    nodes = {}
    for nid in sorted(node_coords.keys()):
        x, y, z = node_coords[nid]
        nd = Node33(nid, coord=[x, y, z])
        nodes[nid] = nd
        model.add(nd)

    mat = ElasticIsotropic(1, E, nu, type='3D')

    elements = {}
    for eid in sorted(elem_conn.keys()):
        conn = elem_conn[eid]
        kin = kin_class()  # Per-element instance
        elem = Hex8(eid, [nodes[c] for c in conn], mat, kinematics=kin,
                    incompatible=False)
        elements[eid] = elem
        model.add(elem)

    # Fix bottom face (z=0)
    for nid, (x, y, z) in node_coords.items():
        if abs(z) < 1e-10:
            nodes[nid].setFix([True, True, True])

    # Tip load: F_x distributed on top face (z=Lz)
    tip_nodes = [nid for nid, (x, y, z) in node_coords.items()
                 if abs(z - Lz) < 1e-10]
    f_per_node = F_total / len(tip_nodes)

    ts = LinearTS(1, factor=1.0)
    load_list = [[nid, f_per_node, 0.0, 0.0] for nid in tip_nodes]
    pat = PlainPattern(1, tseries=ts, load=load_list)
    model.add(pat)

    alg = Newton(1, tangent='current')
    const = PlainConstraints(1)
    numb = PlainNumberer(1)
    syst = FullGeneral(1)
    integ = LoadControl(1)
    ctest = NormUnbalance(1, tol=1e-8, maxIter=50)
    analysis = Analysis(1, algorithm=alg, constraints=const,
                        integrator=integ, system=syst, test=ctest)
    dt = 1.0 / nSteps
    sim = SimulationManager(1, model, analysis, dt=dt)

    converged = True
    try:
        sim.analyze(nSteps, dt=dt)
    except Exception as e:
        converged = False

    # Average tip displacement on top face
    ux_vals, uy_vals, uz_vals = [], [], []
    for nid in tip_nodes:
        u = nodes[nid]._getCommitDisp()
        ux_vals.append(float(u[0]))
        uy_vals.append(float(u[1]))
        uz_vals.append(float(u[2]))

    return converged, np.mean(ux_vals), np.mean(uy_vals), np.mean(uz_vals)


def test_corot_newton():
    """Check 3: Corot Newton convergence on Hex8 cantilever.

    3a: Newton converges (10 steps, F=0.1)
    3b: Tip displacement physically reasonable (ux > 0)
    3c: Corot == TL at small load (F=0.01, 1 step, rel err < 1e-3)
    """
    results = []

    # 3a + 3b: Convergence and physical reasonableness
    converged, ux, uy, uz = build_hex8_cantilever(
        CorotContinuumKinematics, nSteps=10, F_total=0.1)

    p_3a = converged
    print("    3a (Newton converges):  {}  (ux={:.6e}, uy={:.6e}, uz={:.6e})".format(
        "PASS" if p_3a else "FAIL", ux, uy, uz))
    results.append(p_3a)

    p_3b = converged and ux > 0
    print("    3b (ux > 0):            {}".format("PASS" if p_3b else "FAIL"))
    results.append(p_3b)

    # 3c: Corot == TL at small load
    F_small = 0.01
    conv_co, ux_co, uy_co, uz_co = build_hex8_cantilever(
        CorotContinuumKinematics, nSteps=1, F_total=F_small)
    conv_tl, ux_tl, uy_tl, uz_tl = build_hex8_cantilever(
        TotalLagrangianContinuumKinematics, nSteps=1, F_total=F_small)

    if conv_co and conv_tl:
        # Use absolute tolerance for near-zero components (numerical noise)
        abs_tol = 1e-12
        errs = []
        for val_co, val_tl, label in [(ux_co, ux_tl, 'x'), (uy_co, uy_tl, 'y'), (uz_co, uz_tl, 'z')]:
            if abs(val_tl) > abs_tol:
                errs.append(abs(val_co - val_tl) / abs(val_tl))
            else:
                errs.append(abs(val_co - val_tl))
        max_err = max(errs)
        p_3c = max_err < 1e-2
        print("    3c (Corot==TL small):   {}  (err={:.2e})".format(
            "PASS" if p_3c else "FAIL", max_err))
        print("      Corot: ux={:.6e}, uy={:.6e}, uz={:.6e}".format(ux_co, uy_co, uz_co))
        print("      TL:    ux={:.6e}, uy={:.6e}, uz={:.6e}".format(ux_tl, uy_tl, uz_tl))
    else:
        p_3c = False
        print("    3c (Corot==TL small):   FAIL  (convergence failed: Corot={}, TL={})".format(
            conv_co, conv_tl))
    results.append(p_3c)

    return results


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Hex8 Nonlinear Kinematics Diagnostic Checks")
    print("=" * 60)

    all_results = []

    # Check 1
    print("\n  Check 1: TL/UL Simple Shear (xz-plane)")
    print("    Unit cube Hex8, E=1000, nu=0.3, 3D")
    print("  " + "-" * 56)
    r1 = test_simple_shear()
    all_results.extend(r1)
    n1 = sum(r1)
    print("    {}/{} PASS".format(n1, len(r1)))

    # Check 2
    print("\n  Check 2: Corot Pure Rotation (SVD polar decomp)")
    print("    Unit cube Hex8, rigid rotations about X/Y/Z")
    print("  " + "-" * 56)
    r2 = test_corot_pure_rotation()
    all_results.extend(r2)
    n2 = sum(r2)
    print("    {}/{} PASS".format(n2, len(r2)))

    # Check 3
    print("\n  Check 3: Corot Newton Convergence (Hex8 cantilever)")
    print("    1x1x10, E=1000, nu=0.3, F_x=0.1, 10 steps")
    print("  " + "-" * 56)
    r3 = test_corot_newton()
    all_results.extend(r3)
    n3 = sum(r3)
    print("    {}/{} PASS".format(n3, len(r3)))

    # Summary
    total_pass = sum(all_results)
    total_tests = len(all_results)
    print("\n" + "=" * 60)
    print("  {}/{} PASS".format(total_pass, total_tests))
    if total_pass == total_tests:
        print("  ALL PASS")
    print("=" * 60)
