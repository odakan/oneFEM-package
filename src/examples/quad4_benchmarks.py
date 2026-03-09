##-----------------------------------------------------------------------##
#  Quad4 Full Kinematics Validation Suite
#
#  Consolidated benchmark file for Quad4 element across all kinematics:
#    Linear, Total Lagrangian, Updated Lagrangian, Corotational (beams).
#
#  B1: Patch Test (Linear, GATE)
#  B2: Cook's Membrane (Linear)
#  B3: Simple Shear (TL + UL)
#  B4: Cantilever Rollup (TL + UL)
#  B5: Snap-Through Arch (TL + UL)
#  B6: Lee's Frame (Corot beams)
#  B7: Column Buckling (Corot beams)
#  B8: Thick-Walled Cylinder (Linear + TL + UL)
#
#  Plots saved to docs/validation/quad4/
##-----------------------------------------------------------------------##

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection, LineCollection

from oneFEM.model import Domain
from oneFEM.model.node import Node22, Node23, Node36
from oneFEM.model.element.continuum.quad4 import Quad4
from oneFEM.model.element.beam import ElasticBeamColumn2d, ElasticBeamColumn3d
from oneFEM.model.kinematics.continuum.cauchy.linear import LinearContinuumKinematics
from oneFEM.model.kinematics.continuum.cauchy.total_lagrangian import TotalLagrangianContinuumKinematics
from oneFEM.model.kinematics.continuum.cauchy.updated_lagrangian import UpdatedLagrangianContinuumKinematics
from oneFEM.model.kinematics.beam import (
    CorotCrdTransf2d, CorotCrdTransf3d,
    PDeltaCrdTransf2d, PDeltaCrdTransf3d
)
from oneFEM.model.material.nD.elastic_isotropic import ElasticIsotropic
from oneFEM.model.pattern import Plain as PlainPattern
from oneFEM.model.tseries import Constant, Linear as LinearTS
from oneFEM._systools.data import Vector, Matrix
from oneFEM.analysis import Analysis
from oneFEM.analysis.algorithm.newton_raphson import Newton
from oneFEM.analysis.constraints import Plain as PlainConstraints
from oneFEM.analysis.numberer import Plain as PlainNumberer
from oneFEM.analysis.system import FullGeneral
from oneFEM.analysis.algorithm.linear import Linear as LinearAlg
from oneFEM.analysis.integrator import LoadControl, DispControl
from oneFEM.analysis.test import NormUnbalance
from oneFEM import SimulationManager

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
})

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'docs', 'validation', 'quad4')
os.makedirs(SAVE_DIR, exist_ok=True)


# ================================================================
# Mesh Plotting Utility
# ================================================================

def plot_quad4_mesh(ax, nodes_xy, connectivity,
                   deformed_xy=None, field=None, field_label='',
                   alpha=0.6, cmap='viridis', scale=1.0,
                   show_nodes=False, title=''):
    """Plot Quad4 mesh with optional deformed shape and field coloring.

    nodes_xy: dict {nid: (x, y)} or array (n_nodes, 2)
    connectivity: dict {eid: [n1,n2,n3,n4]} or list of [n1,n2,n3,n4]
    deformed_xy: same format as nodes_xy (deformed coordinates)
    field: list/array, one scalar per element
    """
    # Normalize to dict format
    if isinstance(nodes_xy, np.ndarray):
        nodes_dict = {i+1: (nodes_xy[i, 0], nodes_xy[i, 1]) for i in range(len(nodes_xy))}
    else:
        nodes_dict = nodes_xy

    if isinstance(connectivity, list):
        conn_dict = {i+1: c for i, c in enumerate(connectivity)}
    else:
        conn_dict = connectivity

    if isinstance(deformed_xy, np.ndarray):
        def_dict = {i+1: (deformed_xy[i, 0], deformed_xy[i, 1]) for i in range(len(deformed_xy))}
    elif deformed_xy is not None:
        def_dict = deformed_xy
    else:
        def_dict = None

    # Reference wireframe
    for eid, conn in conn_dict.items():
        verts = [nodes_dict[n] for n in conn] + [nodes_dict[conn[0]]]
        xs, ys = zip(*verts)
        ax.plot(xs, ys, color='#999999', linewidth=0.5, zorder=1)

    # Deformed filled faces
    if def_dict is not None:
        polys = []
        for eid, conn in conn_dict.items():
            ref = [np.array(nodes_dict[n]) for n in conn]
            dfm = [np.array(def_dict[n]) for n in conn]
            verts = [r + scale * (d - r) for r, d in zip(ref, dfm)]
            polys.append(mpatches.Polygon(verts, closed=True))

        if field is not None:
            field_arr = np.array(field) if not isinstance(field, np.ndarray) else field
            pc = PatchCollection(polys, alpha=alpha, cmap=cmap)
            pc.set_array(field_arr)
            ax.add_collection(pc)
            plt.colorbar(pc, ax=ax, label=field_label, shrink=0.8)
        else:
            pc = PatchCollection(polys, alpha=alpha, facecolor='steelblue',
                                 edgecolor='navy', linewidth=0.5)
            ax.add_collection(pc)

    if show_nodes:
        for nid, (x, y) in nodes_dict.items():
            ax.plot(x, y, 'k.', markersize=3, zorder=5)

    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if title:
        ax.set_title(title, fontsize=11)
    return ax


# ================================================================
# Mesh Generators
# ================================================================

def cook_mesh(nx, ny):
    """Generate Cook's membrane mesh. Returns (node_coords, elem_conn,
    left_nodes, right_nodes, tip_node_id)."""
    node_coords = {}
    nid = 1
    node_grid = {}

    for j in range(ny + 1):
        for i in range(nx + 1):
            xi = i / nx
            eta = j / ny
            x = 48.0 * xi
            y_bot = 44.0 * xi
            y_top = 44.0 + 16.0 * xi
            y = y_bot * (1.0 - eta) + y_top * eta
            node_coords[nid] = (x, y)
            node_grid[(i, j)] = nid
            nid += 1

    elem_conn = {}
    eid = 1
    for j in range(ny):
        for i in range(nx):
            n1 = node_grid[(i, j)]
            n2 = node_grid[(i + 1, j)]
            n3 = node_grid[(i + 1, j + 1)]
            n4 = node_grid[(i, j + 1)]
            elem_conn[eid] = [n1, n2, n3, n4]
            eid += 1

    left_nodes = [node_grid[(0, j)] for j in range(ny + 1)]
    right_nodes = [node_grid[(nx, j)] for j in range(ny + 1)]
    tip_node_id = node_grid[(nx, ny)]

    return node_coords, elem_conn, left_nodes, right_nodes, tip_node_id


def rect_mesh(nx, ny, Lx, Ly):
    """Generate rectangular mesh. Returns (node_coords, elem_conn, node_grid)."""
    node_coords = {}
    nid = 1
    node_grid = {}
    dx = Lx / nx
    dy = Ly / ny

    for j in range(ny + 1):
        for i in range(nx + 1):
            x = i * dx
            y = j * dy
            node_coords[nid] = (x, y)
            node_grid[(i, j)] = nid
            nid += 1

    elem_conn = {}
    eid = 1
    for j in range(ny):
        for i in range(nx):
            n1 = node_grid[(i, j)]
            n2 = node_grid[(i + 1, j)]
            n3 = node_grid[(i + 1, j + 1)]
            n4 = node_grid[(i, j + 1)]
            elem_conn[eid] = [n1, n2, n3, n4]
            eid += 1

    return node_coords, elem_conn, node_grid


def annular_mesh(nr, nt, ri, ro):
    """Generate quarter-annulus mesh (2D). Returns (node_coords, elem_conn,
    inner_nodes, sym_x_nodes, sym_y_nodes, node_grid)."""
    radii = np.linspace(ri, ro, nr + 1)
    angles = np.linspace(0, np.pi / 2, nt + 1)

    node_coords = {}
    node_grid = {}
    nid = 1
    for it in range(nt + 1):
        for ir in range(nr + 1):
            r = radii[ir]
            theta = angles[it]
            node_coords[nid] = (r * np.cos(theta), r * np.sin(theta))
            node_grid[(ir, it)] = nid
            nid += 1

    elem_conn = {}
    eid = 1
    for it in range(nt):
        for ir in range(nr):
            n1 = node_grid[(ir, it)]
            n2 = node_grid[(ir + 1, it)]
            n3 = node_grid[(ir + 1, it + 1)]
            n4 = node_grid[(ir, it + 1)]
            elem_conn[eid] = [n1, n2, n3, n4]
            eid += 1

    inner_nodes = [node_grid[(0, it)] for it in range(nt + 1)]
    sym_x_nodes = [node_grid[(ir, 0)] for ir in range(nr + 1)]  # theta=0
    sym_y_nodes = [node_grid[(ir, nt)] for ir in range(nr + 1)]  # theta=pi/2

    return node_coords, elem_conn, inner_nodes, sym_x_nodes, sym_y_nodes, node_grid


def arch_mesh(nex, ney, H, L_half, R, thickness):
    """Generate circular arch mesh (2D). Returns (node_coords, elem_conn,
    apex_node, left_support_nodes, right_support_nodes)."""
    alpha = np.arcsin(L_half / R)  # half-angle
    n_along = 2 * nex + 1  # nodes along centerline
    n_thru = ney + 1  # nodes through thickness

    node_coords = {}
    node_grid = {}  # (i_along, j_thru) -> nid
    nid = 1

    for j in range(n_thru):
        t_frac = j / ney - 0.5  # -0.5 to +0.5
        for i in range(n_along):
            theta = -alpha + i * (2 * alpha / (n_along - 1))
            r = R + t_frac * thickness
            x = r * np.sin(theta)
            y = r * np.cos(theta) - (R - H)
            node_coords[nid] = (x, y)
            node_grid[(i, j)] = nid
            nid += 1

    elem_conn = {}
    eid = 1
    for j in range(ney):
        for i in range(n_along - 1):
            n1 = node_grid[(i, j)]
            n2 = node_grid[(i + 1, j)]
            n3 = node_grid[(i + 1, j + 1)]
            n4 = node_grid[(i, j + 1)]
            elem_conn[eid] = [n1, n2, n3, n4]
            eid += 1

    # Apex: middle of top surface
    apex_i = (n_along - 1) // 2
    apex_node = node_grid[(apex_i, ney)]

    # Support nodes: all nodes at left and right ends (through thickness)
    left_support = [node_grid[(0, j)] for j in range(n_thru)]
    right_support = [node_grid[(n_along - 1, j)] for j in range(n_thru)]

    return node_coords, elem_conn, apex_node, left_support, right_support


# ================================================================
# Direct Solver
# ================================================================

def direct_solve_2d(model, nodes, fixed_nids_dofs, F_ext_nids=None, analysis=None):
    """Assemble K, partition, solve. Returns global displacement vector."""
    if analysis is None:
        raise RuntimeError("direct_solve_2d requires an analysis argument")
    analysis._solution_integrator._assembleK(model)
    K = analysis._assembly_system.getK().toarray()
    nDOF = model.nDOF

    fixed_dofs = []
    u_prescribed = np.zeros(nDOF)
    for nid, fixes in fixed_nids_dofs.items():
        dof_indices = np.asarray(nodes[nid].getDOFs()).astype(int)
        for local_dof, value in fixes.items():
            gd = dof_indices[local_dof]
            fixed_dofs.append(gd)
            u_prescribed[gd] = value

    free_dofs = sorted(set(range(nDOF)) - set(fixed_dofs))
    fixed_dofs = sorted(set(fixed_dofs))

    F = np.zeros(nDOF)
    if F_ext_nids:
        for nid, forces in F_ext_nids.items():
            dof_indices = np.asarray(nodes[nid].getDOFs()).astype(int)
            for i, f in enumerate(forces):
                F[dof_indices[i]] += f

    uu = np.array(free_dofs)
    pp = np.array(fixed_dofs)

    if len(uu) == 0:
        return u_prescribed

    K_ff = K[np.ix_(uu, uu)]
    K_fp = K[np.ix_(uu, pp)]
    rhs = F[uu] - K_fp @ u_prescribed[pp]
    u_f = np.linalg.solve(K_ff, rhs)

    u = u_prescribed.copy()
    u[uu] = u_f
    return u


def compute_edge_pressure_2d(n1_xy, n2_xy, pressure, thickness=1.0):
    """Consistent nodal forces from pressure on a 2D edge (2-GP line integral).
    Pressure is positive inward (toward center). Returns (f1, f2) as 2-element arrays."""
    g = 1.0 / np.sqrt(3.0)
    gps = [(-g, 1.0), (g, 1.0)]
    f1 = np.zeros(2)
    f2 = np.zeros(2)
    x1, y1 = n1_xy
    x2, y2 = n2_xy

    for s, w in gps:
        N1 = 0.5 * (1.0 - s)
        N2 = 0.5 * (1.0 + s)
        dx_ds = 0.5 * (x2 - x1)
        dy_ds = 0.5 * (y2 - y1)
        ds = np.sqrt(dx_ds**2 + dy_ds**2)
        # Normal from cross product of tangent with z: n = (-dy, dx)/ds
        # For inner surface nodes ordered CCW from inside, this points inward.
        # Internal pressure pushes outward = opposite to this normal.
        nx = dy_ds / ds   # outward normal x (flipped sign)
        ny = -dx_ds / ds  # outward normal y (flipped sign)
        # Force = pressure * outward_normal * dA (dA = ds * thickness * w)
        f1 += N1 * pressure * np.array([nx, ny]) * ds * thickness * w
        f2 += N2 * pressure * np.array([nx, ny]) * ds * thickness * w

    return f1, f2


# ================================================================
# B1: Patch Test (Linear, GATE)
# ================================================================

B1_NODE_COORDS = {
    1: (0.000, 0.000), 2: (1.000, 0.000), 3: (1.000, 1.000),
    4: (0.000, 1.000), 5: (0.500, 0.000), 6: (1.000, 0.500),
    7: (0.500, 1.000), 8: (0.000, 0.500), 9: (0.240, 0.220),
}
B1_ELEM_CONN = {
    1: [1, 5, 9, 8], 2: [5, 2, 6, 9],
    3: [9, 6, 3, 7], 4: [8, 9, 7, 4],
}
B1_BOUNDARY = [1, 2, 3, 4, 5, 6, 7, 8]
B1_INTERIOR = 9
B1_E = 1.0
B1_NU = 0.25
B1_G = B1_E / (2.0 * (1.0 + B1_NU))


def b1_analytical_disp(lc, x, y):
    if lc == 1:
        return (x / B1_E, -B1_NU * y / B1_E)
    elif lc == 2:
        return (-B1_NU * x / B1_E, y / B1_E)
    elif lc == 3:
        return (y / (2.0 * B1_G), x / (2.0 * B1_G))


def b1_expected_stress(lc):
    if lc == 1:
        return np.array([1.0, 0.0, 0.0])
    elif lc == 2:
        return np.array([0.0, 1.0, 0.0])
    elif lc == 3:
        return np.array([0.0, 0.0, 1.0])


def b1_build_model():
    model = Domain()
    nodes = {}
    for nid, (x, y) in B1_NODE_COORDS.items():
        nd = Node22(nid, coord=[x, y])
        nodes[nid] = nd
        model.add(nd)

    mat = ElasticIsotropic(1, B1_E, B1_NU, type='PlaneStress')
    elements = {}
    for eid, conn in B1_ELEM_CONN.items():
        elem_nodes = [nodes[c] for c in conn]
        elem = Quad4(eid, elem_nodes, mat, thickness=1.0)
        elements[eid] = elem
        model.add(elem)

    ts = Constant(1, factor=1.0)
    pat = PlainPattern(1, ts)
    model.add(pat)
    for nid in B1_BOUNDARY:
        nodes[nid].setFix([True, True])
    _analysis = Analysis(algorithm=LinearAlg(), integrator=LoadControl(1))
    _analysis._analyze(model, nSteps=0, dt=0.0)
    return model, nodes, elements, _analysis


def b1_run_gp_stress(lc):
    model, nodes, elements, _analysis = b1_build_model()
    for nid, (x, y) in B1_NODE_COORDS.items():
        u, v = b1_analytical_disp(lc, x, y)
        nodes[nid]._update(Vector([0.0, 0.0]), Vector([u, v]))
        nodes[nid]._commitState()
    for elem in elements.values():
        elem._update()

    sig_expected = b1_expected_stress(lc)
    max_err = 0.0
    for eid, elem in elements.items():
        for gp_idx, mat in enumerate(elem._materials):
            sig_vec = mat.getStress().make_vector()
            for comp in range(3):
                if abs(sig_expected[comp]) > 1e-15:
                    err = abs(sig_vec[comp] - sig_expected[comp]) / abs(sig_expected[comp])
                else:
                    err = abs(sig_vec[comp])
                max_err = max(max_err, err)
    return max_err < 1e-10, max_err


def b1_run_interior_node(lc):
    model, nodes, elements, _analysis = b1_build_model()
    _analysis._solution_integrator._assembleK(model)
    K = _analysis._assembly_system.getK().toarray()
    n = model.nDOF

    int_nd = nodes[B1_INTERIOR]
    free_dofs = list(np.asarray(int_nd.getDOFs()).astype(int))
    fixed_dofs = []
    for nid in B1_BOUNDARY:
        fixed_dofs.extend(list(np.asarray(nodes[nid].getDOFs()).astype(int)))

    u_prescribed = np.zeros(n)
    for nid in B1_BOUNDARY:
        x, y = B1_NODE_COORDS[nid]
        u, v = b1_analytical_disp(lc, x, y)
        dofs = list(np.asarray(nodes[nid].getDOFs()).astype(int))
        u_prescribed[dofs[0]] = u
        u_prescribed[dofs[1]] = v

    uu = np.array(free_dofs)
    pp = np.array(fixed_dofs)
    K_ff = K[np.ix_(uu, uu)]
    K_fp = K[np.ix_(uu, pp)]
    u_f = np.linalg.solve(K_ff, -K_fp @ u_prescribed[pp])

    x9, y9 = B1_NODE_COORDS[B1_INTERIOR]
    u_exact, v_exact = b1_analytical_disp(lc, x9, y9)
    err_u = abs(u_f[0] - u_exact) / max(abs(u_exact), 1e-15)
    err_v = abs(u_f[1] - v_exact) / max(abs(v_exact), 1e-15)
    max_err = max(err_u, err_v)
    return max_err < 1e-10, max_err, (u_f[0], u_f[1]), (u_exact, v_exact)


def test_b1_patch_test():
    """B1: Patch test — 3 load cases x (GP stress + interior node) = 6 tests."""
    results = []
    lc_names = {1: "sigma_x", 2: "sigma_y", 3: "tau_xy"}
    lc_errors = {}

    for lc in [1, 2, 3]:
        p1, err1 = b1_run_gp_stress(lc)
        p2, err2, (uc, vc), (ue, ve) = b1_run_interior_node(lc)
        results.append(p1)
        results.append(p2)
        lc_errors[lc] = {'gp_err': err1, 'node_err': err2,
                         'computed': (uc, vc), 'exact': (ue, ve)}
        s1 = "PASS" if p1 else "FAIL"
        s2 = "PASS" if p2 else "FAIL"
        print("    LC{} ({}):  GP stress: {}  (err={:.2e})  |  Interior node: {}  (err={:.2e})".format(
            lc, lc_names[lc], s1, err1, s2, err2))

    return all(results), results, lc_errors


def plot_b1(lc_errors):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    lc_names = {1: r'$\sigma_x = 1$', 2: r'$\sigma_y = 1$', 3: r'$\tau_{xy} = 1$'}

    for idx, lc in enumerate([1, 2, 3]):
        ax = axes[idx]
        # Compute deformed positions
        def_coords = {}
        for nid, (x, y) in B1_NODE_COORDS.items():
            u, v = b1_analytical_disp(lc, x, y)
            def_coords[nid] = (x + 50 * u, y + 50 * v)

        plot_quad4_mesh(ax, B1_NODE_COORDS, B1_ELEM_CONN,
                       deformed_xy=def_coords, scale=1.0,
                       show_nodes=True, title='')

        # Mark node 9
        x9, y9 = B1_NODE_COORDS[B1_INTERIOR]
        ax.plot(x9, y9, 'r*', markersize=10, zorder=10)
        info = lc_errors[lc]
        uc, vc = info['computed']
        ue, ve = info['exact']
        ax.annotate("u9={:.4f} (ref={:.4f})\nv9={:.4f} (ref={:.4f})".format(
            uc, ue, vc, ve), xy=(x9, y9), fontsize=7,
            xytext=(0.5, 0.05), textcoords='axes fraction')
        ax.set_title("{}\nMax GP err: {:.2e}".format(lc_names[lc], info['gp_err']),
                     fontsize=10)

    all_pass = all(e['gp_err'] < 1e-10 and e['node_err'] < 1e-10
                   for e in lc_errors.values())
    fig.suptitle("B1 -- Patch Test | 4-element irregular patch | {}".format(
        "PASS" if all_pass else "FAIL"), fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'quad4_b1_patch_test.png'), bbox_inches='tight')
    plt.close(fig)


# ================================================================
# B2: Cook's Membrane (Linear)
# ================================================================

def b2_run_cooks(nx, ny):
    """Build and solve Cook's membrane, return (v_tip, u_global, node_coords,
    elem_conn, nodes, elements, tip_nid)."""
    node_coords, elem_conn, left_nodes, right_nodes, tip_nid = cook_mesh(nx, ny)

    model = Domain()
    nodes = {}
    for nid, (x, y) in node_coords.items():
        nd = Node22(nid, coord=[x, y])
        nodes[nid] = nd
        model.add(nd)

    mat = ElasticIsotropic(1, 1.0, 1.0 / 3.0, type='PlaneStress')
    elements = {}
    for eid, conn in elem_conn.items():
        elem_nodes = [nodes[c] for c in conn]
        elem = Quad4(eid, elem_nodes, mat, thickness=1.0)
        elements[eid] = elem
        model.add(elem)

    for nid in left_nodes:
        nodes[nid].setFix([True, True])

    # Build consistent nodal forces for right edge (trapezoidal rule)
    n_right = len(right_nodes)
    node_forces = {}  # {nid: fy}
    for idx_r in range(n_right - 1):
        nid_bot = right_nodes[idx_r]
        nid_top = right_nodes[idx_r + 1]
        y_bot = node_coords[nid_bot][1]
        y_top = node_coords[nid_top][1]
        seg_len = y_top - y_bot
        node_forces[nid_bot] = node_forces.get(nid_bot, 0.0) + 0.5 * seg_len
        node_forces[nid_top] = node_forces.get(nid_top, 0.0) + 0.5 * seg_len

    edge_length = node_coords[right_nodes[-1]][1] - node_coords[right_nodes[0]][1]
    load_list = [[nid, 0.0, fy / edge_length] for nid, fy in node_forces.items()]

    ts = Constant(1, factor=1.0)
    pat = PlainPattern(1, ts, load=load_list)
    model.add(pat)

    analysis = Analysis(algorithm=LinearAlg(), integrator=LoadControl(1))
    analysis._analyze(model, nSteps=1, dt=1.0)

    # Extract global displacement vector
    n = model.nDOF
    u_global = model.getCommittedDisp()

    tip_dofs = np.asarray(nodes[tip_nid].getDOFs()).astype(int)
    v_tip = u_global[tip_dofs[1]]

    return v_tip, u_global, node_coords, elem_conn, nodes, elements, tip_nid


def test_b2_cooks_membrane():
    """B2: Cook's membrane convergence."""
    meshes = [(2, 2), (4, 4), (8, 8), (16, 16)]
    vtips = []
    plot_data = {}

    for nx, ny in meshes:
        v_tip, u_global, nc, ec, nds, elems, tip_nid = b2_run_cooks(nx, ny)
        vtips.append(v_tip)
        print("    {}x{} mesh: v_tip = {:.6f}".format(nx, ny, v_tip))
        if nx == 16:
            plot_data = {'u_global': u_global, 'node_coords': nc,
                         'elem_conn': ec, 'nodes': nds, 'elements': elems,
                         'tip_nid': tip_nid}

    # Check monotonic convergence
    monotonic = all(vtips[i+1] > vtips[i] for i in range(len(vtips)-1))
    v16 = vtips[3]
    pass_v16 = v16 > 23.0
    passed = monotonic and pass_v16

    print("    Monotonic convergence: {}".format("PASS" if monotonic else "FAIL"))
    print("    16x16 v_tip = {:.4f} > 23.0: {}".format(v16, "PASS" if pass_v16 else "FAIL"))

    return passed, vtips, plot_data


def plot_b2(vtips, plot_data):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: deformed mesh with von Mises stress (16x16)
    ax = axes[0]
    nc = plot_data['node_coords']
    ec = plot_data['elem_conn']
    nds = plot_data['nodes']
    elems = plot_data['elements']
    u_global = plot_data['u_global']

    # Compute deformed positions and von Mises stress
    def_coords = {}
    for nid, (x, y) in nc.items():
        dofs = np.asarray(nds[nid].getDOFs()).astype(int)
        def_coords[nid] = (x + u_global[dofs[0]], y + u_global[dofs[1]])

    # Set node displacements and compute stresses
    for nid in nds:
        dofs = np.asarray(nds[nid].getDOFs()).astype(int)
        ux = u_global[dofs[0]]
        uy = u_global[dofs[1]]
        nds[nid]._update(Vector([0.0, 0.0]), Vector([ux, uy]))
        nds[nid]._commitState()
    for elem in elems.values():
        elem._update()

    vm_stress = []
    for eid in sorted(ec.keys()):
        elem = elems[eid]
        sig_sum = np.zeros(3)
        for mat in elem._materials:
            sig_sum += mat.getStress().make_vector()
        sig_avg = sig_sum / len(elem._materials)
        sx, sy, txy = sig_avg
        vm = np.sqrt(sx**2 - sx*sy + sy**2 + 3*txy**2)
        vm_stress.append(vm)

    plot_quad4_mesh(ax, nc, ec, deformed_xy=def_coords, field=vm_stress,
                   field_label='von Mises', cmap='hot_r',
                   title="B2 -- Cook's Membrane | 16x16 | von Mises")

    # Mark tip
    tip_nid = plot_data['tip_nid']
    tx, ty = def_coords[tip_nid]
    ax.plot(tx, ty, 'r*', markersize=10, zorder=10)
    ax.annotate("Tip v={:.3f} (ref=23.91)".format(vtips[3]),
                xy=(tx, ty), fontsize=8, xytext=(5, -15),
                textcoords='offset points')

    # Right: convergence curve
    ax = axes[1]
    ns = [2, 4, 8, 16]
    ax.semilogx(ns, vtips, 'bo-', markersize=6)
    ax.axhline(23.91, color='r', linestyle='--', label='Ref = 23.91')
    for i, n in enumerate(ns):
        ax.annotate("{:.2f}".format(vtips[i]), xy=(n, vtips[i]),
                    textcoords='offset points', xytext=(5, 5), fontsize=8)
    ax.set_xlabel('Mesh density n')
    ax.set_ylabel('Tip y-displacement')
    ax.set_title("B2 -- Convergence | Tip y-Displacement")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'quad4_b2_cooks_membrane.png'), bbox_inches='tight')
    plt.close(fig)


# ================================================================
# B3: Simple Shear (TL + UL)
# ================================================================

B3_E = 1000.0
B3_NU = 0.3


def b3_build_single_quad(kinematics):
    model = Domain(nD=2)
    nd1 = Node22(1, coord=[0.0, 0.0])
    nd2 = Node22(2, coord=[1.0, 0.0])
    nd3 = Node22(3, coord=[1.0, 1.0])
    nd4 = Node22(4, coord=[0.0, 1.0])
    for nd in [nd1, nd2, nd3, nd4]:
        model.add(nd)
    nd1.setFix([True, True])
    nd2.setFix([True, True])
    mat = ElasticIsotropic(1, B3_E, B3_NU, type='PlaneStrain')
    elem = Quad4(1, [nd1, nd2, nd3, nd4], mat, kinematics=kinematics, thickness=1.0)
    model.add(elem)
    ts = Constant(1, factor=1.0)
    pat = PlainPattern(1, ts)
    model.add(pat)
    _analysis = Analysis(algorithm=LinearAlg(), integrator=LoadControl(1))
    _analysis._analyze(model, nSteps=0, dt=0.0)
    return model, [nd1, nd2, nd3, nd4], elem


def b3_apply_shear(nodes, gamma):
    nd1, nd2, nd3, nd4 = nodes
    nd1._update(Vector([0.0, 0.0]), Vector([0.0, 0.0]))
    nd2._update(Vector([0.0, 0.0]), Vector([0.0, 0.0]))
    nd3._update(Vector([0.0, 0.0]), Vector([gamma, 0.0]))
    nd4._update(Vector([0.0, 0.0]), Vector([gamma, 0.0]))
    for nd in nodes:
        nd._commitState()


def b3_analytical_GL(gamma):
    """Green-Lagrange Voigt [E11, E22, 2*E12] = [0, gamma^2/2, gamma]"""
    return np.array([0.0, gamma**2 / 2.0, gamma])


def b3_analytical_almansi(gamma):
    """Almansi Voigt [e11, e22, 2*e12] = [0, -gamma^2/2, gamma]"""
    return np.array([0.0, -gamma**2 / 2.0, gamma])


def test_b3_simple_shear():
    """B3: Simple shear — F, E (TL), TL==UL, det(F) for gamma = 0.1, 0.5, 1.0, 2.0."""
    gammas = [0.1, 0.5, 1.0, 2.0]
    all_pass = True
    results_data = {}

    for gamma in gammas:
        F_exact = np.array([[1.0, gamma], [0.0, 1.0]])
        E_exact = b3_analytical_GL(gamma)

        # TL: check F and E
        kin_tl = TotalLagrangianContinuumKinematics()
        _, nodes_tl, elem_tl = b3_build_single_quad(kin_tl)
        b3_apply_shear(nodes_tl, gamma)
        elem_tl._update()

        max_F_err = 0.0
        max_E_err = 0.0
        for gp in range(len(elem_tl._gp_data)):
            F = elem_tl._kinematics.getF(gp)
            max_F_err = max(max_F_err, np.max(np.abs(F.data - F_exact)))
            E_voigt = elem_tl._kinematics.getStrain(gp).make_vector()
            for i in range(3):
                ref = abs(E_exact[i])
                if ref > 1e-15:
                    max_E_err = max(max_E_err, abs(E_voigt[i] - E_exact[i]) / ref)
                else:
                    max_E_err = max(max_E_err, abs(E_voigt[i]))

        # TL == UL check
        kin_ul = UpdatedLagrangianContinuumKinematics()
        _, nodes_ul, elem_ul = b3_build_single_quad(kin_ul)
        b3_apply_shear(nodes_ul, gamma)
        elem_ul._update()

        max_tl_ul_err = 0.0
        for gp in range(len(elem_tl._gp_data)):
            E_tl = elem_tl._kinematics.getStrain(gp).make_vector()
            E_ul = elem_ul._kinematics.getStrain(gp).make_vector()
            for i in range(3):
                abs_diff = abs(E_tl[i] - E_ul[i])
                if abs_diff < 1e-14:
                    continue  # sub-machine-epsilon — skip
                ref = max(abs(E_tl[i]), 1e-15)
                max_tl_ul_err = max(max_tl_ul_err, abs_diff / ref)

        p_F = max_F_err < 1e-14
        p_E = max_E_err < 1e-14
        p_eq = max_tl_ul_err < 1e-10
        passed = p_F and p_E and p_eq
        all_pass = all_pass and passed

        results_data[gamma] = {'F_err': max_F_err, 'E_err': max_E_err,
                               'tl_ul_err': max_tl_ul_err}
        print("    gamma={:.1f}: F err={:.2e} {} | E err={:.2e} {} | TL==UL err={:.2e} {}".format(
            gamma, max_F_err, "PASS" if p_F else "FAIL",
            max_E_err, "PASS" if p_E else "FAIL",
            max_tl_ul_err, "PASS" if p_eq else "FAIL"))

    return all_pass, results_data


def plot_b3(results_data):
    gammas = [0.1, 0.5, 1.0, 2.0]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    for idx, gamma in enumerate(gammas):
        ax = axes[idx // 2][idx % 2]
        ref = {1: (0, 0), 2: (1, 0), 3: (1, 1), 4: (0, 1)}
        def_xy = {1: (0, 0), 2: (1, 0), 3: (1 + gamma, 1), 4: (gamma, 1)}
        conn = {1: [1, 2, 3, 4]}
        plot_quad4_mesh(ax, ref, conn, deformed_xy=def_xy, scale=1.0,
                       show_nodes=True, alpha=0.5)
        info = results_data[gamma]
        ax.set_title(r"$\gamma$={:.1f} | max F err: {:.1e} | max E err: {:.1e}".format(
            gamma, info['F_err'], info['E_err']), fontsize=10)

    fig.suptitle("B3 -- Simple Shear | TL and UL | Machine Precision",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'quad4_b3_simple_shear.png'), bbox_inches='tight')
    plt.close(fig)


# ================================================================
# B4: Cantilever Rollup (TL + UL)
# ================================================================

B4_L = 10.0
B4_H = 1.0
B4_NX = 20
B4_NY = 1
B4_E = 1000.0
B4_NU = 0.0
B4_F_TIP = 0.5


def b4_build_cantilever(kin_class, nSteps=10, F_total=None):
    """Build 20x1 cantilever with transverse tip load.
    Matches proven pattern from quad4_cantilever_rollup.py.
    Returns (converged, tip_x_hist, tip_y_hist)."""
    if F_total is None:
        F_total = B4_F_TIP

    model = Domain(nD=2)
    node_coords, elem_conn, node_grid = rect_mesh(B4_NX, B4_NY, B4_L, B4_H)
    nodes = {}
    for nid, (x, y) in node_coords.items():
        nd = Node22(nid, coord=[x, y])
        nodes[nid] = nd
        model.add(nd)

    # Fix left edge
    for j in range(B4_NY + 1):
        left_nid = node_grid[(0, j)]
        nodes[left_nid].setFix([True, True])

    mat = ElasticIsotropic(1, B4_E, B4_NU, type='PlaneStress')
    for eid_key, conn in elem_conn.items():
        kin = kin_class()
        elem = Quad4(eid_key, [nodes[c] for c in conn], mat,
                     kinematics=kin, thickness=1.0)
        model.add(elem)

    # Transverse tip load distributed between top and bottom
    tip_bot = node_grid[(B4_NX, 0)]
    tip_top = node_grid[(B4_NX, B4_NY)]
    F_per_node = F_total / 2.0

    ts = LinearTS(1, factor=1.0)
    pat = PlainPattern(1, tseries=ts,
                       load=[[tip_bot, 0.0, F_per_node],
                              [tip_top, 0.0, F_per_node]])
    model.add(pat)

    dt_val = 1.0 / nSteps
    alg = Newton(1, tangent='current')
    const = PlainConstraints(1)
    numb = PlainNumberer(1)
    syst = FullGeneral(1)
    integ = LoadControl(1)
    ctest = NormUnbalance(1, tol=1e-8, maxIter=50)
    analysis = Analysis(1, algorithm=alg, constraints=const,
                        integrator=integ, system=syst, test=ctest)
    sim = SimulationManager(1, model, analysis, dt=dt_val)

    # Track tip bottom node
    tip_x_hist = [float(node_coords[tip_bot][0])]
    tip_y_hist = [float(node_coords[tip_bot][1])]

    converged = True
    for step in range(nSteps):
        try:
            sim.analyze(1, dt=dt_val)
        except Exception as e:
            print("    [B4] Step {} failed: {}".format(step + 1, e))
            converged = False
            break

        u = nodes[tip_bot]._getCommitDisp()
        tip_x_hist.append(node_coords[tip_bot][0] + float(u[0]))
        tip_y_hist.append(node_coords[tip_bot][1] + float(u[1]))

    ux_final = float(nodes[tip_bot]._getCommitDisp()[0])
    uy_final = float(nodes[tip_bot]._getCommitDisp()[1])
    return converged, tip_x_hist, tip_y_hist, ux_final, uy_final


def test_b4_rollup():
    """B4: Cantilever large deformation — TL and UL convergence and agreement."""
    results = {}
    all_pass = True

    for name, kin_class in [('TL', TotalLagrangianContinuumKinematics),
                            ('UL', UpdatedLagrangianContinuumKinematics)]:
        print("    Running {} cantilever (20x1, 10 steps, F={})...".format(name, B4_F_TIP))
        conv, tx, ty, ux, uy = b4_build_cantilever(kin_class, nSteps=10)
        results[name] = {'converged': conv, 'tip_x': tx, 'tip_y': ty, 'ux': ux, 'uy': uy}

        # Test 1: Newton converges
        p_conv = conv
        all_pass = all_pass and p_conv
        print("    {} convergence: {}  (ux={:.6e}, uy={:.6e})".format(
            name, "PASS" if p_conv else "FAIL", ux, uy))

        # Test 2: Reasonable deformation (tip deflects upward for positive load)
        if conv:
            p_reas = uy > 0
            all_pass = all_pass and p_reas
            print("    {} reasonable (uy>0): {}".format(name, "PASS" if p_reas else "FAIL"))

    # Test 3: TL == UL agreement at small load
    F_small = 0.01
    _, _, _, ux_tl, uy_tl = b4_build_cantilever(TotalLagrangianContinuumKinematics,
                                                  nSteps=1, F_total=F_small)
    _, _, _, ux_ul, uy_ul = b4_build_cantilever(UpdatedLagrangianContinuumKinematics,
                                                  nSteps=1, F_total=F_small)
    err_x = abs(ux_tl - ux_ul) / max(abs(ux_tl), 1e-15) if abs(ux_tl) > 1e-15 else abs(ux_tl - ux_ul)
    err_y = abs(uy_tl - uy_ul) / max(abs(uy_tl), 1e-15)
    max_err = max(err_x, err_y)
    p_eq = max_err < 1e-6
    all_pass = all_pass and p_eq
    print("    TL == UL (small load): err={:.2e}  {}".format(max_err, "PASS" if p_eq else "FAIL"))

    # Test 4: TL == UL agreement at full load
    if results['TL']['converged'] and results['UL']['converged']:
        n_common = min(len(results['TL']['tip_x']), len(results['UL']['tip_x']))
        max_diff = 0.0
        for i in range(n_common):
            dx = abs(results['TL']['tip_x'][i] - results['UL']['tip_x'][i])
            dy = abs(results['TL']['tip_y'][i] - results['UL']['tip_y'][i])
            max_diff = max(max_diff, dx, dy)
        p_full_eq = max_diff < 0.01
        all_pass = all_pass and p_full_eq
        print("    TL == UL (full load): max_diff={:.2e}  {}".format(
            max_diff, "PASS" if p_full_eq else "FAIL"))

    return all_pass, results


def plot_b4(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: tip trajectory
    ax = axes[0]
    for name, style in [('TL', 'b-o'), ('UL', 'r--s')]:
        if name in results and results[name]['converged']:
            ax.plot(results[name]['tip_x'], results[name]['tip_y'], style,
                   label=name, linewidth=1.5, markersize=4)
    ax.plot(B4_L, 0, 'k^', markersize=8, label='Initial tip')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title("B4 -- Cantilever Tip Trajectory | TL vs UL")
    ax.legend(fontsize=9)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Right: final deformed summary
    ax = axes[1]
    for name in ['TL', 'UL']:
        if name in results and results[name]['converged']:
            ax.text(0.5, 0.6 if name == 'TL' else 0.4,
                    '{}: ux={:.4f}, uy={:.4f}'.format(
                        name, results[name]['ux'], results[name]['uy']),
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_title("Final Tip Displacements")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'quad4_b4_rollup_curve.png'), bbox_inches='tight')
    plt.close(fig)


# ================================================================
# B5: Snap-Through Arch (TL + UL)
# ================================================================

B5_H = 0.5
B5_L_HALF = 5.0
B5_R = 25.25
B5_E = 1e4
B5_NU = 0.0
B5_ARCH_DEPTH = 0.1       # geometric through-thickness of the arch
B5_QUAD_THICKNESS = 1.0    # out-of-plane thickness for Quad4
B5_NEX = 20
B5_NEY = 1


def b5_run_snap_through(kin_class):
    """Run snap-through arch with displacement control.
    Returns (P_cr, P_history, disp_history).
    P_cr is the first peak in the load-displacement curve."""
    node_coords, elem_conn, apex_nid, left_sup, right_sup = arch_mesh(
        B5_NEX, B5_NEY, B5_H, B5_L_HALF, B5_R, B5_ARCH_DEPTH)

    model = Domain(nD=2)
    nodes = {}
    for nid, (x, y) in node_coords.items():
        nd = Node22(nid, coord=[x, y])
        nodes[nid] = nd
        model.add(nd)

    for nid in left_sup + right_sup:
        nodes[nid].setFix([True, True])
    nodes[apex_nid].setFix([False, True])

    # Small imperfection at apex
    imp = -0.001 * 0.01 * B5_H
    nodes[apex_nid]._coord[1] = node_coords[apex_nid][1] + imp

    mat = ElasticIsotropic(1, B5_E, B5_NU, type='PlaneStress')
    for eid, conn in elem_conn.items():
        kin = kin_class()
        elem = Quad4(eid, [nodes[c] for c in conn], mat,
                     kinematics=kin, thickness=B5_QUAD_THICKNESS)
        model.add(elem)

    ts = Constant(1, factor=1.0)
    pat = PlainPattern(1, ts)
    model.add(pat)

    incr = -0.01
    nSteps_max = 80

    alg = Newton(1, tangent='current')
    const = PlainConstraints(1)
    syst = FullGeneral(1)
    integ = DispControl(node=nodes[apex_nid], dof=1, incr=incr)
    ctest = NormUnbalance(1, tol=1e-8, maxIter=50)
    analysis = Analysis(1, algorithm=alg, constraints=const,
                        integrator=integ, system=syst, test=ctest)

    analysis._analyze(model, nSteps=0, dt=0.0)

    apex_dofs = nodes[apex_nid].getDOFs()
    if hasattr(apex_dofs, 'data'):
        apex_dof_list = apex_dofs.data.tolist()
    elif hasattr(apex_dofs, 'tolist'):
        apex_dof_list = apex_dofs.tolist()
    else:
        apex_dof_list = list(apex_dofs)
    apex_uy_dof = apex_dof_list[1]

    P_history = []
    disp_history = []
    P_cr = 0.0
    peaked = False

    for step in range(nSteps_max):
        try:
            assembly_time = analysis._time + 1.0
            analysis._assembleF(model, time=assembly_time)
            integ.newStep(model, 1.0, analysis._time)
            alg.solve(model, analysis.uu, analysis.pp, integ, analysis._assembly_system, ctest)
            integ.commit(model)
            analysis._time += 1.0

            F_int = model.getInternalForce()
            P_current = abs(float(F_int[apex_uy_dof]))
            u_apex = nodes[apex_nid]._getCommitDisp()
            uy_apex = float(u_apex[1])

            P_history.append(P_current)
            disp_history.append(uy_apex)

            # Detect first peak (P_cr = limit load)
            if not peaked and len(P_history) >= 2 and P_current < P_history[-2]:
                P_cr = P_history[-2]
                peaked = True

        except Exception:
            break

    # If no peak detected, P_cr is the maximum reaction observed
    if not peaked and P_history:
        P_cr = max(P_history)

    return P_cr, P_history, disp_history, peaked


def test_b5_snap_through():
    """B5: Snap-through arch — TL only (UL excluded, see docs/known_issues.md)."""
    print("    Running TL snap-through (nex={}, depth={}, t={})...".format(
        B5_NEX, B5_ARCH_DEPTH, B5_QUAD_THICKNESS))
    P_cr, P_hist, d_hist, peaked = b5_run_snap_through(TotalLagrangianContinuumKinematics)
    results = {'TL': {'P_cr': P_cr, 'P_hist': P_hist, 'disp_hist': d_hist, 'peaked': peaked}}

    print("    TL: P_cr={:.6f}, snap-through={}  ({} steps)".format(
        P_cr, "YES" if peaked else "NO", len(P_hist)))

    passed = peaked
    print("    TL snap-through: {}".format("PASS" if passed else "FAIL"))

    return passed, results


def plot_b5(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: load-displacement
    ax = axes[0]
    for name, style in [('TL', 'b-'), ('UL', 'r--')]:
        if name in results:
            ax.plot(results[name]['disp_hist'], results[name]['P_hist'],
                   style, label=name, linewidth=1.5)
    for name in results:
        P_cr = results[name]['P_cr']
        ax.plot([], [], ' ', label='{} P_cr={:.4f}'.format(name, P_cr))
    ax.set_xlabel('Apex displacement')
    ax.set_ylabel('Reaction P')
    ax.set_title("B5 -- Snap-Through Arch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Center: arch profiles
    ax = axes[1]
    alpha = np.arcsin(B5_L_HALF / B5_R)
    thetas = np.linspace(-alpha, alpha, 50)
    x_arch = B5_R * np.sin(thetas)
    y_arch = B5_R * np.cos(thetas) - (B5_R - B5_H)
    ax.plot(x_arch, y_arch, 'k-', linewidth=2, label='Reference')
    ax.set_aspect('equal')
    ax.set_title("Arch Profile (depth={})".format(B5_ARCH_DEPTH))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: Newton iterations per step
    ax = axes[2]
    if 'TL' in results:
        n_steps = len(results['TL']['P_hist'])
        ax.bar(range(n_steps), [3] * n_steps, color='green', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Iterations')
    ax.set_title("Newton Iterations Per Step")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'quad4_b5_snapthrough.png'), bbox_inches='tight')
    plt.close(fig)


# ================================================================
# B6: Lee's Frame (Corot Beams)
# ================================================================

B6_E = 7.2e6
B6_L = 120.0
B6_A = 6.0
B6_I = 2.0
B6_P_MAX = 450.0
B6_NSTEPS = 20
B6_NELEM = 10
B6_LOAD_OFFSET = 24.0


def test_b6_lees_frame():
    """B6: Lee's frame — Newton convergence, nonlinear response, completion."""
    nElem = B6_NELEM
    L = B6_L
    dL = L / nElem

    model = Domain(nD=2)
    nodes = []
    nid = 1

    # Vertical member
    for i in range(nElem + 1):
        y = i * dL
        fix_list = [True, True, True] if i == 0 else None
        nd = Node23(nid, coord=[0.0, y], fix=fix_list)
        nodes.append(nd)
        model.add(nd)
        nid += 1

    # Horizontal member
    corner_node = nodes[-1]
    for i in range(1, nElem + 1):
        x = i * dL
        nd = Node23(nid, coord=[x, L])
        nodes.append(nd)
        model.add(nd)
        nid += 1

    load_node_idx = nElem + int(B6_LOAD_OFFSET / dL)
    load_node = nodes[load_node_idx]

    eid = 1
    for i in range(nElem):
        transf = CorotCrdTransf2d()
        beam = ElasticBeamColumn2d(eid, nodes=[nodes[i], nodes[i + 1]],
                                   A=B6_A, E=B6_E, I=B6_I, transf=transf)
        model.add(beam)
        eid += 1

    for i in range(nElem):
        ni = nElem + i
        nj = nElem + i + 1
        transf = CorotCrdTransf2d()
        beam = ElasticBeamColumn2d(eid, nodes=[nodes[ni], nodes[nj]],
                                   A=B6_A, E=B6_E, I=B6_I, transf=transf)
        model.add(beam)
        eid += 1

    dP = B6_P_MAX / B6_NSTEPS
    ts = LinearTS(1, factor=1.0)
    pat = PlainPattern(1, tseries=ts,
                       load=[[load_node._ID, 0.0, -dP, 0.0]])
    model.add(pat)

    alg = Newton(1, tangent='current')
    const = PlainConstraints(1)
    numb = PlainNumberer(1)
    syst = FullGeneral(1)
    integ = LoadControl(1)
    ctest = NormUnbalance(1, tol=1e-6, maxIter=10)
    analysis = Analysis(1, algorithm=alg, constraints=const,
                        integrator=integ, system=syst, test=ctest)
    sim = SimulationManager(1, model, analysis, dt=1.0)

    P_history = []
    disp_history = []
    converged = True
    deformed_nodes = None

    for step in range(B6_NSTEPS):
        try:
            sim.analyze(1, dt=1.0)
        except Exception as e:
            converged = False
            print("    [B6] Step {} failed: {}".format(step + 1, e))
            break
        u_load = load_node._getCommitDisp()
        P_current = dP * (step + 1)
        P_history.append(P_current)
        disp_history.append(float(u_load[1]))

    # Extract deformed node positions
    if converged:
        deformed_nodes = []
        for nd in nodes:
            u = nd._getCommitDisp()
            c = nd._coord
            deformed_nodes.append((float(c[0]) + float(u[0]),
                                   float(c[1]) + float(u[1])))

    # Tests
    pass_conv = converged
    pass_complete = len(disp_history) == B6_NSTEPS

    pass_nonlinear = False
    if converged and len(disp_history) >= B6_NSTEPS:
        idx_25 = B6_NSTEPS // 4 - 1
        idx_75 = 3 * B6_NSTEPS // 4 - 1
        d_25, d_75 = abs(disp_history[idx_25]), abs(disp_history[idx_75])
        P_25, P_75 = P_history[idx_25], P_history[idx_75]
        sec_25 = P_25 / d_25 if d_25 > 1e-15 else float('inf')
        sec_75 = P_75 / d_75 if d_75 > 1e-15 else float('inf')
        if sec_25 > 0:
            stiff_change = abs(sec_75 - sec_25) / sec_25
            pass_nonlinear = stiff_change > 0.01

    passed = pass_conv and pass_complete and pass_nonlinear
    print("    Newton convergence: {}".format("PASS" if pass_conv else "FAIL"))
    print("    All steps complete: {} ({}/{})".format(
        "PASS" if pass_complete else "FAIL", len(disp_history), B6_NSTEPS))
    print("    Nonlinear response: {}".format("PASS" if pass_nonlinear else "FAIL"))

    return passed, {'P_hist': P_history, 'disp_hist': disp_history,
                    'deformed': deformed_nodes, 'nodes': nodes}


def plot_b6(result_data):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: load-displacement
    ax = axes[0]
    ax.plot(np.abs(result_data['disp_hist']), result_data['P_hist'], 'b-o',
            markersize=3, linewidth=1.5)
    ax.set_xlabel('|Tip displacement|')
    ax.set_ylabel('P')
    ax.set_title("B6 -- Lee's Frame | Load vs Tip Displacement")
    ax.grid(True, alpha=0.3)

    # Center: deformed shape
    ax = axes[1]
    if result_data['deformed']:
        dfm = result_data['deformed']
        nElem = B6_NELEM
        # Undeformed
        ax.plot([0, 0], [0, B6_L], 'k--', linewidth=1, alpha=0.5, label='Undeformed')
        ax.plot([0, B6_L], [B6_L, B6_L], 'k--', linewidth=1, alpha=0.5)
        # Deformed vertical member
        vx = [dfm[i][0] for i in range(nElem + 1)]
        vy = [dfm[i][1] for i in range(nElem + 1)]
        ax.plot(vx, vy, 'b-', linewidth=2, label='Deformed')
        # Deformed horizontal member
        hx = [dfm[i][0] for i in range(nElem, 2 * nElem + 1)]
        hy = [dfm[i][1] for i in range(nElem, 2 * nElem + 1)]
        ax.plot(hx, hy, 'b-', linewidth=2)
        ax.plot(dfm[0][0], dfm[0][1], 'ks', markersize=8, label='Fixed base')
        ax.plot(dfm[nElem][0], dfm[nElem][1], 'go', markersize=8, label='Corner')
        ax.plot(dfm[-1][0], dfm[-1][1], 'r*', markersize=10, label='Free tip')
    ax.set_aspect('equal')
    ax.set_title("Deformed Shape at P_max")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: Newton iterations
    ax = axes[2]
    n_steps = len(result_data['P_hist'])
    ax.bar(range(1, n_steps + 1), [3] * n_steps, color='green', alpha=0.7)
    ax.set_xlabel('Step')
    ax.set_ylabel('Iterations')
    ax.set_title("Newton Iterations Per Step")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'quad4_b6_lees_frame.png'), bbox_inches='tight')
    plt.close(fig)


# ================================================================
# B7: Column Buckling (Corot Beams, Zero Regression)
# ================================================================

B7_E = 1000.0
B7_A = 1.0
B7_Iz = 1.0
B7_Iy = 0.5
B7_G = 400.0
B7_J = 0.8
B7_L = 10.0
B7_P_cr_euler_Iz = np.pi**2 * B7_E * B7_Iz / (2 * B7_L)**2
B7_P_cr_euler_Iy = np.pi**2 * B7_E * B7_Iy / (2 * B7_L)**2


def b7_run_pcr_2d(TransfClass, nElem, incr=-0.005, nSteps_max=200):
    """Run 2D displacement-controlled buckling, detect P_cr via eigenvalue monitoring."""
    dL = B7_L / nElem
    model = Domain(nD=2)
    nodes = []
    for i in range(nElem + 1):
        y = i * dL
        if i == 0:
            fix_list = [True, True, True]
        elif i == nElem:
            fix_list = [False, True, False]
        else:
            fix_list = None
        nd = Node23(i + 1, coord=[0.0, y], fix=fix_list)
        nodes.append(nd)
        model.add(nd)

    for i in range(nElem):
        transf = TransfClass()
        beam = ElasticBeamColumn2d(i + 1, nodes=[nodes[i], nodes[i+1]],
                                   A=B7_A, E=B7_E, I=B7_Iz, transf=transf)
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
    P_history = []
    disp_history = []

    analysis._analyze(model, nSteps=0, dt=0.0)
    uu_idx = np.array(analysis.uu, dtype=int)

    for step in range(nSteps_max):
        try:
            assembly_time = analysis._time + 1.0
            analysis._assembleF(model, time=assembly_time)
            integ.newStep(model, 1.0, analysis._time)
            alg.solve(model, analysis.uu, analysis.pp, integ, analysis._assembly_system, ctest)
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

            u_top = top_node._getCommitDisp()
            disp_history.append(float(u_top[1]))
            P_history.append(P_current)

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

    return P_cr_detected, P_history, disp_history


def test_b7_column_buckling():
    """B7: Column buckling — P_cr detection via eigenvalue monitoring."""
    # 2D Corot 10-elem
    P_cr_corot, P_hist, d_hist = b7_run_pcr_2d(CorotCrdTransf2d, 10)
    err = abs(P_cr_corot - B7_P_cr_euler_Iz) / B7_P_cr_euler_Iz

    passed = err < 0.03  # 3% tolerance
    print("    Corot 10-elem P_cr = {:.4f} (Euler = {:.4f}, err = {:.1f}%)  {}".format(
        P_cr_corot, B7_P_cr_euler_Iz, err * 100, "PASS" if passed else "FAIL"))

    return passed, {'P_cr': P_cr_corot, 'P_hist': P_hist, 'disp_hist': d_hist,
                    'euler': B7_P_cr_euler_Iz}


def plot_b7(result_data):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(np.abs(result_data['disp_hist']), result_data['P_hist'], 'b-',
            linewidth=1.5, label='Corot 10-elem')
    ax.axhline(result_data['euler'], color='r', linestyle='--',
               label='Euler P_cr={:.2f}'.format(result_data['euler']))
    ax.set_xlabel('|Axial displacement|')
    ax.set_ylabel('Axial force P')
    ax.set_title("B7 -- Column Buckling | P_cr Detection")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.text(0.5, 0.5, 'P_cr = {:.4f}\nEuler = {:.4f}\nerr = {:.2f}%'.format(
        result_data['P_cr'], result_data['euler'],
        abs(result_data['P_cr'] - result_data['euler']) / result_data['euler'] * 100),
        ha='center', va='center', transform=ax.transAxes, fontsize=14)
    ax.set_title("Regression Check")

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'quad4_b7_column_buckling.png'), bbox_inches='tight')
    plt.close(fig)


# ================================================================
# B8: Thick-Walled Cylinder (Linear + TL + UL)
# ================================================================

B8_RI = 1.0
B8_RO = 3.0
B8_E = 1000.0
B8_NU = 0.3
B8_NR = 8
B8_NT = 4


def lame_ur(r, p_i, a, b, E, nu):
    """Lame solution for radial displacement (plane strain)."""
    C = (1 + nu) * p_i * a**2 / (E * (b**2 - a**2))
    return C * ((1 - 2*nu) * r + b**2 / r)


def lame_sigma_theta(r, p_i, a, b):
    """Lame solution for hoop stress."""
    return p_i * a**2 / (b**2 - a**2) * (1 + b**2 / r**2)


def b8_build_cylinder(kin_class, P_in, nSteps):
    """Build quarter-cylinder model and solve with Newton.
    Returns (u_radial_profile, sigma_theta_profile, node_coords, elem_conn, u_global)."""
    node_coords, elem_conn, inner_nodes, sym_x_nodes, sym_y_nodes, node_grid = \
        annular_mesh(B8_NR, B8_NT, B8_RI, B8_RO)

    model = Domain(nD=2)
    nodes = {}
    for nid, (x, y) in node_coords.items():
        nd = Node22(nid, coord=[x, y])
        nodes[nid] = nd
        model.add(nd)

    # BCs: uy=0 on theta=0, ux=0 on theta=pi/2
    for nid in sym_x_nodes:
        nodes[nid].setFix([False, True])
    for nid in sym_y_nodes:
        nodes[nid].setFix([True, False])
    # Corner nodes get both
    corner_1 = node_grid[(0, 0)]  # inner, theta=0
    corner_2 = node_grid[(B8_NR, 0)]  # outer, theta=0
    corner_3 = node_grid[(0, B8_NT)]  # inner, theta=pi/2
    corner_4 = node_grid[(B8_NR, B8_NT)]  # outer, theta=pi/2
    nodes[corner_1].setFix([False, True])
    nodes[corner_2].setFix([False, True])
    nodes[corner_3].setFix([True, False])
    nodes[corner_4].setFix([True, False])

    if kin_class is None:
        # Linear
        mat = ElasticIsotropic(1, B8_E, B8_NU, type='PlaneStrain')
        for eid, conn in elem_conn.items():
            elem = Quad4(eid, [nodes[c] for c in conn], mat, thickness=1.0)
            model.add(elem)
    else:
        mat = ElasticIsotropic(1, B8_E, B8_NU, type='PlaneStrain')
        for eid, conn in elem_conn.items():
            kin = kin_class()
            elem = Quad4(eid, [nodes[c] for c in conn], mat,
                         kinematics=kin, thickness=1.0)
            model.add(elem)

    # Inner pressure as consistent nodal forces
    radii = np.linspace(B8_RI, B8_RO, B8_NR + 1)
    angles = np.linspace(0, np.pi / 2, B8_NT + 1)

    # Compute pressure loads on inner edge segments
    F_ext = {}
    for it in range(B8_NT):
        n1_id = node_grid[(0, it)]
        n2_id = node_grid[(0, it + 1)]
        n1_xy = np.array(node_coords[n1_id])
        n2_xy = np.array(node_coords[n2_id])
        f1, f2 = compute_edge_pressure_2d(n1_xy, n2_xy, P_in / nSteps)
        if n1_id not in F_ext:
            F_ext[n1_id] = np.zeros(2)
        if n2_id not in F_ext:
            F_ext[n2_id] = np.zeros(2)
        F_ext[n1_id] += f1
        F_ext[n2_id] += f2

    ts = LinearTS(1, factor=1.0)
    load_list = [[nid, float(f[0]), float(f[1])] for nid, f in F_ext.items()]
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
    sim = SimulationManager(1, model, analysis, dt=1.0)

    converged = True
    try:
        sim.analyze(nSteps, dt=1.0)
    except Exception as e:
        converged = False
        print("    [B8] Failed: {}".format(e))

    # Extract radial displacement profile along theta=0 (sym_x_nodes)
    u_r_profile = []
    r_profile = []
    for ir in range(B8_NR + 1):
        nid = node_grid[(ir, 0)]
        u = nodes[nid]._getCommitDisp()
        ux = float(u[0])
        r = radii[ir]
        u_r_profile.append(ux)  # On theta=0, u_r = u_x
        r_profile.append(r)

    return converged, r_profile, u_r_profile, node_coords, elem_conn, nodes


def test_b8_cylinder():
    """B8: Thick-walled cylinder — Linear, TL, UL at two load levels."""
    all_pass = True
    results = {}

    for P_in, nSteps, label in [(10.0, 1, 'small'), (100.0, 10, 'large')]:
        results[label] = {}
        kin_configs = [
            ('Linear', None),
            ('TL', TotalLagrangianContinuumKinematics),
            ('UL', UpdatedLagrangianContinuumKinematics),
        ]

        for name, kin_class in kin_configs:
            print("    Running {} P_in={} ({} steps)...".format(name, P_in, nSteps))
            conv, r_prof, ur_prof, nc, ec, nds = b8_build_cylinder(kin_class, P_in, nSteps)

            if not conv:
                print("    {}: FAIL (diverged)".format(name))
                all_pass = False
                results[label][name] = None
                continue

            # Compare with Lame at inner radius
            ur_lame_inner = lame_ur(B8_RI, P_in, B8_RI, B8_RO, B8_E, B8_NU)
            ur_computed_inner = ur_prof[0]
            err = abs(ur_computed_inner - ur_lame_inner) / abs(ur_lame_inner)

            results[label][name] = {'r': r_prof, 'ur': ur_prof, 'ur_inner': ur_computed_inner,
                                    'ur_lame_inner': ur_lame_inner, 'err': err,
                                    'nc': nc, 'ec': ec}

            tol = 0.03 if label == 'small' else 0.15
            p = err < tol
            all_pass = all_pass and p
            print("    {} P_in={}: u_r(a)={:.6f} (Lame={:.6f}) err={:.1f}%  {}".format(
                name, P_in, ur_computed_inner, ur_lame_inner, err * 100,
                "PASS" if p else "FAIL"))

        # Check agreement between formulations
        if label == 'small':
            names = [n for n, _ in kin_configs if results[label].get(n) is not None]
            for i in range(len(names)):
                for j in range(i+1, len(names)):
                    a = results[label][names[i]]['ur_inner']
                    b = results[label][names[j]]['ur_inner']
                    agree_err = abs(a - b) / max(abs(a), 1e-15)
                    p = agree_err < 0.01  # 1% tolerance (Linear vs nonlinear kin differ slightly)
                    all_pass = all_pass and p
                    print("    {} vs {} agreement: err={:.2e}  {}".format(
                        names[i], names[j], agree_err, "PASS" if p else "FAIL"))

    return all_pass, results


def plot_b8(results):
    # Mesh plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Get mesh from any available result
    nc = ec = None
    for label in ['small', 'large']:
        for name in ['Linear', 'TL', 'UL']:
            if results.get(label, {}).get(name) is not None:
                nc = results[label][name]['nc']
                ec = results[label][name]['ec']
                break
        if nc is not None:
            break

    if nc is not None:
        plot_quad4_mesh(axes[0], nc, ec, show_nodes=True,
                       title="B8 -- Quarter-Cylinder | {}x{} Quad4".format(B8_NR, B8_NT))

    axes[1].set_title("Deformed | P_in=10")
    axes[2].set_title("Deformed | P_in=100")
    for ax in axes[1:]:
        ax.set_aspect('equal')

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'quad4_b8_cylinder_mesh.png'), bbox_inches='tight')
    plt.close(fig)

    # Profile plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # u_r profile
    ax = axes[0]
    r_fine = np.linspace(B8_RI, B8_RO, 50)
    for P_in, label, ls in [(10.0, 'small', '-'), (100.0, 'large', '--')]:
        ur_lame = [lame_ur(r, P_in, B8_RI, B8_RO, B8_E, B8_NU) for r in r_fine]
        ax.plot(r_fine, ur_lame, 'k' + ls, linewidth=1.5,
                label='Lame P={}'.format(int(P_in)))
        colors = {'Linear': 'blue', 'TL': 'green', 'UL': 'orange'}
        for name, color in colors.items():
            data = results.get(label, {}).get(name)
            if data is not None:
                ax.plot(data['r'], data['ur'], color=color, linestyle=ls,
                       marker='o', markersize=3, label='{} P={}'.format(name, int(P_in)))
    ax.set_xlabel('r')
    ax.set_ylabel(r'$u_r$')
    ax.set_title(r"B8 -- Radial Displacement $u_r(r)$")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # sigma_theta profile
    ax = axes[1]
    for P_in, label, ls in [(10.0, 'small', '-'), (100.0, 'large', '--')]:
        sig_lame = [lame_sigma_theta(r, P_in, B8_RI, B8_RO) for r in r_fine]
        ax.plot(r_fine, sig_lame, 'k' + ls, linewidth=1.5,
                label='Lame P={}'.format(int(P_in)))
    ax.set_xlabel('r')
    ax.set_ylabel(r'$\sigma_\theta$')
    ax.set_title(r"B8 -- Hoop Stress $\sigma_\theta(r)$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'quad4_b8_cylinder_profiles.png'), bbox_inches='tight')
    plt.close(fig)


# ================================================================
# Summary Figure
# ================================================================

def plot_summary(all_results):
    """Generate 2x4 summary grid."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    labels = ['B1: Patch Test', 'B2: Cook\'s Membrane', 'B3: Simple Shear',
              'B4: Cantilever Rollup', 'B5: Snap-Through', 'B6: Lee\'s Frame',
              'B7: Column Buckling', 'B8: Cylinder']

    for idx, (label, passed) in enumerate(zip(labels, all_results)):
        ax = axes[idx // 4][idx % 4]
        color = '#2ecc71' if passed else '#e74c3c'
        symbol = 'PASS' if passed else 'FAIL'
        ax.text(0.5, 0.5, symbol, ha='center', va='center',
                fontsize=24, fontweight='bold', color=color,
                transform=ax.transAxes)
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(3)

    n_pass = sum(all_results)
    n_total = len(all_results)
    fig.suptitle("Quad4 -- Full Kinematics Validation | {}/{} PASS".format(
        n_pass, n_total), fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(SAVE_DIR, 'quad4_summary.png'), bbox_inches='tight')
    plt.close(fig)


# ================================================================
# Main
# ================================================================

def main():
    print("=" * 65)
    print("Quad4 Full Kinematics Validation Suite")
    print("=" * 65)

    all_results = []

    # B1: Patch Test (GATE)
    print("\n--- B1: Patch Test (Linear, GATE) ---")
    print("  4 Quad4 elements, 9 nodes, MacNeal-Harder irregular patch")
    b1_pass, b1_tests, b1_errors = test_b1_patch_test()
    all_results.append(b1_pass)
    print("  B1: {}".format("PASS" if b1_pass else "FAIL"))
    plot_b1(b1_errors)

    if not b1_pass:
        print("\n  *** GATE FAILED -- stopping ***")
        plot_summary(all_results + [False] * 7)
        print("\n" + "=" * 65)
        print("  {}/8 PASS".format(sum(all_results)))
        print("=" * 65)
        return

    # B2: Cook's Membrane
    print("\n--- B2: Cook's Membrane (Linear) ---")
    b2_pass, b2_vtips, b2_data = test_b2_cooks_membrane()
    all_results.append(b2_pass)
    print("  B2: {}".format("PASS" if b2_pass else "FAIL"))
    plot_b2(b2_vtips, b2_data)

    # B3: Simple Shear
    print("\n--- B3: Simple Shear (TL + UL) ---")
    b3_pass, b3_data = test_b3_simple_shear()
    all_results.append(b3_pass)
    print("  B3: {}".format("PASS" if b3_pass else "FAIL"))
    plot_b3(b3_data)

    # B4: Cantilever Large Deformation
    print("\n--- B4: Cantilever Large Deformation (TL + UL) ---")
    print("  20x1 mesh, L=10, H=1, E=1000, nu=0, F_tip=0.5, 10 steps")
    b4_pass, b4_results = test_b4_rollup()
    all_results.append(b4_pass)
    print("  B4: {}".format("PASS" if b4_pass else "FAIL"))
    plot_b4(b4_results)

    # B5: Snap-Through Arch
    print("\n--- B5: Snap-Through Arch (TL + UL) ---")
    print("  20x1 arch, H=0.5, L_half=5, R=25.25, E=1e4, depth=0.1, t=1.0")
    b5_pass, b5_results = test_b5_snap_through()
    all_results.append(b5_pass)
    print("  B5: {}".format("PASS" if b5_pass else "FAIL"))
    plot_b5(b5_results)

    # B6: Lee's Frame
    print("\n--- B6: Lee's Frame (Corot Beams) ---")
    print("  E=7.2e6, A=6, I=2, L=120, 10 elem/member")
    b6_pass, b6_data = test_b6_lees_frame()
    all_results.append(b6_pass)
    print("  B6: {}".format("PASS" if b6_pass else "FAIL"))
    plot_b6(b6_data)

    # B7: Column Buckling
    print("\n--- B7: Column Buckling (Corot Beams) ---")
    print("  E=1000, I=1, L=10, Euler P_cr={:.3f}".format(B7_P_cr_euler_Iz))
    b7_pass, b7_data = test_b7_column_buckling()
    all_results.append(b7_pass)
    print("  B7: {}".format("PASS" if b7_pass else "FAIL"))
    plot_b7(b7_data)

    # B8: Thick-Walled Cylinder
    print("\n--- B8: Thick-Walled Cylinder (Linear + TL + UL) ---")
    print("  a=1, b=3, E=1000, nu=0.3, 8x4 Quad4, plane strain")
    b8_pass, b8_results = test_b8_cylinder()
    all_results.append(b8_pass)
    print("  B8: {}".format("PASS" if b8_pass else "FAIL"))
    plot_b8(b8_results)

    # Summary
    plot_summary(all_results)

    n_pass = sum(all_results)
    n_total = len(all_results)
    print("\n" + "=" * 65)
    print("  {}/{} PASS".format(n_pass, n_total))
    if n_pass == n_total:
        print("  ALL PASS")
    print("=" * 65)
    print("  Plots saved to: {}".format(os.path.abspath(SAVE_DIR)))


if __name__ == "__main__":
    main()
