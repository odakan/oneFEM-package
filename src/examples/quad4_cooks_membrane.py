##-----------------------------------------------------------------------##
#  Benchmark: Cook's Membrane (Quad4 convergence test)
#
#  Trapezoidal membrane clamped on left edge, distributed shear on right.
#  Geometry: left edge x=0, y in [0,44]; right edge x=48, y in [44,60].
#
#  E = 1.0, nu = 1/3, PlaneStress, thickness = 1.0
#  Total shear force on right edge = 1.0 (distributed as consistent nodal loads)
#
#  Reference tip displacement (top-right corner, v):
#    Converged value ~23.96 (extrapolated, Quad4 is stiff)
#    16x16 mesh: ~23.91 (Pian & Sumihara 1984, de Souza Neto et al.)
#
#  Pass criterion: monotonic convergence, 16x16 mesh v_tip > 23.0
##-----------------------------------------------------------------------##

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from oneFEM.model import Domain
from oneFEM.model.node import Node22
from oneFEM.model.element.continuum.quad4 import Quad4
from oneFEM.model.material.nD.elastic_isotropic import ElasticIsotropic
from oneFEM.model.pattern import Plain as PlainPattern
from oneFEM.model.tseries import Constant
from oneFEM.analysis.main import Analysis
from oneFEM.analysis.algorithm.linear import Linear
from oneFEM.analysis.integrator.static.load_control import LoadControl


def cook_mesh(nx, ny):
    """Generate Cook's membrane mesh.

    Geometry:
      Bottom-left: (0, 0)
      Bottom-right: (48, 44)
      Top-right: (48, 60)
      Top-left: (0, 44)

    :param nx: number of elements in x
    :param ny: number of elements in y
    :return: node_coords dict {nid: (x, y)}, elem_conn dict {eid: [n1,n2,n3,n4]},
             left_nodes list, right_nodes list, tip_node_id
    """
    node_coords = {}
    nid = 1
    node_grid = {}  # (i, j) -> nid

    for j in range(ny + 1):
        for i in range(nx + 1):
            # Parametric coordinates
            xi = i / nx
            eta = j / ny

            # Bilinear interpolation of corner coordinates
            # Corners: BL=(0,0), BR=(48,44), TR=(48,60), TL=(0,44)
            x = 48.0 * xi
            y_bot = 44.0 * xi        # bottom edge: 0 -> 44
            y_top = 44.0 + 16.0 * xi  # top edge: 44 -> 60
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

    # Left edge nodes (x=0, i=0)
    left_nodes = [node_grid[(0, j)] for j in range(ny + 1)]

    # Right edge nodes (x=48, i=nx)
    right_nodes = [node_grid[(nx, j)] for j in range(ny + 1)]

    # Tip node = top-right corner
    tip_node_id = node_grid[(nx, ny)]

    return node_coords, elem_conn, left_nodes, right_nodes, tip_node_id


def run_cooks_membrane(nx, ny, verbose=False):
    """Build and solve Cook's membrane for given mesh density."""
    node_coords, elem_conn, left_nodes, right_nodes, tip_node_id = cook_mesh(nx, ny)

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

    # Fix left edge
    for nid in left_nodes:
        nodes[nid].setFix([True, True])

    # Build consistent nodal forces for right edge
    # Trapezoidal rule: uniform load on each segment, half to each end
    n_right = len(right_nodes)
    node_forces = {}  # {nid: fy}
    for idx in range(n_right - 1):
        nid_bot = right_nodes[idx]
        nid_top = right_nodes[idx + 1]
        y_bot = node_coords[nid_bot][1]
        y_top = node_coords[nid_top][1]
        seg_len = y_top - y_bot
        node_forces[nid_bot] = node_forces.get(nid_bot, 0.0) + 0.5 * seg_len
        node_forces[nid_top] = node_forces.get(nid_top, 0.0) + 0.5 * seg_len

    # Normalize so total = 1.0
    edge_length = node_coords[right_nodes[-1]][1] - node_coords[right_nodes[0]][1]
    load_list = []
    for nid, fy in node_forces.items():
        load_list.append([nid, 0.0, fy / edge_length])  # [nodeID, fx, fy]

    ts = Constant(1, factor=1.0)
    pat = PlainPattern(1, ts, load=load_list)
    model.add(pat)

    # Solve via Analysis
    analysis = Analysis(algorithm=Linear(), integrator=LoadControl(1))
    analysis._analyze(model, nSteps=1, dt=1.0)

    # Extract tip displacement
    tip_nd = nodes[tip_node_id]
    u_tip = tip_nd._getCommitDisp()
    v_tip = float(u_tip[1])

    if verbose:
        u_tip_x = float(u_tip[0])
        print(f"    {nx}x{ny} mesh ({len(elem_conn)} elements, {len(node_coords)} nodes):")
        print(f"      u_tip = {u_tip_x:.6f}")
        print(f"      v_tip = {v_tip:.6f}")

    return v_tip


if __name__ == "__main__":
    print("=" * 60)
    print("Cook's Membrane — Quad4 Convergence Test")
    print("  E = 1.0, nu = 1/3, PlaneStress, t = 1.0")
    print("  Total shear force on right edge = 1.0")
    print("=" * 60)

    meshes = [(2, 2), (4, 4), (8, 8), (16, 16), (32, 32)]
    results = []

    for nx, ny in meshes:
        v_tip = run_cooks_membrane(nx, ny, verbose=True)
        results.append((nx, ny, v_tip))

    # Check monotonic convergence
    print(f"\n{'='*60}")
    print("  Convergence summary:")
    print(f"  {'Mesh':>8s}  {'v_tip':>12s}  {'Monotonic':>10s}")
    print(f"  {'-'*8}  {'-'*12}  {'-'*10}")

    all_monotonic = True
    for idx, (nx, ny, v) in enumerate(results):
        if idx > 0:
            mono = v > results[idx - 1][2]
            if not mono:
                all_monotonic = False
            mono_str = "PASS" if mono else "FAIL"
        else:
            mono_str = "---"
        print(f"  {nx:>2d}x{ny:<2d}     {v:>12.6f}  {mono_str:>10s}")

    # Check 16x16 result
    v_16 = results[3][2]  # 16x16
    ref_min = 23.0
    test_16x16 = v_16 > ref_min
    test_mono = all_monotonic

    print(f"\n  Monotonic convergence: {'PASS' if test_mono else 'FAIL'}")
    print(f"  16x16 v_tip = {v_16:.4f} > {ref_min:.1f}: {'PASS' if test_16x16 else 'FAIL'}")

    n_pass = sum([test_mono, test_16x16])
    print(f"\n{'='*60}")
    print(f"  {n_pass}/2 PASS")
    if n_pass == 2:
        print("  ALL PASS")
    print(f"{'='*60}")
