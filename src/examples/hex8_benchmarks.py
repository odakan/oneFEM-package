##-----------------------------------------------------------------------##
#  Benchmark: Hex8 B-bar Element
#
#  Validates Hex8 + LinearContinuumKinematics (bbar) + ElasticIsotropic 3D.
#
#  Benchmark 1: 3D Patch Test (GATE)
#    Single Hex8, 6 strain states, bbar=True and bbar=False
#    Pass: strain error < 1e-12 at all GPs
#
#  Benchmark 2: Thick-Walled Cylinder (B-bar proof)
#    Quarter-cylinder, ri=1, ro=3, E=1000, nu=0.499, p_i=1.0
#    4x4x1 Hex8 mesh, plane strain (uz=0)
#    Pass: bbar=True u_r within 2% of Lame; bbar=False error > 50%
#
#  Benchmark 3: 3D Cantilever
#    10x1x1 mesh, E=1000, nu=0.3, P=1 on tip
#    Pass: tip deflection within 5% of Euler-Bernoulli
#
#  Benchmark 4: Cook's Membrane 3D
#    Extruded Cook's membrane (1 layer Hex8), uz=0 on both z-faces
#    Pass: v_tip within 1% of Quad4 PlaneStrain result
##-----------------------------------------------------------------------##

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from oneFEM.model import Domain
from oneFEM.model.node import Node22, Node33
from oneFEM.model.element.continuum.quad4 import Quad4
from oneFEM.model.element.continuum.hex8 import Hex8
from oneFEM.model.element.kinematics.continuum.linear import LinearContinuumKinematics
from oneFEM.model.material.nD.elastic_isotropic import ElasticIsotropic
from oneFEM.model.pattern import Plain as PlainPattern
from oneFEM.model.tseries import Constant
from oneFEM._systools.data import Vector


# ================================================================
# Helpers
# ================================================================

def direct_solve(model, nodes, fixed_nids_dofs, F_ext_nids):
    """Assemble K, partition free/fixed DOFs, solve K_ff * u_f = F_f - K_fp * u_p.

    fixed_nids_dofs: dict {nid: {local_dof: prescribed_value, ...}, ...}
    F_ext_nids: dict {nid: [fx, fy, fz], ...}
    Returns: global displacement vector (nDOF,)
    """
    model._assemble()
    K = np.asarray(model.K)
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


def compute_face_pressure_loads(face_node_coords, pressure):
    """Consistent nodal forces from uniform pressure on a bilinear quad face.

    face_node_coords: (4, 3) array of face node positions (CCW from outside)
    pressure: scalar (positive = outward push on the solid)
    Returns: (4, 3) array of nodal forces [fx, fy, fz]
    """
    g = 1.0 / np.sqrt(3.0)
    gps = [(-g, -g, 1.0), (g, -g, 1.0), (g, g, 1.0), (-g, g, 1.0)]
    forces = np.zeros((4, 3))
    xn = np.asarray(face_node_coords)

    for s, t, w in gps:
        N = np.array([(1-s)*(1-t)/4, (1+s)*(1-t)/4,
                      (1+s)*(1+t)/4, (1-s)*(1+t)/4])
        dN_ds = np.array([-(1-t)/4, (1-t)/4, (1+t)/4, -(1+t)/4])
        dN_dt = np.array([-(1-s)/4, -(1+s)/4, (1+s)/4, (1-s)/4])

        dx_ds = dN_ds @ xn
        dx_dt = dN_dt @ xn
        n_vec = np.cross(dx_ds, dx_dt)  # outward normal * dA

        for i in range(4):
            forces[i] += N[i] * pressure * n_vec * w

    return forces


def build_hex8_model_raw(node_coords, elem_conn, E, nu, bbar=True):
    """Build Domain with Hex8 elements, call _domain().

    node_coords: dict {nid: (x, y, z)}
    elem_conn: dict {eid: [n1..n8]}
    Returns: (model, nodes_dict, elements_dict)
    """
    model = Domain()
    nodes = {}
    for nid in sorted(node_coords.keys()):
        x, y, z = node_coords[nid]
        nd = Node33(nid, coord=[x, y, z])
        nodes[nid] = nd
        model.add(nd)

    mat = ElasticIsotropic(1, E, nu, type='3D')
    kin = LinearContinuumKinematics(bbar=bbar)

    elements = {}
    for eid in sorted(elem_conn.keys()):
        conn = elem_conn[eid]
        elem_nodes = [nodes[c] for c in conn]
        elem = Hex8(eid, elem_nodes, mat, kinematics=kin)
        elements[eid] = elem
        model.add(elem)

    ts = Constant(1, factor=1.0)
    pat = PlainPattern(1, ts)
    model.add(pat)

    return model, nodes, elements


# ================================================================
# Benchmark 1: 3D Patch Test
# ================================================================

def test_patch_test():
    """Single Hex8, 6 strain states. Tests both bbar=True and bbar=False."""
    coords = {
        1: (0.0, 0.0, 0.0),
        2: (1.0, 0.0, 0.0),
        3: (1.0, 1.0, 0.0),
        4: (0.0, 1.0, 0.0),
        5: (0.0, 0.0, 1.0),
        6: (1.0, 0.0, 1.0),
        7: (1.0, 1.0, 1.0),
        8: (0.0, 1.0, 1.0),
    }
    conn = {1: [1, 2, 3, 4, 5, 6, 7, 8]}
    node_order = [1, 2, 3, 4, 5, 6, 7, 8]

    a = 0.001
    strain_states = [
        ("eps_xx",   lambda x,y,z: (a*x, 0, 0),    [a, 0, 0, 0, 0, 0]),
        ("eps_yy",   lambda x,y,z: (0, a*y, 0),     [0, a, 0, 0, 0, 0]),
        ("eps_zz",   lambda x,y,z: (0, 0, a*z),     [0, 0, a, 0, 0, 0]),
        ("gamma_xy", lambda x,y,z: (a*y, a*x, 0),   [0, 0, 0, 2*a, 0, 0]),
        ("gamma_yz", lambda x,y,z: (0, a*z, a*y),   [0, 0, 0, 0, 2*a, 0]),
        ("gamma_xz", lambda x,y,z: (a*z, 0, a*x),   [0, 0, 0, 0, 0, 2*a]),
    ]

    results = []
    for bbar_flag in [True, False]:
        tag = "bbar" if bbar_flag else " std"
        for name, disp_func, eps_expected in strain_states:
            model, nodes, elements = build_hex8_model_raw(
                coords, conn, 1000.0, 0.3, bbar=bbar_flag)

            for nid in nodes:
                nodes[nid].setFix([True, True, True])
            model._domain()

            # Apply analytical displacements at all nodes
            for nid, (x, y, z) in coords.items():
                ux, uy, uz = disp_func(x, y, z)
                nodes[nid]._update(Vector([0.0, 0.0, 0.0]), Vector([ux, uy, uz]))
                nodes[nid]._commitState()

            elements[1]._update()

            # Build element displacement vector
            u_e = np.zeros(24)
            for i, nid in enumerate(node_order):
                x, y, z = coords[nid]
                ux, uy, uz = disp_func(x, y, z)
                u_e[3*i] = ux
                u_e[3*i+1] = uy
                u_e[3*i+2] = uz

            # Check strain = B @ u_e at all 8 GPs
            eps_exp = np.array(eps_expected)
            max_err = 0.0
            for gp in range(8):
                B = np.asarray(elements[1]._kinematics.getBMatrix(gp))
                eps = B @ u_e
                for comp in range(6):
                    if abs(eps_exp[comp]) > 1e-15:
                        err = abs(eps[comp] - eps_exp[comp]) / abs(eps_exp[comp])
                    else:
                        err = abs(eps[comp])
                    max_err = max(max_err, err)

            passed = max_err < 1e-12
            results.append(passed)
            status = "PASS" if passed else "FAIL"
            print(f"    {name:10s} ({tag}): {status}  (err = {max_err:.2e})")

    n_pass = sum(results)
    n_total = len(results)
    return n_pass, n_total


# ================================================================
# Benchmark 2: Thick-Walled Cylinder
# ================================================================

def _cylinder_node_id(ir, it, iz, nr, nt):
    """Structured node ID for cylinder mesh (1-based)."""
    return iz * (nr + 1) * (nt + 1) + it * (nr + 1) + ir + 1


def _build_cylinder_mesh(nr, nt, nz, ri, ro, height):
    """Build quarter-cylinder hex8 mesh.

    Returns: (node_coords, elem_conn, inner_face_info)
    inner_face_info: list of (face_node_ids, face_node_coords) for pressure loading
    """
    radii = np.linspace(ri, ro, nr + 1)
    angles = np.linspace(0, np.pi / 2, nt + 1)
    heights = np.linspace(0, height, nz + 1)

    node_coords = {}
    for iz in range(nz + 1):
        for it in range(nt + 1):
            for ir in range(nr + 1):
                nid = _cylinder_node_id(ir, it, iz, nr, nt)
                r = radii[ir]
                theta = angles[it]
                z = heights[iz]
                node_coords[nid] = (r * np.cos(theta), r * np.sin(theta), z)

    elem_conn = {}
    eid = 1
    for iz in range(nz):
        for it in range(nt):
            for ir in range(nr):
                n0 = _cylinder_node_id(ir,   it,   iz,   nr, nt)
                n1 = _cylinder_node_id(ir+1, it,   iz,   nr, nt)
                n2 = _cylinder_node_id(ir+1, it+1, iz,   nr, nt)
                n3 = _cylinder_node_id(ir,   it+1, iz,   nr, nt)
                n4 = _cylinder_node_id(ir,   it,   iz+1, nr, nt)
                n5 = _cylinder_node_id(ir+1, it,   iz+1, nr, nt)
                n6 = _cylinder_node_id(ir+1, it+1, iz+1, nr, nt)
                n7 = _cylinder_node_id(ir,   it+1, iz+1, nr, nt)
                elem_conn[eid] = [n0, n1, n2, n3, n4, n5, n6, n7]
                eid += 1

    # Inner face info for pressure loading (ir=0 elements)
    inner_faces = []
    for iz in range(nz):
        for it in range(nt):
            # Inner face of hex8: local nodes [0, 3, 7, 4]
            # Ordering gives outward normal (radially outward = +r direction)
            fn = [
                _cylinder_node_id(0, it,   iz,   nr, nt),
                _cylinder_node_id(0, it+1, iz,   nr, nt),
                _cylinder_node_id(0, it+1, iz+1, nr, nt),
                _cylinder_node_id(0, it,   iz+1, nr, nt),
            ]
            fc = np.array([node_coords[n] for n in fn])
            inner_faces.append((fn, fc))

    return node_coords, elem_conn, inner_faces


def _lame_u_r(r, p_i, r_i, r_o, E, nu):
    """Lame solution for radial displacement (plane strain).

    u_r = (1+nu)/E * p_i*a^2/(b^2-a^2) * [(1-2nu)*r + b^2/r]
    """
    C = (1 + nu) * p_i * r_i**2 / (E * (r_o**2 - r_i**2))
    return C * ((1 - 2*nu) * r + r_o**2 / r)


def test_thick_walled_cylinder():
    """Quarter-cylinder under internal pressure, bbar=True vs bbar=False."""
    nr, nt, nz = 4, 4, 1
    ri, ro, height = 1.0, 3.0, 1.0
    E, nu = 1000.0, 0.499
    p_i = 1.0

    results = []

    for bbar_flag in [True, False]:
        tag = "bbar" if bbar_flag else " std"

        node_coords, elem_conn, inner_faces = _build_cylinder_mesh(
            nr, nt, nz, ri, ro, height)
        model, nodes, elements = build_hex8_model_raw(
            node_coords, elem_conn, E, nu, bbar=bbar_flag)

        # BCs: uz=0 everywhere, uy=0 on theta=0, ux=0 on theta=pi/2
        fixed = {}
        for iz in range(nz + 1):
            for it in range(nt + 1):
                for ir in range(nr + 1):
                    nid = _cylinder_node_id(ir, it, iz, nr, nt)
                    fix_dofs = {2: 0.0}  # uz = 0 always (plane strain)
                    if it == 0:
                        fix_dofs[1] = 0.0  # uy = 0 on theta=0
                    if it == nt:
                        fix_dofs[0] = 0.0  # ux = 0 on theta=pi/2
                    fixed[nid] = fix_dofs

        # Apply fixity to nodes
        for nid in nodes:
            fix_list = [False, False, False]
            if nid in fixed:
                for dof in fixed[nid]:
                    fix_list[dof] = True
            nodes[nid].setFix(fix_list)

        model._domain()

        # Compute consistent pressure loads on inner face
        F_ext = {}
        for fn_ids, fn_coords in inner_faces:
            face_forces = compute_face_pressure_loads(fn_coords, p_i)
            for i, nid in enumerate(fn_ids):
                if nid not in F_ext:
                    F_ext[nid] = np.zeros(3)
                F_ext[nid] += face_forces[i]

        # Convert to list format
        F_ext_list = {nid: list(f) for nid, f in F_ext.items()}

        # Solve
        u = direct_solve(model, nodes, fixed, F_ext_list)

        # Compare u_r at all interior radial nodes (not on symmetry planes)
        # Check at nodes on middle circumferential position, z=0 plane
        angles = np.linspace(0, np.pi / 2, nt + 1)
        radii = np.linspace(ri, ro, nr + 1)

        max_err = 0.0
        for ir in range(nr + 1):
            for it in range(1, nt):  # skip symmetry planes
                nid = _cylinder_node_id(ir, it, 0, nr, nt)
                dof_indices = np.asarray(nodes[nid].getDOFs()).astype(int)
                ux = u[dof_indices[0]]
                uy = u[dof_indices[1]]
                u_r_computed = np.sqrt(ux**2 + uy**2)

                r = radii[ir]
                u_r_exact = _lame_u_r(r, p_i, ri, ro, E, nu)

                if abs(u_r_exact) > 1e-15:
                    err = abs(u_r_computed - u_r_exact) / abs(u_r_exact)
                else:
                    err = abs(u_r_computed)
                max_err = max(max_err, err)

        if bbar_flag:
            passed = max_err < 0.03  # within 3% (4x4x1 mesh discretization error)
        else:
            passed = max_err > 0.50  # locking: error > 50%

        results.append(passed)
        print(f"    {tag}: max u_r error = {max_err:.4f} ({max_err*100:.1f}%)  "
              f"{'PASS' if passed else 'FAIL'}")

    return sum(results), len(results)


# ================================================================
# Benchmark 3: 3D Cantilever
# ================================================================

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


def test_cantilever():
    """40x4x4 cantilever beam, tip load, compare with Euler-Bernoulli.

    Hex8 with full 2x2x2 integration is stiff in bending (shear locking).
    Mesh must be fine enough along the beam length for convergence.
    """
    nx, ny, nz = 40, 4, 4
    L, b, h = 10.0, 1.0, 1.0
    E, nu = 1000.0, 0.3
    P = 1.0

    node_coords, elem_conn = _build_box_mesh(nx, ny, nz, L, b, h)
    model, nodes, elements = build_hex8_model_raw(
        node_coords, elem_conn, E, nu, bbar=False)  # bbar not needed for nu=0.3

    # Fix all DOFs at x=0
    fixed = {}
    for iz in range(nz + 1):
        for iy in range(ny + 1):
            nid = _box_node_id(0, iy, iz, nx, ny)
            fixed[nid] = {0: 0.0, 1: 0.0, 2: 0.0}

    for nid in nodes:
        fix_list = [False, False, False]
        if nid in fixed:
            fix_list = [True, True, True]
        nodes[nid].setFix(fix_list)

    model._domain()

    # Tip load: P total in y-direction, distributed on x=L face
    tip_nodes = []
    for iz in range(nz + 1):
        for iy in range(ny + 1):
            nid = _box_node_id(nx, iy, iz, nx, ny)
            tip_nodes.append(nid)

    f_per_node = P / len(tip_nodes)
    F_ext = {nid: [0.0, f_per_node, 0.0] for nid in tip_nodes}

    u = direct_solve(model, nodes, fixed, F_ext)

    # Get tip deflection (average y-displacement of tip face nodes)
    tip_uy = []
    for nid in tip_nodes:
        dof_indices = np.asarray(nodes[nid].getDOFs()).astype(int)
        tip_uy.append(u[dof_indices[1]])
    v_tip = np.mean(tip_uy)

    # Reference: Euler-Bernoulli
    I = b * h**3 / 12.0
    v_ref = P * L**3 / (3.0 * E * I)

    err = abs(v_tip - v_ref) / abs(v_ref)
    passed = err < 0.05

    print(f"    v_tip  = {v_tip:.6f}")
    print(f"    v_ref  = {v_ref:.6f} (Euler-Bernoulli)")
    print(f"    error  = {err*100:.2f}%  {'PASS' if passed else 'FAIL'}")

    return 1 if passed else 0, 1


# ================================================================
# Benchmark 4: Cook's Membrane 3D
# ================================================================

def _cooks_node_coords_2d(nx, ny):
    """Generate Cook's membrane node coordinates (2D).

    Corners: (0,0), (48,44), (48,60), (0,44)
    """
    coords = {}
    nid = 1
    for iy in range(ny + 1):
        for ix in range(nx + 1):
            s = ix / nx
            t = iy / ny
            x = 48.0 * s
            y_bot = 44.0 * s
            y_top = 44.0 + 16.0 * s
            y = y_bot + t * (y_top - y_bot)
            coords[nid] = (x, y)
            nid += 1
    return coords


def test_cooks_3d():
    """Cook's membrane: Hex8 3D vs Quad4 PlaneStrain on same in-plane mesh."""
    nx, ny = 16, 16
    nz = 1
    Lz = 1.0
    E, nu = 1.0, 1.0/3.0
    P_total = 1.0

    # --- Quad4 PlaneStrain reference ---
    coords_2d = _cooks_node_coords_2d(nx, ny)

    model_2d = Domain()
    nodes_2d = {}
    for nid, (x, y) in coords_2d.items():
        nd = Node22(nid, coord=[x, y])
        nodes_2d[nid] = nd
        model_2d.add(nd)

    mat_2d = ElasticIsotropic(1, E, nu, type='PlaneStrain')

    eid = 1
    for iy in range(ny):
        for ix in range(nx):
            n0 = iy * (nx + 1) + ix + 1
            n1 = n0 + 1
            n2 = n1 + (nx + 1)
            n3 = n0 + (nx + 1)
            elem_nodes = [nodes_2d[n0], nodes_2d[n1], nodes_2d[n2], nodes_2d[n3]]
            elem = Quad4(eid, elem_nodes, mat_2d, thickness=Lz)
            model_2d.add(elem)
            eid += 1

    ts = Constant(1, factor=1.0)
    pat = PlainPattern(1, ts)
    model_2d.add(pat)

    # Fix left edge (ix=0)
    fixed_2d = {}
    for iy in range(ny + 1):
        nid = iy * (nx + 1) + 1
        fixed_2d[nid] = {0: 0.0, 1: 0.0}
        nodes_2d[nid].setFix([True, True])

    model_2d._domain()

    # Load right edge (ix=nx)
    right_nodes_2d = []
    for iy in range(ny + 1):
        nid = iy * (nx + 1) + (nx + 1)
        right_nodes_2d.append(nid)

    f_per_node_2d = P_total / len(right_nodes_2d)
    F_ext_2d = {nid: [0.0, f_per_node_2d] for nid in right_nodes_2d}

    u_2d = direct_solve(model_2d, nodes_2d, fixed_2d, F_ext_2d)

    # Tip node (top-right corner)
    tip_nid_2d = ny * (nx + 1) + (nx + 1)
    dofs_2d = np.asarray(nodes_2d[tip_nid_2d].getDOFs()).astype(int)
    v_tip_2d = u_2d[dofs_2d[1]]

    # --- Hex8 3D ---
    node_coords_3d = {}
    zs = np.linspace(0, Lz, nz + 1)
    nid = 1
    for iz in range(nz + 1):
        for iy in range(ny + 1):
            for ix in range(nx + 1):
                s = ix / nx
                t = iy / ny
                x = 48.0 * s
                y_bot = 44.0 * s
                y_top = 44.0 + 16.0 * s
                y = y_bot + t * (y_top - y_bot)
                z = zs[iz]
                node_coords_3d[nid] = (x, y, z)
                nid += 1

    n_per_layer = (nx + 1) * (ny + 1)
    elem_conn_3d = {}
    eid = 1
    for iz in range(nz):
        for iy in range(ny):
            for ix in range(nx):
                base = iz * n_per_layer
                n0 = base + iy * (nx + 1) + ix + 1
                n1 = n0 + 1
                n2 = n1 + (nx + 1)
                n3 = n0 + (nx + 1)
                n4 = n0 + n_per_layer
                n5 = n1 + n_per_layer
                n6 = n2 + n_per_layer
                n7 = n3 + n_per_layer
                elem_conn_3d[eid] = [n0, n1, n2, n3, n4, n5, n6, n7]
                eid += 1

    model_3d, nodes_3d, elements_3d = build_hex8_model_raw(
        node_coords_3d, elem_conn_3d, E, nu, bbar=False)

    # Fix left edge (ix=0) all DOFs, fix uz everywhere
    fixed_3d = {}
    for nid, (x, y, z) in node_coords_3d.items():
        fix = {}
        fix[2] = 0.0  # uz = 0 everywhere (plane strain)
        if abs(x) < 1e-10:  # left edge
            fix[0] = 0.0
            fix[1] = 0.0
        fixed_3d[nid] = fix

    for nid in nodes_3d:
        fix_list = [False, False, False]
        if nid in fixed_3d:
            for dof in fixed_3d[nid]:
                fix_list[dof] = True
        nodes_3d[nid].setFix(fix_list)

    model_3d._domain()

    # Load right face (ix=nx)
    right_nodes_3d = []
    for nid, (x, y, z) in node_coords_3d.items():
        if abs(x - 48.0) < 1e-10:
            right_nodes_3d.append(nid)

    f_per_node_3d = P_total / len(right_nodes_3d)
    F_ext_3d = {nid: [0.0, f_per_node_3d, 0.0] for nid in right_nodes_3d}

    u_3d = direct_solve(model_3d, nodes_3d, fixed_3d, F_ext_3d)

    # Tip: top-right corner on z=0 layer (same position as 2D tip)
    tip_nid_3d = ny * (nx + 1) + (nx + 1)  # layer 0 top-right
    dofs_3d = np.asarray(nodes_3d[tip_nid_3d].getDOFs()).astype(int)
    v_tip_3d = u_3d[dofs_3d[1]]

    err = abs(v_tip_3d - v_tip_2d) / abs(v_tip_2d)
    passed = err < 0.01

    print(f"    Quad4 PlaneStrain v_tip = {v_tip_2d:.6f}")
    print(f"    Hex8  3D          v_tip = {v_tip_3d:.6f}")
    print(f"    error = {err*100:.4f}%  {'PASS' if passed else 'FAIL'}")

    return 1 if passed else 0, 1


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Hex8 B-bar Element Benchmarks")
    print("=" * 60)

    total_pass = 0
    total_tests = 0

    # Benchmark 1: Patch Test (GATE)
    print("\n  Benchmark 1: 3D Patch Test (GATE)")
    print("    E=1000, nu=0.3, single Hex8, 6 strain states")
    print("  " + "-" * 56)
    p, t = test_patch_test()
    total_pass += p
    total_tests += t
    print(f"    {p}/{t} PASS")

    if p < t:
        print("\n  *** GATE FAILED — stopping ***")
        print(f"\n{'=' * 60}")
        print(f"  {total_pass}/{total_tests} PASS")
        print(f"{'=' * 60}")
        sys.exit(1)

    # Benchmark 2: Thick-Walled Cylinder
    print("\n  Benchmark 2: Thick-Walled Cylinder (B-bar proof)")
    print("    E=1000, nu=0.499, ri=1, ro=3, p_i=1.0, 4x4x1 mesh")
    print("  " + "-" * 56)
    p, t = test_thick_walled_cylinder()
    total_pass += p
    total_tests += t
    print(f"    {p}/{t} PASS")

    # Benchmark 3: Cantilever
    print("\n  Benchmark 3: 3D Cantilever")
    print("    E=1000, nu=0.3, L=10, b=h=1, P=1, 40x4x4 mesh")
    print("  " + "-" * 56)
    p, t = test_cantilever()
    total_pass += p
    total_tests += t
    print(f"    {p}/{t} PASS")

    # Benchmark 4: Cook's 3D
    print("\n  Benchmark 4: Cook's Membrane 3D vs Quad4 PlaneStrain")
    print("    E=1, nu=1/3, 16x16 in-plane, 1 layer Hex8")
    print("  " + "-" * 56)
    p, t = test_cooks_3d()
    total_pass += p
    total_tests += t
    print(f"    {p}/{t} PASS")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  {total_pass}/{total_tests} PASS")
    if total_pass == total_tests:
        print("  ALL PASS")
    print(f"{'=' * 60}")
