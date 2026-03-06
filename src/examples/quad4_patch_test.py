##-----------------------------------------------------------------------##
#  Benchmark: Quad4 Patch Test (MacNeal-Harder 4-element patch)
#
#  Validates Quad4 + LinearContinuumKinematics + ElasticIsotropic (PlaneStress)
#  on an irregular 9-node, 4-element patch. Prescribes boundary displacements
#  consistent with 3 constant stress states. Interior node 9 is free.
#
#  Reference: MacNeal & Harder (1985)
#
#  Pass criterion: stress at ALL GPs matches prescribed constant stress
#                  to machine precision (rel error < 1e-10)
##-----------------------------------------------------------------------##

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from oneFEM.model import Domain
from oneFEM.model.node import Node22
from oneFEM.model.element.continuum.quad4 import Quad4
from oneFEM.model.material.nD.elastic_isotropic import ElasticIsotropic
from oneFEM.model.pattern import Plain as PlainPattern
from oneFEM.model.tseries import Constant
from oneFEM._systools.data import Vector

# ---- Patch geometry ----
node_coords = {
    1: (0.000, 0.000),
    2: (1.000, 0.000),
    3: (1.000, 1.000),
    4: (0.000, 1.000),
    5: (0.500, 0.000),
    6: (1.000, 0.500),
    7: (0.500, 1.000),
    8: (0.000, 0.500),
    9: (0.240, 0.220),  # irregular interior node
}

# 4 Quad4 elements (CCW node ordering, node 9 is shared interior)
elem_connectivity = {
    1: [1, 5, 9, 8],
    2: [5, 2, 6, 9],
    3: [9, 6, 3, 7],
    4: [8, 9, 7, 4],
}

boundary_nodes = [1, 2, 3, 4, 5, 6, 7, 8]
interior_node = 9

E_val = 1.0
nu_val = 0.25
G_val = E_val / (2.0 * (1.0 + nu_val))  # = 0.4


def analytical_disp(load_case, x, y):
    if load_case == 1:  # pure sigma_x = 1
        return (x / E_val, -nu_val * y / E_val)
    elif load_case == 2:  # pure sigma_y = 1
        return (-nu_val * x / E_val, y / E_val)
    elif load_case == 3:  # pure tau_xy = 1
        return (y / (2.0 * G_val), x / (2.0 * G_val))


def expected_stress(load_case):
    if load_case == 1:
        return np.array([1.0, 0.0, 0.0])
    elif load_case == 2:
        return np.array([0.0, 1.0, 0.0])
    elif load_case == 3:
        return np.array([0.0, 0.0, 1.0])


def build_model():
    """Create model with nodes and elements, run _domain()."""
    model = Domain()
    nodes = {}
    for nid, (x, y) in node_coords.items():
        nd = Node22(nid, coord=[x, y])
        nodes[nid] = nd
        model.add(nd)

    mat = ElasticIsotropic(1, E_val, nu_val, type='PlaneStress')

    elements = {}
    for eid, conn in elem_connectivity.items():
        elem_nodes = [nodes[c] for c in conn]
        elem = Quad4(eid, elem_nodes, mat, thickness=1.0)
        elements[eid] = elem
        model.add(elem)

    # Need at least one pattern for Domain to work
    ts = Constant(1, factor=1.0)
    pat = PlainPattern(1, ts)
    model.add(pat)

    # Fix boundary nodes (no actual forces needed)
    for nid in boundary_nodes:
        nodes[nid].setFix([True, True])

    # Initialize DOF numbering and element geometry
    model._domain()

    return model, nodes, elements


def run_patch_test(load_case):
    """Impose analytical displacements on ALL nodes, check GP stresses."""
    model, nodes, elements = build_model()

    # Set analytical displacements on all nodes
    for nid, (x, y) in node_coords.items():
        u, v = analytical_disp(load_case, x, y)
        nd = nodes[nid]
        nd._update(Vector([0.0, 0.0]), Vector([u, v]))
        nd._commitState()

    # Update elements (computes strains/stresses at GPs)
    for elem in elements.values():
        elem._update()

    # Check stress at all GPs
    sig_expected = expected_stress(load_case)
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


def run_interior_node_test(load_case):
    """Fix boundary nodes, solve for interior node, compare with analytical."""
    model, nodes, elements = build_model()

    # Assemble global stiffness
    model._assemble()
    K = np.asarray(model.K)
    n = model.nDOF

    # DOF indices
    int_nd = nodes[interior_node]
    free_dofs = list(np.asarray(int_nd.getDOFs()).astype(int))

    fixed_dofs = []
    for nid in boundary_nodes:
        fixed_dofs.extend(list(np.asarray(nodes[nid].getDOFs()).astype(int)))

    # Prescribed displacement vector
    u_prescribed = np.zeros(n)
    for nid in boundary_nodes:
        x, y = node_coords[nid]
        u, v = analytical_disp(load_case, x, y)
        dofs = list(np.asarray(nodes[nid].getDOFs()).astype(int))
        u_prescribed[dofs[0]] = u
        u_prescribed[dofs[1]] = v

    # Partition: K_ff * u_f = -K_fp * u_p
    uu = np.array(free_dofs)
    pp = np.array(fixed_dofs)

    K_ff = K[np.ix_(uu, uu)]
    K_fp = K[np.ix_(uu, pp)]
    u_p = u_prescribed[pp]

    rhs = -K_fp @ u_p
    u_f = np.linalg.solve(K_ff, rhs)

    # Compare with analytical
    x9, y9 = node_coords[interior_node]
    u_exact, v_exact = analytical_disp(load_case, x9, y9)

    err_u = abs(u_f[0] - u_exact) / max(abs(u_exact), 1e-15)
    err_v = abs(u_f[1] - v_exact) / max(abs(v_exact), 1e-15)
    max_err = max(err_u, err_v)

    return max_err < 1e-10, max_err, (u_f[0], u_f[1]), (u_exact, v_exact)


if __name__ == "__main__":
    print("=" * 60)
    print("Quad4 Patch Test")
    print("  E = 1.0, nu = 0.25, PlaneStress, t = 1.0")
    print("  4 Quad4 elements, 9 nodes (1 irregular interior)")
    print("=" * 60)

    results = []
    lc_names = {1: "pure sigma_x", 2: "pure sigma_y", 3: "pure tau_xy"}

    for lc in [1, 2, 3]:
        print(f"\n  Load case {lc} ({lc_names[lc]}):")

        passed, max_err = run_patch_test(lc)
        status = "PASS" if passed else "FAIL"
        print(f"    GP stress:       {status}  (max err = {max_err:.2e})")
        results.append(passed)

        passed2, max_err2, (uc, vc), (ue, ve) = run_interior_node_test(lc)
        status2 = "PASS" if passed2 else "FAIL"
        print(f"    Interior node:   {status2}  (max err = {max_err2:.2e})")
        print(f"      computed: u={uc:.10e}, v={vc:.10e}")
        print(f"      exact:    u={ue:.10e}, v={ve:.10e}")
        results.append(passed2)

    n_pass = sum(results)
    n_total = len(results)
    print(f"\n{'=' * 60}")
    print(f"  {n_pass}/{n_total} PASS")
    if n_pass == n_total:
        print("  ALL PASS")
    print(f"{'=' * 60}")
