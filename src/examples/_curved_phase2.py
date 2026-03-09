"""Phase 2: TL + incompatible, cross-section refinement."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time as _time
import numpy as np
from hex8_curved_cantilever import (
    build_curved_mesh, BATHE_REF, E_VAL, NU_VAL, P_TOTAL,
)
from oneFEM.model import Domain
from oneFEM.model.node import Node33
from oneFEM.model.element.continuum.hex8 import Hex8
from oneFEM.model.kinematics.continuum.cauchy.total_lagrangian import TotalLagrangianContinuumKinematics
from oneFEM.model.material.nD.elastic_isotropic import ElasticIsotropic
from oneFEM.model.pattern import Plain as PlainPattern
from oneFEM.model.tseries import Linear as LinearTS
from oneFEM.analysis import Analysis
from oneFEM.analysis.algorithm.newton_raphson import Newton
from oneFEM.analysis.integrator import LoadControl
from oneFEM.analysis.test import NormUnbalance


def run_tl_incomp(n_arc, n_rad, n_width, nSteps=20):
    node_coords, elem_conn, free_nids, fixed_nids = build_curved_mesh(n_arc, n_rad, n_width)
    n_elem = n_arc * n_rad * n_width
    n_nodes = (n_arc + 1) * (n_rad + 1) * (n_width + 1)
    n_dof = n_nodes * 3

    model = Domain()
    nodes = {}
    for nid in sorted(node_coords.keys()):
        x, y, z = node_coords[nid]
        nd = Node33(nid, coord=[x, y, z])
        nodes[nid] = nd
        model.add(nd)
    for nid in fixed_nids:
        nodes[nid].setFix([True, True, True])

    mat = ElasticIsotropic(1, E_VAL, NU_VAL, type='3D')
    for eid in sorted(elem_conn.keys()):
        conn = elem_conn[eid]
        kin = TotalLagrangianContinuumKinematics()
        elem = Hex8(eid, [nodes[c] for c in conn], mat, kinematics=kin,
                    incompatible=True)
        model.add(elem)

    f_per_node = P_TOTAL / len(free_nids)
    ts = LinearTS(1, factor=1.0)
    load_list = [[nid, 0.0, 0.0, f_per_node] for nid in free_nids]
    pat = PlainPattern(1, tseries=ts, load=load_list)
    model.add(pat)

    dt_val = 1.0 / nSteps
    alg = Newton(1, tangent='current')
    ctest = NormUnbalance(1, tol=1e-5, maxIter=50)
    analysis = Analysis(algorithm=alg, integrator=LoadControl(1), test=ctest)

    t0 = _time.perf_counter()
    analysis._analyze(model, nSteps=0, dt=0.0)
    integrator_obj = analysis._solution_integrator

    max_iters = 0
    last_good_step = 0
    for step in range(nSteps):
        assembly_time = analysis._time + dt_val
        analysis._assembleF(model, time=assembly_time)
        integrator_obj.newStep(model, dt_val, analysis._time)
        try:
            analysis._solution_algorithm.solve(
                model, analysis.uu, analysis.pp,
                integrator_obj, analysis._assembly_system, ctest)
            iters = ctest._count
            if iters > max_iters:
                max_iters = iters
        except Exception:
            wall = _time.perf_counter() - t0
            return None, None, None, False, last_good_step, nSteps, n_elem, n_dof, wall, max_iters
        integrator_obj.commit(model)
        analysis._time += dt_val
        last_good_step = step + 1

    wall = _time.perf_counter() - t0

    ux_vals, uy_vals, uz_vals = [], [], []
    for nid in free_nids:
        u = nodes[nid]._getCommitDisp()
        ux_vals.append(float(u[0]))
        uy_vals.append(float(u[1]))
        uz_vals.append(float(u[2]))
    return (np.mean(ux_vals), np.mean(uy_vals), np.mean(uz_vals),
            True, nSteps, nSteps, n_elem, n_dof, wall, max_iters)


print("=" * 95)
print("Phase 2: TL + incompatible modes, cross-section refinement")
print(f"Reference: ux={BATHE_REF['ux']}, uy={BATHE_REF['uy']}, uz={BATHE_REF['uz']}")
print("=" * 95)
print(f"  {'Mesh':>10s}  {'Elems':>6s}  {'DOFs':>6s}  {'ux':>9s} {'err%':>6s}  "
      f"{'uy':>9s} {'err%':>6s}  {'uz':>9s} {'err%':>6s}  {'Status':>8s}  {'Time':>6s}  {'MaxIt':>5s}")
print("  " + "-" * 90)

meshes = [
    (12, 4, 4),
    (24, 4, 4),
    (24, 8, 8),
]

for n_arc, n_rad, n_width in meshes:
    label = f"{n_arc}x{n_rad}x{n_width}"
    ux, uy, uz, conv, last, total, n_elem, n_dof, wall, max_it = run_tl_incomp(n_arc, n_rad, n_width)
    if conv:
        ux_err = abs(ux - BATHE_REF['ux']) / abs(BATHE_REF['ux']) * 100
        uy_err = abs(uy - BATHE_REF['uy']) / abs(BATHE_REF['uy']) * 100
        uz_err = abs(uz - BATHE_REF['uz']) / abs(BATHE_REF['uz']) * 100
        print(f"  {label:>10s}  {n_elem:>6d}  {n_dof:>6d}  {ux:>9.4f} {ux_err:>5.1f}%  "
              f"{uy:>9.4f} {uy_err:>5.1f}%  {uz:>9.4f} {uz_err:>5.1f}%  "
              f"{'OK':>8s}  {wall:>5.1f}s  {max_it:>5d}")
    else:
        print(f"  {label:>10s}  {n_elem:>6d}  {n_dof:>6d}  {'—':>9s} {'—':>6s}  "
              f"{'—':>9s} {'—':>6s}  {'—':>9s} {'—':>6s}  "
              f"{'FAIL '+str(last)+'/'+str(total):>8s}  {wall:>5.1f}s  {max_it:>5d}")

print("  " + "-" * 90)
print(f"  {'Bathe ref':>10s}  {'—':>6s}  {'—':>6s}  {BATHE_REF['ux']:>9.4f} {'0.0':>5s}%  "
      f"{BATHE_REF['uy']:>9.4f} {'0.0':>5s}%  {BATHE_REF['uz']:>9.4f} {'0.0':>5s}%")
