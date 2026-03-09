"""Run curved cantilever: TL + incompatible, 48x4x4, 40 steps."""
import sys, os
sys.stdout.reconfigure(line_buffering=True)
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

N_ARC, N_RAD, N_WIDTH = 48, 4, 4
N_STEPS = 40
MAX_ITER = 100

print(f"Curved cantilever: TL + incompatible, {N_ARC}x{N_RAD}x{N_WIDTH}, {N_STEPS} steps, maxIter={MAX_ITER}")
print(f"Reference: ux={BATHE_REF['ux']}, uy={BATHE_REF['uy']}, uz={BATHE_REF['uz']}")
print("=" * 80)

node_coords, elem_conn, free_nids, fixed_nids = build_curved_mesh(N_ARC, N_RAD, N_WIDTH)
n_elem = N_ARC * N_RAD * N_WIDTH
n_nodes = (N_ARC + 1) * (N_RAD + 1) * (N_WIDTH + 1)
n_dof = n_nodes * 3
print(f"Mesh: {n_elem} elements, {n_nodes} nodes, {n_dof} DOFs")

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

dt_val = 1.0 / N_STEPS
alg = Newton(1, tangent='current')
ctest = NormUnbalance(1, tol=1e-5, maxIter=MAX_ITER)
analysis = Analysis(algorithm=alg, integrator=LoadControl(1), test=ctest)

t0 = _time.perf_counter()
analysis._analyze(model, nSteps=0, dt=0.0)
integrator_obj = analysis._solution_integrator

max_iters = 0
for step in range(N_STEPS):
    assembly_time = analysis._time + dt_val
    analysis._assembleF(model, time=assembly_time)
    integrator_obj.newStep(model, dt_val, analysis._time)
    try:
        analysis._solution_algorithm.solve(
            model, analysis.uu, analysis.pp,
            integrator_obj, analysis._assembly_system, ctest)
        iters = ctest._currentIter
        if iters > max_iters:
            max_iters = iters
        print(f"  Step {step+1}/{N_STEPS}: converged in {iters} iterations")
    except Exception as e:
        wall = _time.perf_counter() - t0
        print(f"  Step {step+1}/{N_STEPS}: FAILED ({e})")
        print(f"\nFailed at step {step+1}/{N_STEPS} after {wall:.1f}s, max iterations used: {max_iters}")
        # Report last converged displacements
        ux_vals, uy_vals, uz_vals = [], [], []
        for nid in free_nids:
            u = nodes[nid]._getCommitDisp()
            ux_vals.append(float(u[0]))
            uy_vals.append(float(u[1]))
            uz_vals.append(float(u[2]))
        ux, uy, uz = np.mean(ux_vals), np.mean(uy_vals), np.mean(uz_vals)
        print(f"Last converged tip: ux={ux:.4f}, uy={uy:.4f}, uz={uz:.4f}")
        sys.exit(1)

    integrator_obj.commit(model)
    analysis._time += dt_val

wall = _time.perf_counter() - t0

ux_vals, uy_vals, uz_vals = [], [], []
for nid in free_nids:
    u = nodes[nid]._getCommitDisp()
    ux_vals.append(float(u[0]))
    uy_vals.append(float(u[1]))
    uz_vals.append(float(u[2]))
ux, uy, uz = np.mean(ux_vals), np.mean(uy_vals), np.mean(uz_vals)

ux_err = abs(ux - BATHE_REF['ux']) / abs(BATHE_REF['ux']) * 100
uy_err = abs(uy - BATHE_REF['uy']) / abs(BATHE_REF['uy']) * 100
uz_err = abs(uz - BATHE_REF['uz']) / abs(BATHE_REF['uz']) * 100

print(f"\nCompleted {N_STEPS}/{N_STEPS} steps in {wall:.1f}s (max Newton iter: {max_iters})")
print(f"Tip displacements:")
print(f"  ux = {ux:>10.4f}  (ref {BATHE_REF['ux']:>8.1f}, err {ux_err:.1f}%)")
print(f"  uy = {uy:>10.4f}  (ref {BATHE_REF['uy']:>8.1f}, err {uy_err:.1f}%)")
print(f"  uz = {uz:>10.4f}  (ref {BATHE_REF['uz']:>8.1f}, err {uz_err:.1f}%)")
