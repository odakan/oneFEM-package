# create a triangle truss model
# in kN and m
#
#         1 kN -->  o node 3
#                  /| [2, 0, 2]
#                 / |
#                /  |
#               /   |
#              /    |
#             /     |
#            /      |
#           /       |
#          /        |
#         /         |
#        /          |
#      >o-----------o node 2
#       ^           ^ [2, 0, 0]
#    node 1
#    [0, 0, 0]
#

import numpy as np

# import domain
from oneFEM.model import Domain
# modeling tools
from oneFEM.model.element.truss import Truss
from oneFEM.model.node import Node36
from oneFEM.model.element.section import Rectangular
from oneFEM.model.material.uniaxial import Elastic
from oneFEM.model.tseries import Constant
from oneFEM.model.pattern import Plain as PlainPattern
# result tools
from oneFEM.output.recorder import NodeRecorder
from oneFEM.output.recorder import ElementRecorder
# import analysis
from oneFEM.analysis import Analysis
# analysis tools
from oneFEM.analysis.algorithm import Linear
from oneFEM.analysis.constraints import Plain as PlainConstraints
from oneFEM.analysis.numberer import Plain as PlainNumberer
from oneFEM.analysis.system import FullGeneral
from oneFEM.analysis.integrator import LoadControl
from oneFEM.analysis.test import NormUnbalance
# import simulation manager
from oneFEM import SimulationManager

# initialize the domain
simple_triangle = Domain(nD=3)

# create nodes and add to domain
nd1 = Node36(1, coord=[0.0, 0.0, 0.0], fix=[1, 1, 1, 1, 1, 1])
nd2 = Node36(2, coord=[2.0, 0.0, 0.0], fix=[0, 1, 1, 1, 1, 1])
nd3 = Node36(3, coord=[2.0, 0.0, 2.0], fix=[0, 1, 0, 1, 1, 1])
simple_triangle.add(nd1, nd2, nd3)

# create elastic uniaxial material and add to domain
steel_fiber = Elastic(1, E=2e9)

# create section and add to domain
steel_box = Rectangular(1, mat=steel_fiber, h=0.2, w=0.1)

# create the truss elements and add to domain
tr1 = Truss(1, nodes=[nd1, nd2], section=steel_box)  # horizontal (node1→node2)
tr2 = Truss(2, nodes=[nd2, nd3], section=steel_box)  # vertical   (node2→node3)
tr3 = Truss(3, nodes=[nd1, nd3], section=steel_box)  # diagonal   (node1→node3)
simple_triangle.add(tr1, tr2, tr3)

# set time series
t_const = Constant(1, factor=1.0)

# create horizontal load and add to domain
push = [
    # [nodeID, x, y, z, rx, ry, rz]
    [3, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00]
]
horizontal_push = PlainPattern(1, tseries=t_const, load=push)
simple_triangle.add(horizontal_push)

# create recorders
top_nd_disp = NodeRecorder(1, nd3, dofs=[1], results=['displacement'])
bottom_nd_react = NodeRecorder(2, nd1, dofs=[1], results=['reaction'])
bottom_ele_strain = ElementRecorder(3, tr1, results=['strain'])
simple_triangle.add(top_nd_disp, bottom_nd_react, bottom_ele_strain)

# create the analysis
alg = Linear(1)
const = PlainConstraints(1)
numb = PlainNumberer(1)
syst = FullGeneral(1)
integ = LoadControl(1)
ctest = NormUnbalance(1)
static_push = Analysis(1, algorithm=alg, constraints=const,
                       integrator=integ, system=syst, test=ctest)

# assign simulation manager and run
run_1 = SimulationManager(1, simple_triangle, static_push, dt=0.0)
run_1.analyze(1, dt=0.1)

# ---- Analytical verification ----
# Material: E = 2e9 Pa, Section: A = 0.2 * 0.1 = 0.02 m^2
# EA = 2e9 * 0.02 = 4e7 N
#
# Element 1 (horizontal): nd1→nd2, L=2, n=[1,0,0]
# Element 2 (vertical):   nd2→nd3, L=2, n=[0,0,1]
# Element 3 (diagonal):   nd1→nd3, L=2√2, n=[1/√2, 0, 1/√2]
#
# Free DOFs (0-indexed):
#   nd2: DOF 6  (x-translation)
#   nd3: DOF 12 (x-translation), DOF 14 (z-translation)
#
# We build K_ff (3x3) and F_f (3x1) for these 3 free DOFs:
#   DOF indices: [6, 12, 14]

EA = 2e9 * 0.02  # = 4e7
L1 = 2.0  # element 1 length
L2 = 2.0  # element 2 length
L3 = 2.0 * np.sqrt(2.0)  # element 3 length
c3 = 1.0 / np.sqrt(2.0)  # direction cosine for diagonal

# Build 3x3 free-DOF stiffness (rows/cols correspond to dofs [6, 12, 14])
K_ff = np.zeros((3, 3))

# Element 1: connects nd1(fixed)→nd2(free DOF 6)
# k_e1 = EA/L1 * [1 0 0; 0 0 0; 0 0 0] in full space, only DOF 6 contributes
# K_ff[0,0] += EA/L1
K_ff[0, 0] += EA / L1

# Element 2: connects nd2(free DOF 6 doesn't couple here - n=[0,0,1])→nd3(free DOFs 12,14)
# For nd2: translational x-DOF=6, z-DOF=8; fix[8]=1 so DOF 8 is fixed
# For nd3: translational x-DOF=12, z-DOF=14
# n2 = [0, 0, 1], nn = [[0,0,0],[0,0,0],[0,0,1]]
# Element 2 only couples z-DOFs: DOF 8 (nd2-z, fixed) and DOF 14 (nd3-z, free)
# k_jj block, z-z component: EA/L2
K_ff[2, 2] += EA / L2

# Element 3: connects nd1(fixed)→nd3(free DOFs 12, 14)
# n3 = [c3, 0, c3], nn = c3^2 * [[1,0,1],[0,0,0],[1,0,1]]
# k_jj block (x,z for nd3): factor * nn
factor3 = EA / L3
nn_xx = c3 * c3
nn_xz = c3 * c3
nn_zz = c3 * c3
# DOF 12 (x) and DOF 14 (z) contributions from k_jj of element 3
K_ff[1, 1] += factor3 * nn_xx  # x-x
K_ff[1, 2] += factor3 * nn_xz  # x-z
K_ff[2, 1] += factor3 * nn_xz  # z-x
K_ff[2, 2] += factor3 * nn_zz  # z-z

# Force vector: 1 kN in x on node 3 → DOF 12
F_ff = np.array([0.0, 1.0, 0.0])

# Solve
u_analytical = np.linalg.solve(K_ff, F_ff)

# Get numerical results
u_nd2_x = nd2._getCommitDisp()[0]  # x-displacement of node 2
u_nd3_x = nd3._getCommitDisp()[0]  # x-displacement of node 3
u_nd3_z = nd3._getCommitDisp()[2]  # z-displacement of node 3

print("\n========== TRUSS BENCHMARK RESULTS ==========")
print(f"  E = {2e9:.2e} Pa,  A = {0.02} m^2,  EA = {EA:.2e} N")
print(f"  Applied load: Fx = 1.0 kN at node 3")
print()
print("  Free DOF displacements:")
print(f"    nd2 x-disp: numerical = {u_nd2_x:.10e},  analytical = {u_analytical[0]:.10e}")
print(f"    nd3 x-disp: numerical = {u_nd3_x:.10e},  analytical = {u_analytical[1]:.10e}")
print(f"    nd3 z-disp: numerical = {u_nd3_z:.10e},  analytical = {u_analytical[2]:.10e}")
print()

tol = 1e-10
pass_nd2_x = abs(u_nd2_x - u_analytical[0]) < tol
pass_nd3_x = abs(u_nd3_x - u_analytical[1]) < tol
pass_nd3_z = abs(u_nd3_z - u_analytical[2]) < tol
all_pass = pass_nd2_x and pass_nd3_x and pass_nd3_z

print(f"  nd2 x-disp: {'PASS' if pass_nd2_x else 'FAIL'}")
print(f"  nd3 x-disp: {'PASS' if pass_nd3_x else 'FAIL'}")
print(f"  nd3 z-disp: {'PASS' if pass_nd3_z else 'FAIL'}")
print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
print("=============================================\n")

# print recorder data
print("Recorder data:")
print(f"  Node 3 displacement history: {[v.tolist() if hasattr(v, 'tolist') else v for v in top_nd_disp.data['displacement']]}")
print(f"  Node 1 reaction history:     {[v.tolist() if hasattr(v, 'tolist') else v for v in bottom_nd_react.data['reaction']]}")
print(f"  Element 1 strain history:    {bottom_ele_strain.data['strain']}")
