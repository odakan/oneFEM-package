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
tr1 = Truss(1, nodes=[nd1, nd2], section=steel_box)
tr2 = Truss(2, nodes=[nd1, nd2], section=steel_box)
tr3 = Truss(3, nodes=[nd1, nd2], section=steel_box)
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
alg = Linear()
const = PlainConstraints()
numb = PlainNumberer()
syst = FullGeneral()
integ = LoadControl()
ctest = NormUnbalance()
static_push = Analysis(1, algorithm=alg, constraints=const, 
                        integrator=integ, system=syst, test=ctest)

# assign simulation manager and run
run_1 = SimulationManager(simple_triangle, static_push, dt=0.0)
run_1.analyze(10, dt=0.1)