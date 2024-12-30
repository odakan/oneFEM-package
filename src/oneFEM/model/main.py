##-----------------------------------------------------------------------##
#                                                                         #
#        #--oneFEM--#: One FEM software in a galaxy far far away          #
#                                                                         #
#                   Computational Mechanics 2022                          #
#                       University of Pavia                               #
#               Written by: Onur Deniz AKAN, IUSS Pavia                   #
#                         17 January 2022                                 #
#                                                                         #
##-----------------------------------------------------------------------##
#DOMAIN main object definition
#   Domain holds the model objects and the functions operate over them

from .node.main import Node
from .tseries.main import TSeries
from .pattern.main import Pattern
from .element.main import Element
from .material.main import Material
from .constraint.main import Constraint

class Domain(object):
    def __init__(self, nD=0):
        # Model objects 
        # (features name mangling due to __name -> ex: obj._Model__nodes)
        self.__nodes = []       # Nodes (list)
        self.__patterns = []    # Load Pattern (list)
        self.__elements = []    # Elements (list)
        self.__materials = []   # Materials (list)
        self.__time_series = [] # Load time series (list)
        self.__constraints = [] # Constraints (list)

        # Model variables
        self.__nDim = int(0)  # model dimension 2D or 3D
        self.__nDof = int(0)  # default number of node dofs

        # Model statistics
        self.__nNds = 0     # total number of nodes
        self.__nDOF = 0     # total number of d.o.f.s
        self.__nElems = 0   # total number of elements
        self.__nCons = 0    # total number of constraints
        self.__nTS = 0      # total number of time serires
        self.__nPtrns = 0   # total number of loading patterns
        self.__nMats = 0    # totam number of root (user) materials

        # Initialize global matrices and vectors
        self._K = 0  # Global tangent stiffness matrix
        self._C = 0  # Damping matrix
        self._M = 0  # Mass matrix
        self._u = 0  # Displacement increment vector
        self._v = 0  # Velocity increment vector
        self._a = 0  # Acceleration increment vector
        self._F = 0  # Force increment vector

        # Initialize the DOF counter
        self._dof_counter(0)


    def _domain(self):
        # Compute the initial state of the domain
        self.nNds = len(self.nodes)
        if self.nNds < 1:
            raise ValueError("MAIN: No nodes in the domain!")

        self.nElems = len(self.elements)
        if self.nElems < 1:
            raise ValueError("MAIN: No elements in the domain!")

        # Count nDOFs
        self.nDOF = self.nodes.dof_numberer()

        # Call domain for all elements
        for element in self.elements:
            element.domain()


    def _assemble(self):
        # Assemble stiffness matrix and force vector
        self.K = [[0] * self.nDOF for _ in range(self.nDOF)]
        self.F = [0] * self.nDOF
        self.u = [0] * self.nDOF

        # Assemble K and F for each element
        for element in self.elements:
            gd = []
            idx = 0
            for node in element.nds:
                ld = list(range(idx, idx + node.nDOF))
                idx = ld[-1] + 1
                gd.extend(node.dofs)
            for i in gd:
                for j in gd:
                    self.K[i][j] += element.k
            for i in gd:
                self.F[i] += element.f

        # Assemble F and u for each node
        for node in self.nodes:
            gd = node.dofs
            for i in gd:
                self.F[i] += node.force
                self.u[i] += node.imposed


    def _update(self):
        # Placeholder for domain state update
        pass


    def _commit(self):
        # Commit domain state
        for node in self.nodes:
            d = node.dofs
            if len(self.v) > 1:  # If transient
                node.commit(self.u[d], self.v[d], self.a[d])
            else:  # If static
                node.commit(self.u[d])


    def _revert(self):
        # Placeholder for reverting domain state
        pass


    def _record(self):
        # Placeholder for recording domain state
        pass


    def _add(self, command):
        # Add an object to the domain based on the command
        if command[0] == "element":
            ele = Element()
            if self.elements[-1].ID == -1:
                self.elements[-1] = ele.add_element(command, self.nodes)
            else:
                self.elements.append(ele.add_element(command, self.nodes))
        elif command[0] == "material":
            mat = Material()
            if self.materials[-1].ID == -1:
                self.materials[-1] = mat.add_material(command)
            else:
                self.materials.append(mat.add_material(command))
        elif command[0] == "node":
            nd = Node()
            if self.nodes.ID == -1:
                self.nodes = nd.add_node(command)
            else:
                self.nodes.append(nd.add_node(command))
        elif command[0] == "pattern":
            rec = Pattern()
            if self.patterns[-1].ID == -1:
                self.patterns[-1] = rec.add_pattern(command)
            else:
                self.recorders.append(rec.add_pattern(command))
        elif command[0] == "timeseries":
            ts = TSeries()
            if self.time_series[-1].ID == -1:
                self.time_series[-1] = ts.add_tseries(command)
            else:
                self.time_series.append(ts.add_tseries(command))
        else:
            print("Domain: Unknown command!")


    def _remove(self):
        pass