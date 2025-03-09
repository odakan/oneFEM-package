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
from ..output.recorder.main import Recorder

class Domain(object):
    def __init__(self, nD=0):
        # Model objects 
        # (features name mangling due to __name -> ex: obj._Model__nodes)
        self.__nodes = []       # Nodes (list)
        self.__patterns = []    # Load Pattern (list)
        self.__elements = []    # Elements (list)
        #self.__materials = []   # Materials (list)
        #self.__time_series = [] # Load time series (list)
        self.__constraints = [] # Constraints (list)
        self.__recorders = []   # Recorders (list)

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
        self.__nMats = 0    # total number of root (user) materials
        self.__nRecs = 0    # total number of recorders

        # Initialize global matrices and vectors
        self._K = 0  # Global tangent stiffness matrix
        self._C = 0  # Damping matrix
        self._M = 0  # Mass matrix
        self._u = 0  # Displacement increment vector
        self._v = 0  # Velocity increment vector
        self._a = 0  # Acceleration increment vector
        self._F = 0  # Force increment vector

        ## Initialize the DOF counter
        #self._dof_counter(0)


    def _domain(self):
        # Compute the initial state of the domain
        self.nNds = len(self.__nodes)
        if self.nNds < 1:
            raise ValueError("Domain: No nodes in the domain!")

        self.nElems = len(self.__elements)
        if self.nElems < 1:
            raise ValueError("Domain: No elements in the domain!")

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
                    self.K[i][j] += element.__k
            for i in gd:
                self.F[i] += element.__f

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


    # Domain API methods
    def add(self, *objs):
        # Iterate over all passed objects
        for obj in objs:
            if isinstance(obj, Element):
                self.__elements.append(obj)

            #elif isinstance(obj, Material):
            #    self.__materials.append(obj)

            elif isinstance(obj, Node):
                self.__nodes.append(obj)

            elif isinstance(obj, Pattern):
                self.__patterns.append(obj)

            #elif isinstance(obj, TSeries):
            #    self.__time_series.append(obj)

            elif isinstance(obj, Constraint):
                self.__constraints.append(obj)

            elif isinstance(obj, Recorder):
                self.__recorders.append(obj)

            else:
                print(f"Domain: Unknown object! {obj}")


    def remove(self, obj):
        # Remove an object from the domain based on the command
        if isinstance(obj, Element):
            for element in self.__elements:
                if element == obj:

                    # check if recorder is connected to element


                    self.__elements.remove(element)
                else:
                    print("Domain: Element not found!")

        #elif isinstance(obj, Material):
        #    for material in self.__materials:
        #        if material == obj:
        #            self.__materials.remove(material)
        #        else:
        #            print("Domain: Material not found!")

        elif isinstance(obj, Node):
            for node in self.__nodes:
                if node == obj:

                    # check if element is connected to the node

                    # check if constraint is connected to the node

                    # check if pattern is connected to the node

                    # check if recorder is connected to the node

                    self.__nodes.remove(node)
                else:
                    print("Domain: Node not found!")

        elif isinstance(obj, Pattern):
            for pattern in self.__patterns:
                if pattern == obj:
                    self.__patterns.remove(pattern)
                else:
                    print("Domain: Pattern not found!")

        #elif isinstance(obj, TSeries):
        #    for tseries in self.__time_series:
        #        if tseries == obj:
        #            # check if pattern is connected to the time series
        #
        #            self.__time_series.remove(tseries)
        #        else:
        #            print("Domain: Time Series not found!")

        elif isinstance(obj, Constraint):
            for constraint in self.__constraints:
                if constraint == obj:
                    self.__constraints.remove(constraint)
                else:
                    print("Domain: Constraint not found!")

        elif isinstance(obj, Recorder):
            for recorder in self.__recorders:
                if recorder == obj:
                    self.__recorders.remove(recorder)
                else:
                    print("Domain: Recorder not found!")

        else:
            print("Domain: Unknown object!")