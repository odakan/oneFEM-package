##-----------------------------------------------------------------------##
#                                                                         #
#        #--oneFEM--#: One FEM software in a galaxy far far away          #
#                                                                         #
#                   Computational Mechanics 2022                          #
#                       University of Pavia                               #
#               Written by: Onur Deniz AKAN, IUSS Pavia                   #
#                         15 January 2022                                 #
#                                                                         #
##-----------------------------------------------------------------------##
# 
# Author: Onur Deniz Akan
# Date: 08/02/2025
# Version: 0.1
#
#Truss main object definition
#   definition of the truss element object

from ..main import Element
from ..main import Section
from ..main import Node
# import data structures
from ...._systools.data import Vector, Matrix
# import precision math tools
from ...._systools.math_tools import _is_close

class Truss(Element):
    """
    Truss element object
    """
    def __init__(self, ele_id, nodes=[Node(), Node()], section=Section()):
        """
        Constructor for Truss element object
        """
        # initialize the element
        super().__init__(ele_id)

        # check if the nodes are defined
        if len(nodes) != 2:
            raise ValueError("Truss._domain(): 2 Nodes must be defined for element {}".format(self.__ID))
        
        if nodes[0] is None or nodes[1] is None:
            raise ValueError("Truss._domain(): Nodes are not defined for element {}".format(self.__ID))

        if not isinstance(nodes[0], Node):
            raise ValueError("Truss._domain(): i Node is not defined for element {}".format(self.__ID))

        if not isinstance(nodes[1], Node):
            raise ValueError("Truss._domain(): j Node is not defined for element {}".format(self.__ID))
            
        self.__nodes = nodes # set nodes of the element

        # check the node dimensions
        if self.__nodes[0].__nD != self.__nodes[1].__nD:
            raise ValueError("Truss._domain(): Nodes do not have the same number of dimensions for element {}".format(self.__ID))
        else:
            if self.__nodes[0].__nD == 2 or self.__nodes[0].__nD == 3:
                self.__nD = self.__nodes[0].__nD
            else:
                raise ValueError("Truss._domain(): unsupported problem dimension for element {}".format(self.__ID))

        # initialize list of number of degrees of freedom of each node
        self.__nDOF = Vector([self.__nodes[0].__nD, self.__nodes[1].__nD], dtype=int)

        # check if the section is defined
        if section is None:
            raise ValueError("Truss._domain(): Section is not defined for element {}".format(self.__ID))
        
        if not isinstance(section, Section):
            raise ValueError("Truss._domain(): Section is not defined for element {}".format(self.__ID))

        self.__section = section.copy() # store a copy of the section for the element

        # element properties
        self.__L = None # length of the element
        self.__n = Vector(shape=3) # local axis of the element
        self.__R3 = Matrix(shape=[3,3]) # rotation matrix (3-by-3)


    def _domain(self):
        """
        Compute the element domain
        """

        # Compute the element length
        self.__L = ((self.__nodes[1].__coords[0] - self.__nodes[0].__coords[0])**2 + 
                    (self.__nodes[1].__coords[1] - self.__nodes[0].__coords[1])**2 +
                    (self.__nodes[1].__coords[2] - self.__nodes[0].__coords[2])**2)**0.5
        
        if _is_close(self.__L, 0.0):
            raise ValueError("Truss._domain(): Element length is zero for element {}".format(self.__ID))
        
        # Compute the local axis of the element
        dx = self.__nodes[1].__coords[0] - self.__nodes[0].__coords[0]
        dy = self.__nodes[1].__coords[1] - self.__nodes[0].__coords[1]
        dz = self.__nodes[1].__coords[2] - self.__nodes[0].__coords[2]
        self.__n = Vector([dx, dy, dz], dtype=float) / self.__L

        # Compute the rotation matrix
        Y_orient = Vector([0, 1, 0])
        Z_orient = Vector([0, 0, 1])
        if  self.__n._is_parallel(Y_orient): # use global Y
            Z_orient =  self.__n._cross(Y_orient)
            Z_orient._normalize()
            Y_orient = self.__n._cross(Z_orient)
            Y_orient._normalize()
        else:
            Y_orient =  self.__n._cross(Z_orient)
            Y_orient._normalize()
            Z_orient = self.__n._cross(Y_orient)
            Z_orient._normalize()
        
        self.__R3 = Matrix([self.__n, Y_orient, Z_orient])
        

    def _update(self):
        return super()._update()
    

    def _commit(self):
        return super()._commit()
    
    def _revert(self):
        return super()._revert()
    
    def __str__(self):
        return "Truss element {}".format(self.__ID)
    
    def __repr__(self):
        return "Truss element {}".format(self.__ID)
    

    # ELEMENT API
    def copy(self):
        return Truss()


