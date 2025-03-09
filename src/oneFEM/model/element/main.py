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
#ELEMENT main object definition
#   definition of the element object

from ..node.main import Node
from .section.main import Section
from ..material.main import Material
from ..._systools.data import Vector, Matrix

class Element(object):
    def __init__(self, ele_id=-1):
        """Constructor for ELEMENT class."""
        self._ID = ele_id                      # Element ID
        self._nD = 0                           # Number of dimensions
        self._nDOF = Vector(dtype=int)         # List of number of degrees of freedom of ech node
        self._nodes = [Node(), Node()]         # List of nodes
        self._k = Matrix()                     # Element stiffness matrix
        self._f = Vector()                     # Element force vector
        self._material = Material()            # Associated material
        self._section = Section()              # Associated section

    def _domain(self):
        """Compute the element domain."""
        raise NotImplementedError("ELEMENT._domain(): method must be implemented by subclasses.")

    def _commit(self):
        """Commit the element state."""
        raise NotImplementedError("ELEMENT._commit(): method must be implemented by subclasses.")

    def _revert(self):
        """Revert the element state to the last commit."""
        raise NotImplementedError("ELEMENT._revert(): method  must be implemented by subclasses.")
    
    def _update(self):
        """Update the element state."""
        raise NotImplementedError("ELEMENT._update(): method  must be implemented by subclasses.")
    

    # ELEMENT API
    def copy(self):
        """Return a copy of the element."""
        raise NotImplementedError("ELEMENT.copy(): method  must be implemented by subclasses.")
    
    def getNodes(self):
        """Return the list nodes of the element."""
        return self._nodes
    
    def getDOFs(self):
        """Return the dof vector corresponding to the list of nodes."""
        return Vector([dof for nd in self._nodes for dof in nd.getDOFs()], dtype=int)
    
    def getStiffness(self):
        """Return the stiffness matrix of the element."""
        return self._k
    
    def getForce(self):
        """Return the force vector of the element."""
        return self._f

    def getID(self):
        """Return the element ID."""
        return self._ID
    
    def getND(self):
        """Return the number of dimensions of the element."""
        return self._nD
    
    def getNDOF(self):
        """Return the number of degrees of freedom of the element."""
        return self._nDOF
