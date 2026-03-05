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
import numpy as np

class Truss(Element):
    """
    Truss element object
    """
    def __init__(self, ele_id, nodes=None, section=None, rho=0.0, cMass=False):
        """
        Constructor for Truss element object
        """
        if nodes is None:
            nodes = [Node(), Node()]
        if section is None:
            section = Section()

        # initialize the element
        super().__init__(ele_id)

        # check if the nodes are defined
        if len(nodes) != 2:
            raise ValueError("Truss._domain(): 2 Nodes must be defined for element {}".format(self._ID))

        if nodes[0] is None or nodes[1] is None:
            raise ValueError("Truss._domain(): Nodes are not defined for element {}".format(self._ID))

        if not isinstance(nodes[0], Node):
            raise ValueError("Truss._domain(): i Node is not defined for element {}".format(self._ID))

        if not isinstance(nodes[1], Node):
            raise ValueError("Truss._domain(): j Node is not defined for element {}".format(self._ID))

        self._nodes = nodes # set nodes of the element

        # check the node dimensions
        if self._nodes[0].getND() != self._nodes[1].getND():
            raise ValueError("Truss._domain(): Nodes do not have the same number of dimensions for element {}".format(self._ID))
        else:
            if self._nodes[0].getND() == 2 or self._nodes[0].getND() == 3:
                self._nD = self._nodes[0].getND()
            else:
                raise ValueError("Truss._domain(): unsupported problem dimension for element {}".format(self._ID))

        # initialize list of number of degrees of freedom of each node
        self._nDOF = Vector([self._nodes[0].getNDOF(), self._nodes[1].getNDOF()], dtype=int)

        # check if the section is defined
        if section is None:
            raise ValueError("Truss._domain(): Section is not defined for element {}".format(self._ID))

        if not isinstance(section, Section):
            raise ValueError("Truss._domain(): Section is not defined for element {}".format(self._ID))

        self._section = section.copy() # store a copy of the section for the element

        # element properties
        self.__L = None # length of the element
        self.__n = Vector(shape=3) # direction cosine vector
        self.__nDOF_total = self._nodes[0].getNDOF() + self._nodes[1].getNDOF()
        self.__rho = rho     # mass per unit length
        self.__cMass = cMass # True = consistent mass, False = lumped mass


    def _domain(self):
        """
        Compute the element domain: length, direction cosines, stiffness matrix
        """
        nD = self._nD

        # Get node coordinates
        c0 = self._nodes[0]._coord
        c1 = self._nodes[1]._coord

        # Compute element length
        dx = [c1[i] - c0[i] for i in range(nD)]
        L2 = sum(d*d for d in dx)
        self.__L = L2 ** 0.5

        if _is_close(self.__L, 0.0):
            raise ValueError("Truss._domain(): Element length is zero for element {}".format(self._ID))

        # Compute direction cosine vector (only translational DOFs matter for truss)
        self.__n = Vector([d / self.__L for d in dx])

        # Get EA from section
        EA = self._section.getTangent()

        # Build element stiffness matrix
        # For a truss: ke = (EA/L) * [nn, -nn; -nn, nn] scattered into full DOF blocks
        nDOF_i = self._nodes[0].getNDOF()
        nDOF_j = self._nodes[1].getNDOF()
        total_dof = nDOF_i + nDOF_j

        # Direction cosine outer product (nD x nD)
        nn = self.__n.outer(self.__n)
        factor = EA / self.__L

        # Initialize full element stiffness matrix
        self._k = Matrix(shape=[total_dof, total_dof])

        # Place blocks: translational DOFs are first nD DOFs of each node's block
        # Node i block: rows/cols [0, nD)
        # Node j block: rows/cols [nDOF_i, nDOF_i + nD)
        for a in range(nD):
            for b in range(nD):
                val = factor * nn[a, b]
                # k_ii block
                self._k[a, b] = self._k[a, b] + val
                # k_ij block
                self._k[a, nDOF_i + b] = self._k[a, nDOF_i + b] - val
                # k_ji block
                self._k[nDOF_i + a, b] = self._k[nDOF_i + a, b] - val
                # k_jj block
                self._k[nDOF_i + a, nDOF_i + b] = self._k[nDOF_i + a, nDOF_i + b] + val

        # Initialize element force vector
        self._f = Vector(shape=total_dof)

        # Build mass matrix if rho > 0
        if self.__rho > 0.0:
            self._m = Matrix(shape=[total_dof, total_dof])
            if self.__cMass:
                # Consistent mass: m = (rho*L/6) * [2I, I; I, 2I] on translational DOFs
                m_factor = self.__rho * self.__L / 6.0
                for a in range(nD):
                    # node i - node i block (2*m_factor on diagonal)
                    self._m[a, a] = 2.0 * m_factor
                    # node i - node j block (1*m_factor on diagonal)
                    self._m[a, nDOF_i + a] = m_factor
                    # node j - node i block (1*m_factor on diagonal)
                    self._m[nDOF_i + a, a] = m_factor
                    # node j - node j block (2*m_factor on diagonal)
                    self._m[nDOF_i + a, nDOF_i + a] = 2.0 * m_factor
            else:
                # Lumped mass: m = rho*L/2 on diagonal translational DOFs
                m_lumped = self.__rho * self.__L / 2.0
                for a in range(nD):
                    self._m[a, a] = m_lumped
                    self._m[nDOF_i + a, nDOF_i + a] = m_lumped


    def _update(self):
        """
        Update element state from current trial displacements.
        Recomputes section strain, internal force vector, and tangent stiffness.
        """
        nD = self._nD
        nDOF_i = self._nodes[0].getNDOF()
        nDOF_j = self._nodes[1].getNDOF()
        total_dof = nDOF_i + nDOF_j

        # Get trial displacements (translational DOFs only)
        u_i = self._nodes[0]._getTrialDisp()
        u_j = self._nodes[1]._getTrialDisp()

        # Compute axial deformation: eps = (u_j - u_i) . n / L
        du = Vector([u_j[k] - u_i[k] for k in range(nD)])
        eps = du.dot(self.__n) / self.__L

        # Update section strain (sets new stress and tangent)
        self._section._setTrialStrain(eps)

        # Recompute element force vector from section stress
        f_axial = self._section.getStress()

        self._f = Vector(shape=total_dof)
        for k in range(nD):
            self._f[k] = -f_axial * self.__n[k]
            self._f[nDOF_i + k] = f_axial * self.__n[k]

        # Rebuild tangent stiffness from current section tangent
        EA = self._section.getTangent()
        nn = self.__n.outer(self.__n)
        factor = EA / self.__L

        self._k = Matrix(shape=[total_dof, total_dof])
        for a in range(nD):
            for b in range(nD):
                val = factor * nn[a, b]
                self._k[a, b] = self._k[a, b] + val
                self._k[a, nDOF_i + b] = self._k[a, nDOF_i + b] - val
                self._k[nDOF_i + a, b] = self._k[nDOF_i + a, b] - val
                self._k[nDOF_i + a, nDOF_i + b] = self._k[nDOF_i + a, nDOF_i + b] + val

        return 0


    def _commit(self):
        self._section.commitState()
        return 0

    def _revert(self):
        self._section.revertToLastCommit()
        return 0

    def __str__(self):
        return "Truss element {}".format(self._ID)

    def __repr__(self):
        return "Truss element {}".format(self._ID)


    # ELEMENT API
    def copy(self):
        return Truss(self._ID, nodes=self._nodes, section=self._section,
                     rho=self.__rho, cMass=self.__cMass)

    def getLength(self):
        return self.__L

    def getDirectionCosine(self):
        return self.__n
