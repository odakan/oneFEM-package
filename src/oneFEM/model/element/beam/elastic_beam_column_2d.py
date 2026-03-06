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
# Date: 05/03/2026
# Version: 0.1
#
# ElasticBeamColumn2d element
#   2D elastic beam-column element with coordinate transformation.
#   Uses Euler-Bernoulli beam theory (cubic Hermite shape functions).
#   Takes E, A, I directly (no Section object — follows OpenSees ElasticBeam2d).
#   Requires a CrdTransf object for local-global mapping.
#
#   Basic stiffness (3x3, rigid-body-removed frame):
#     kb = [[EA/L,    0,      0    ],
#           [0,    4*EI/L, 2*EI/L  ],
#           [0,    2*EI/L, 4*EI/L  ]]
#
#   Global stiffness: K = T^T * kb * T  (via CrdTransf)

from ..main import Element
from ...node.main import Node
from ..kinematics.crdTransf.base import CrdTransf
from ...._systools.data import Vector, Matrix
from ...._systools.math_tools import _is_close
import numpy as np


class ElasticBeamColumn2d(Element):
    """
    2D elastic beam-column element.
    Euler-Bernoulli beam theory with CrdTransf for local-global mapping.
    """

    def __init__(self, ele_id, nodes=None, A=0.0, E=0.0, I=0.0, transf=None,
                 rho=0.0, cMass=False):
        """
        Constructor for ElasticBeamColumn2d.

        :param ele_id: Element ID
        :param nodes: List of 2 Node objects [node_i, node_j]
        :param A: Cross-sectional area
        :param E: Young's modulus
        :param I: Moment of inertia
        :param transf: CrdTransf object (e.g., LinearCrdTransf2d)
        :param rho: Mass per unit length (0 = no mass)
        :param cMass: True = consistent mass, False = lumped mass
        """
        if nodes is None:
            nodes = [Node(), Node()]

        super().__init__(ele_id)

        # Validate nodes
        if len(nodes) != 2:
            raise ValueError("ElasticBeamColumn2d.__init__(): 2 Nodes required for element {}".format(self._ID))
        if nodes[0] is None or nodes[1] is None:
            raise ValueError("ElasticBeamColumn2d.__init__(): Nodes are not defined for element {}".format(self._ID))
        if not isinstance(nodes[0], Node) or not isinstance(nodes[1], Node):
            raise ValueError("ElasticBeamColumn2d.__init__(): Invalid Node type for element {}".format(self._ID))
        if nodes[0].getND() != 2 or nodes[1].getND() != 2:
            raise ValueError("ElasticBeamColumn2d.__init__(): Nodes must be 2D for element {}".format(self._ID))
        if nodes[0].getNDOF() != 3 or nodes[1].getNDOF() != 3:
            raise ValueError("ElasticBeamColumn2d.__init__(): Nodes must have 3 DOFs (ux, uy, theta) for element {}".format(self._ID))

        self._nodes = nodes
        self._nD = 2
        self._nDOF = Vector([3, 3], dtype=int)

        # Validate CrdTransf
        if transf is None:
            raise ValueError("ElasticBeamColumn2d.__init__(): CrdTransf is required for element {}".format(self._ID))
        if not isinstance(transf, CrdTransf):
            raise ValueError("ElasticBeamColumn2d.__init__(): Invalid CrdTransf type for element {}".format(self._ID))

        self.__transf = transf

        # Section properties
        self.__A = A
        self.__E = E
        self.__I = I
        self.__EA = E * A
        self.__EI = E * I

        # Mass properties
        self.__rho = rho
        self.__cMass = cMass

        # Basic stiffness and force (set in _domain)
        self.__kb = None   # 3x3 numpy array
        self.__q = None    # 3-element numpy array (basic forces)

    def _domain(self):
        """
        Initialize element geometry via CrdTransf.
        Compute basic stiffness matrix and build global K.
        """
        # Initialize the coordinate transformation
        self.__transf.initialize(self._nodes[0], self._nodes[1])
        L = self.__transf.getInitialLength()

        # Build basic stiffness matrix (3x3)
        EA = self.__EA
        EI = self.__EI
        self.__kb = np.array([
            [EA/L,    0.0,      0.0     ],
            [0.0,     4.0*EI/L, 2.0*EI/L],
            [0.0,     2.0*EI/L, 4.0*EI/L]
        ])

        # Initial basic forces = zero
        self.__q = np.zeros(3)

        # Transform to global stiffness
        kb_mat = Matrix(init=self.__kb)
        self._k = self.__transf.getInitialGlobalStiffMatrix(kb_mat)

        # Initialize element force vector
        self._f = Vector(shape=6)

        # Build mass matrix if rho > 0
        if self.__rho > 0.0:
            self._m = Matrix(shape=[6, 6])
            if self.__cMass:
                # Consistent mass for Euler-Bernoulli beam (local coords)
                # m_local = (rho*A*L/420) * [standard 6x6 matrix]
                # Then transform to global: M_global = R^T * m_local * R
                # where R is the block rotation matrix
                rAL = self.__rho * L
                m_local = np.zeros((6, 6))

                # Axial terms
                m_local[0, 0] = 140.0
                m_local[0, 3] = 70.0
                m_local[3, 0] = 70.0
                m_local[3, 3] = 140.0

                # Bending terms
                m_local[1, 1] = 156.0
                m_local[1, 2] = 22.0 * L
                m_local[1, 4] = 54.0
                m_local[1, 5] = -13.0 * L
                m_local[2, 1] = 22.0 * L
                m_local[2, 2] = 4.0 * L * L
                m_local[2, 4] = 13.0 * L
                m_local[2, 5] = -3.0 * L * L
                m_local[4, 1] = 54.0
                m_local[4, 2] = 13.0 * L
                m_local[4, 4] = 156.0
                m_local[4, 5] = -22.0 * L
                m_local[5, 1] = -13.0 * L
                m_local[5, 2] = -3.0 * L * L
                m_local[5, 4] = -22.0 * L
                m_local[5, 5] = 4.0 * L * L

                m_local *= (rAL / 420.0)

                # Rotate to global: M_global = R^T * m_local * R
                c = self.__transf.getCosTheta()
                s = self.__transf.getSinTheta()
                R = np.array([
                    [ c, s, 0, 0, 0, 0],
                    [-s, c, 0, 0, 0, 0],
                    [ 0, 0, 1, 0, 0, 0],
                    [ 0, 0, 0, c, s, 0],
                    [ 0, 0, 0,-s, c, 0],
                    [ 0, 0, 0, 0, 0, 1]
                ])
                m_global = R.T @ m_local @ R
                self._m = Matrix(init=m_global)
            else:
                # Lumped mass: rho*L/2 on translational DOFs
                m_lumped = self.__rho * L / 2.0
                self._m[0, 0] = m_lumped
                self._m[1, 1] = m_lumped
                self._m[3, 3] = m_lumped
                self._m[4, 4] = m_lumped

    def _update(self):
        """
        Update element state from current trial displacements.
        Recomputes basic forces and transforms to global K and f.
        """
        # Update transformation (no-op for Linear)
        self.__transf.update()

        # Get basic deformations from CrdTransf
        ub = self.__transf.getBasicTrialDisp()
        ub_np = np.asarray(ub)

        # Compute basic forces: q = kb * ub
        self.__q = self.__kb @ ub_np

        # Transform to global resisting force
        q_vec = Vector(list(self.__q))
        self._f = self.__transf.getGlobalResistingForce(q_vec, None)

        # Transform to global stiffness (kb is constant for elastic)
        kb_mat = Matrix(init=self.__kb)
        self._k = self.__transf.getGlobalStiffMatrix(kb_mat, q_vec)

        return 0

    def _commit(self):
        """Commit element state. No material state to commit for elastic element."""
        return 0

    def _revert(self):
        """Revert element state. No material state to revert for elastic element."""
        return 0

    def __str__(self):
        return "ElasticBeamColumn2d element {}".format(self._ID)

    def __repr__(self):
        return "ElasticBeamColumn2d element {}".format(self._ID)

    # ELEMENT API
    def copy(self):
        return ElasticBeamColumn2d(
            self._ID, nodes=self._nodes,
            A=self.__A, E=self.__E, I=self.__I,
            transf=self.__transf.copy(),
            rho=self.__rho, cMass=self.__cMass
        )

    def getBasicForce(self):
        """Return basic forces [N, Mi, Mj]."""
        if self.__q is not None:
            return Vector(list(self.__q))
        return Vector(shape=3)

    def getLength(self):
        return self.__transf.getInitialLength()
