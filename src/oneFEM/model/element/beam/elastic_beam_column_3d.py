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
# ElasticBeamColumn3d element
#   3D elastic beam-column element with coordinate transformation.
#   Uses Euler-Bernoulli beam theory.
#   Takes E, A, Iz, Iy, G, J directly (follows OpenSees ElasticBeam3d).
#   Requires a CrdTransf object for local-global mapping.
#
#   Basic stiffness (6x6, rigid-body-removed frame):
#     kb = diag-block with:
#       [0,0] = EA/L          (axial)
#       [1:3,1:3] = EIz block (bending about z, x-y plane)
#       [3:5,3:5] = EIy block (bending about y, x-z plane)
#       [5,5] = GJ/L          (torsion)
#
#   Global stiffness: K = T^T * kb * T  (via CrdTransf)

from ..main import Element
from ...node.main import Node
from ...kinematics.beam.base import CrdTransf
from ...._systools.data import Vector, Matrix
from ...._systools.math_tools import _is_close
import numpy as np


class ElasticBeamColumn3d(Element):
    """
    3D elastic beam-column element.
    Euler-Bernoulli beam theory with CrdTransf for local-global mapping.
    """

    def __init__(self, ele_id, nodes=None, A=0.0, E=0.0, Iz=0.0, Iy=0.0,
                 G=0.0, J=0.0, transf=None, rho=0.0, cMass=False):
        """
        Constructor for ElasticBeamColumn3d.

        :param ele_id: Element ID
        :param nodes: List of 2 Node objects [node_i, node_j]
        :param A: Cross-sectional area
        :param E: Young's modulus
        :param Iz: Moment of inertia about local z-axis (bending in x-y plane)
        :param Iy: Moment of inertia about local y-axis (bending in x-z plane)
        :param G: Shear modulus
        :param J: Torsional constant
        :param transf: CrdTransf object (e.g., LinearCrdTransf3d)
        :param rho: Mass per unit length (0 = no mass)
        :param cMass: True = consistent mass, False = lumped mass
        """
        if nodes is None:
            nodes = [Node(), Node()]

        super().__init__(ele_id)

        # Validate nodes
        if len(nodes) != 2:
            raise ValueError("ElasticBeamColumn3d.__init__(): 2 Nodes required for element {}".format(self._ID))
        if nodes[0] is None or nodes[1] is None:
            raise ValueError("ElasticBeamColumn3d.__init__(): Nodes are not defined for element {}".format(self._ID))
        if not isinstance(nodes[0], Node) or not isinstance(nodes[1], Node):
            raise ValueError("ElasticBeamColumn3d.__init__(): Invalid Node type for element {}".format(self._ID))
        if nodes[0].getND() != 3 or nodes[1].getND() != 3:
            raise ValueError("ElasticBeamColumn3d.__init__(): Nodes must be 3D for element {}".format(self._ID))
        if nodes[0].getNDOF() < 6 or nodes[1].getNDOF() < 6:
            raise ValueError("ElasticBeamColumn3d.__init__(): Nodes must have >= 6 DOFs for element {}".format(self._ID))

        self._nodes = nodes
        self._nD = 3
        self._nDOF = Vector([nodes[0].getNDOF(), nodes[1].getNDOF()], dtype=int)

        # Validate CrdTransf
        if transf is None:
            raise ValueError("ElasticBeamColumn3d.__init__(): CrdTransf is required for element {}".format(self._ID))
        if not isinstance(transf, CrdTransf):
            raise ValueError("ElasticBeamColumn3d.__init__(): Invalid CrdTransf type for element {}".format(self._ID))

        self.__transf = transf

        # Section properties
        self.__A = A
        self.__E = E
        self.__Iz = Iz
        self.__Iy = Iy
        self.__G = G
        self.__J = J

        # Mass properties
        self.__rho = rho
        self.__cMass = cMass

        # Basic stiffness and force (set in _domain)
        self.__kb = None   # 6x6 numpy array
        self.__q = None    # 6-element numpy array (basic forces)

        # Total DOF count
        self.__nDOF_total = nodes[0].getNDOF() + nodes[1].getNDOF()

    def _domain(self):
        """
        Initialize element geometry via CrdTransf.
        Compute basic stiffness matrix and build global K.
        """
        # Initialize the coordinate transformation
        self.__transf.initialize(self._nodes[0], self._nodes[1])
        L = self.__transf.getInitialLength()

        # Build basic stiffness matrix (6x6)
        E = self.__E
        A = self.__A
        Iz = self.__Iz
        Iy = self.__Iy
        G = self.__G
        J = self.__J

        EA = E * A
        EIz = E * Iz
        EIy = E * Iy
        GJ = G * J

        self.__kb = np.zeros((6, 6))

        # Axial
        self.__kb[0, 0] = EA / L

        # Bending about z (x-y plane): [4EIz/L, 2EIz/L; 2EIz/L, 4EIz/L]
        self.__kb[1, 1] = 4.0 * EIz / L
        self.__kb[1, 2] = 2.0 * EIz / L
        self.__kb[2, 1] = 2.0 * EIz / L
        self.__kb[2, 2] = 4.0 * EIz / L

        # Bending about y (x-z plane): [4EIy/L, 2EIy/L; 2EIy/L, 4EIy/L]
        self.__kb[3, 3] = 4.0 * EIy / L
        self.__kb[3, 4] = 2.0 * EIy / L
        self.__kb[4, 3] = 2.0 * EIy / L
        self.__kb[4, 4] = 4.0 * EIy / L

        # Torsion
        self.__kb[5, 5] = GJ / L

        # Initial basic forces = zero
        self.__q = np.zeros(6)

        # Transform to global stiffness
        kb_mat = Matrix(init=self.__kb)
        self._k = self.__transf.getInitialGlobalStiffMatrix(kb_mat)

        # Initialize element force vector
        nDOF_total = self.__nDOF_total
        self._f = Vector(shape=nDOF_total)

        # Build mass matrix if rho > 0
        if self.__rho > 0.0:
            nDOF_i = self._nodes[0].getNDOF()
            self._m = Matrix(shape=[nDOF_total, nDOF_total])
            if not self.__cMass:
                # Lumped mass: rho*L/2 on translational DOFs (first 3 of each node)
                m_lumped = self.__rho * L / 2.0
                for a in range(3):
                    self._m[a, a] = m_lumped
                    self._m[nDOF_i + a, nDOF_i + a] = m_lumped
            else:
                # Consistent mass: build in local, rotate to global
                rAL = self.__rho * L
                m_local = np.zeros((12, 12))

                # Axial (u' DOFs: 0, 6)
                m_local[0, 0] = 140.0
                m_local[0, 6] = 70.0
                m_local[6, 0] = 70.0
                m_local[6, 6] = 140.0

                # Bending x-y plane (v' DOFs: 1, 7; θz' DOFs: 5, 11)
                m_local[1, 1] = 156.0
                m_local[1, 5] = 22.0 * L
                m_local[1, 7] = 54.0
                m_local[1, 11] = -13.0 * L
                m_local[5, 1] = 22.0 * L
                m_local[5, 5] = 4.0 * L * L
                m_local[5, 7] = 13.0 * L
                m_local[5, 11] = -3.0 * L * L
                m_local[7, 1] = 54.0
                m_local[7, 5] = 13.0 * L
                m_local[7, 7] = 156.0
                m_local[7, 11] = -22.0 * L
                m_local[11, 1] = -13.0 * L
                m_local[11, 5] = -3.0 * L * L
                m_local[11, 7] = -22.0 * L
                m_local[11, 11] = 4.0 * L * L

                # Bending x-z plane (w' DOFs: 2, 8; θy' DOFs: 4, 10)
                m_local[2, 2] = 156.0
                m_local[2, 4] = -22.0 * L
                m_local[2, 8] = 54.0
                m_local[2, 10] = 13.0 * L
                m_local[4, 2] = -22.0 * L
                m_local[4, 4] = 4.0 * L * L
                m_local[4, 8] = -13.0 * L
                m_local[4, 10] = -3.0 * L * L
                m_local[8, 2] = 54.0
                m_local[8, 4] = -13.0 * L
                m_local[8, 8] = 156.0
                m_local[8, 10] = 22.0 * L
                m_local[10, 2] = 13.0 * L
                m_local[10, 4] = -3.0 * L * L
                m_local[10, 8] = 22.0 * L
                m_local[10, 10] = 4.0 * L * L

                # Torsion (θx' DOFs: 3, 9)
                m_local[3, 3] = 140.0
                m_local[3, 9] = 70.0
                m_local[9, 3] = 70.0
                m_local[9, 9] = 140.0

                m_local *= (rAL / 420.0)

                # Rotate to global using block rotation
                R = self.__transf.getRotationMatrix()
                R_block = np.zeros((12, 12))
                for i in range(4):
                    R_block[3*i:3*i+3, 3*i:3*i+3] = R
                m_global = R_block.T @ m_local @ R_block

                # Scatter into element mass matrix (handles nDOF > 6 per node)
                for i in range(12):
                    for j in range(12):
                        if abs(m_global[i, j]) > 1e-30:
                            # Map local 12-DOF index to element DOF index
                            if i < 6:
                                ei = i
                            else:
                                ei = nDOF_i + (i - 6)
                            if j < 6:
                                ej = j
                            else:
                                ej = nDOF_i + (j - 6)
                            self._m[ei, ej] = m_global[i, j]

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
        f_global = self.__transf.getGlobalResistingForce(q_vec, None)

        # Scatter into element force vector (handles nDOF > 6)
        nDOF_i = self._nodes[0].getNDOF()
        nDOF_total = self.__nDOF_total
        self._f = Vector(shape=nDOF_total)
        f_np = np.asarray(f_global)
        for k in range(6):
            self._f[k] = f_np[k]
            self._f[nDOF_i + k] = f_np[6 + k]

        # Transform to global stiffness
        kb_mat = Matrix(init=self.__kb)
        K_global = self.__transf.getGlobalStiffMatrix(kb_mat, q_vec)
        K_np = np.asarray(K_global)

        # Scatter into element stiffness matrix (handles nDOF > 6)
        self._k = Matrix(shape=[nDOF_total, nDOF_total])
        for i in range(12):
            for j in range(12):
                if abs(K_np[i, j]) > 1e-30:
                    if i < 6:
                        ei = i
                    else:
                        ei = nDOF_i + (i - 6)
                    if j < 6:
                        ej = j
                    else:
                        ej = nDOF_i + (j - 6)
                    self._k[ei, ej] = K_np[i, j]

        return 0

    def _commit(self):
        """Commit element state. No material state to commit for elastic element."""
        return 0

    def _revert(self):
        """Revert element state. No material state to revert for elastic element."""
        return 0

    def __str__(self):
        return "ElasticBeamColumn3d element {}".format(self._ID)

    def __repr__(self):
        return "ElasticBeamColumn3d element {}".format(self._ID)

    # ELEMENT API
    def copy(self):
        return ElasticBeamColumn3d(
            self._ID, nodes=self._nodes,
            A=self.__A, E=self.__E, Iz=self.__Iz, Iy=self.__Iy,
            G=self.__G, J=self.__J,
            transf=self.__transf.copy(),
            rho=self.__rho, cMass=self.__cMass
        )

    def getBasicForce(self):
        """Return basic forces [N, Mz_i, Mz_j, My_i, My_j, T]."""
        if self.__q is not None:
            return Vector(list(self.__q))
        return Vector(shape=6)

    def getLength(self):
        return self.__transf.getInitialLength()
