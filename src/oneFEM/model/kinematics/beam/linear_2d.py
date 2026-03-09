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
# LinearCrdTransf2d
#   Small deformation coordinate transformation for 2D beam-column elements.
#   Local axes computed once at initialize() and remain fixed.
#   OpenSees: LinearCrdTransf2d
#
#   Global DOFs per element: [u1, v1, theta1, u2, v2, theta2] (6 DOFs)
#   Basic DOFs: [axial, rot_i, rot_j] (3 DOFs, rigid body modes removed)
#
#   Transformation: ub = T * ug
#   where T (3x6) maps global displacements to basic deformations:
#     q0 = axial elongation  = -cos*u1 - sin*v1 + cos*u2 + sin*v2
#     q1 = rotation at i     = -sin/L*u1 + cos/L*v1 + theta1 + sin/L*u2 - cos/L*v2
#     q2 = rotation at j     = -sin/L*u1 + cos/L*v1 + sin/L*u2 - cos/L*v2 + theta2

from .base import CrdTransf
from ...._systools.data import Vector, Matrix
from ...._systools.math_tools import _is_close
import numpy as np


class LinearCrdTransf2d(CrdTransf):
    """
    Linear coordinate transformation for 2D beam-column elements.
    Small deformation assumption: local axes are fixed (computed once).
    """

    formulation = 'linear'

    def __init__(self):
        super().__init__()
        self.__L = 0.0
        self.__cosTheta = 0.0
        self.__sinTheta = 0.0
        self.__T = None       # 3x6 transformation matrix (numpy)
        self.__node_i = None
        self.__node_j = None

    def initialize(self, node_i, node_j):
        """
        Compute initial local axes from nodal coordinates.
        Must be called once after nodes are set (during element _domain()).
        """
        self.__node_i = node_i
        self.__node_j = node_j

        c0 = node_i._coord
        c1 = node_j._coord

        dx = c1[0] - c0[0]
        dy = c1[1] - c0[1]
        L = (dx*dx + dy*dy) ** 0.5

        if _is_close(L, 0.0):
            raise ValueError("LinearCrdTransf2d.initialize(): Element length is zero!")

        self.__L = L
        self.__cosTheta = dx / L
        self.__sinTheta = dy / L

        # Build the 3x6 transformation matrix T
        c = self.__cosTheta
        s = self.__sinTheta
        sl = s / L
        cl = c / L

        self.__T = np.array([
            [-c,  -s,  0.0,  c,   s,  0.0],
            [-sl,  cl, 1.0,  sl, -cl, 0.0],
            [-sl,  cl, 0.0,  sl, -cl, 1.0]
        ])

        return 0

    def update(self):
        """No-op for linear transformation (axes are fixed)."""
        return 0

    def getInitialLength(self):
        return self.__L

    def getDeformedLength(self):
        return self.__L

    def getBasicTrialDisp(self):
        """
        Transform global trial displacements to basic deformations.
        Returns Vector(3): [axial, rot_i, rot_j]
        """
        u_i = self.__node_i._getTrialDisp()
        u_j = self.__node_j._getTrialDisp()

        # Assemble 6-element global displacement vector
        ug = np.array([u_i[0], u_i[1], u_i[2],
                       u_j[0], u_j[1], u_j[2]])

        ub = self.__T @ ug
        return Vector(list(ub))

    def getBasicTrialVel(self):
        """
        Transform global trial velocities to basic velocities.
        Returns Vector(3).
        """
        v_i = self.__node_i._getTrialVel()
        v_j = self.__node_j._getTrialVel()

        vg = np.array([v_i[0], v_i[1], v_i[2],
                       v_j[0], v_j[1], v_j[2]])

        vb = self.__T @ vg
        return Vector(list(vb))

    def getBasicTrialAccel(self):
        """
        Transform global trial accelerations to basic accelerations.
        Returns Vector(3).
        """
        a_i = self.__node_i._getTrialAccel()
        a_j = self.__node_j._getTrialAccel()

        ag = np.array([a_i[0], a_i[1], a_i[2],
                       a_j[0], a_j[1], a_j[2]])

        ab = self.__T @ ag
        return Vector(list(ab))

    def getGlobalResistingForce(self, basic_force, p0):
        """
        Transform basic resisting forces to global frame.
        f_global = T^T * q + p0

        :param basic_force: Vector(3) of basic forces [N, Mi, Mj]
        :param p0: Vector(6) of element load contributions in global coords, or None
        :return: Vector(6) global resisting force
        """
        q = np.asarray(basic_force)
        fg = self.__T.T @ q

        if p0 is not None:
            fg = fg + np.asarray(p0)

        return Vector(list(fg))

    def getGlobalStiffMatrix(self, basic_stiff, basic_force):
        """
        Transform basic stiffness to global frame.
        K_global = T^T * kb * T
        For linear transformation, basic_force is unused (no geometric stiffness).

        :param basic_stiff: Matrix(3,3) basic stiffness
        :param basic_force: Vector(3) basic forces (unused for linear)
        :return: Matrix(6,6) global stiffness
        """
        kb = np.asarray(basic_stiff)
        Kg = self.__T.T @ kb @ self.__T
        return Matrix(init=Kg)

    def getInitialGlobalStiffMatrix(self, basic_stiff):
        """
        Transform initial basic stiffness to global frame.
        Same as getGlobalStiffMatrix for linear (no geometric terms).
        """
        kb = np.asarray(basic_stiff)
        Kg = self.__T.T @ kb @ self.__T
        return Matrix(init=Kg)

    def getCosTheta(self):
        return self.__cosTheta

    def getSinTheta(self):
        return self.__sinTheta

    def copy(self):
        return LinearCrdTransf2d()

    def __str__(self):
        return "LinearCrdTransf2d"

    def __repr__(self):
        return "LinearCrdTransf2d"
