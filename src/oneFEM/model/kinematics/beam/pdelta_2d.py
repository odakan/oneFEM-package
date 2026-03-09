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
# PDeltaCrdTransf2d
#   P-Delta coordinate transformation for 2D beam-column elements.
#   Adds linearized geometric stiffness (P-delta effect) to the linear
#   transformation. Local axes remain fixed (small deformation assumption).
#
#   Same basic deformations as LinearCrdTransf2d:
#     ub = T * ug  (T is 3x6, computed once)
#
#   Geometric stiffness:
#     K_g = (N/L) * outer(r, r)
#     where r = [-sinTheta, cosTheta, 0, sinTheta, -cosTheta, 0]
#     and N = basic_force[0] (axial force, positive = tension)
#
#   Global stiffness: K = T^T * kb * T + K_g
#   Global resisting force: f = T^T * q + f_pdelta
#     where f_pdelta = (N/L) * (r . ug) * r

from .base import CrdTransf
from ...._systools.data import Vector, Matrix
from ...._systools.math_tools import _is_close
import numpy as np


class PDeltaCrdTransf2d(CrdTransf):
    """
    P-Delta coordinate transformation for 2D beam-column elements.
    Linearized geometric stiffness: captures P-delta effects but not
    large deformations.
    """

    formulation = 'pdelta'

    def __init__(self):
        super().__init__()
        self.__L = 0.0
        self.__cosTheta = 0.0
        self.__sinTheta = 0.0
        self.__T = None       # 3x6 transformation matrix (numpy)
        self.__r = None       # 6-element transverse direction vector
        self.__node_i = None
        self.__node_j = None

    def initialize(self, node_i, node_j):
        """
        Compute initial local axes from nodal coordinates.
        Same as LinearCrdTransf2d.
        """
        self.__node_i = node_i
        self.__node_j = node_j

        c0 = node_i._coord
        c1 = node_j._coord

        dx = c1[0] - c0[0]
        dy = c1[1] - c0[1]
        L = (dx*dx + dy*dy) ** 0.5

        if _is_close(L, 0.0):
            raise ValueError("PDeltaCrdTransf2d.initialize(): Element length is zero!")

        self.__L = L
        self.__cosTheta = dx / L
        self.__sinTheta = dy / L

        c = self.__cosTheta
        s = self.__sinTheta
        sl = s / L
        cl = c / L

        self.__T = np.array([
            [-c,  -s,  0.0,  c,   s,  0.0],
            [-sl,  cl, 1.0,  sl, -cl, 0.0],
            [-sl,  cl, 0.0,  sl, -cl, 1.0]
        ])

        # Transverse direction vector for geometric stiffness
        self.__r = np.array([-s, c, 0.0, s, -c, 0.0])

        return 0

    def update(self):
        """No-op for P-Delta (axes are fixed, same as Linear)."""
        return 0

    def getInitialLength(self):
        return self.__L

    def getDeformedLength(self):
        return self.__L

    def getBasicTrialDisp(self):
        """Same as LinearCrdTransf2d."""
        u_i = self.__node_i._getTrialDisp()
        u_j = self.__node_j._getTrialDisp()

        ug = np.array([u_i[0], u_i[1], u_i[2],
                       u_j[0], u_j[1], u_j[2]])

        ub = self.__T @ ug
        return Vector(list(ub))

    def getBasicTrialVel(self):
        v_i = self.__node_i._getTrialVel()
        v_j = self.__node_j._getTrialVel()

        vg = np.array([v_i[0], v_i[1], v_i[2],
                       v_j[0], v_j[1], v_j[2]])

        vb = self.__T @ vg
        return Vector(list(vb))

    def getBasicTrialAccel(self):
        a_i = self.__node_i._getTrialAccel()
        a_j = self.__node_j._getTrialAccel()

        ag = np.array([a_i[0], a_i[1], a_i[2],
                       a_j[0], a_j[1], a_j[2]])

        ab = self.__T @ ag
        return Vector(list(ab))

    def getGlobalResistingForce(self, basic_force, p0):
        """
        Transform basic forces to global frame with P-delta contribution.
        f = T^T * q + f_pdelta + p0
        f_pdelta = (N/L) * (r . ug) * r
        """
        q = np.asarray(basic_force)
        fg = self.__T.T @ q

        # P-delta force contribution
        N = q[0]  # axial force (positive = tension)
        if abs(N) > 1e-30:
            u_i = self.__node_i._getTrialDisp()
            u_j = self.__node_j._getTrialDisp()
            ug = np.array([u_i[0], u_i[1], u_i[2],
                           u_j[0], u_j[1], u_j[2]])
            delta_perp = self.__r @ ug
            fg += (N / self.__L) * delta_perp * self.__r

        if p0 is not None:
            fg = fg + np.asarray(p0)

        return Vector(list(fg))

    def getGlobalStiffMatrix(self, basic_stiff, basic_force):
        """
        Transform basic stiffness to global with geometric stiffness.
        K = T^T * kb * T + (N/L) * outer(r, r)
        """
        kb = np.asarray(basic_stiff)
        Kg = self.__T.T @ kb @ self.__T

        # Add geometric stiffness
        q = np.asarray(basic_force)
        N = q[0]
        if abs(N) > 1e-30:
            Kg += (N / self.__L) * np.outer(self.__r, self.__r)

        return Matrix(init=Kg)

    def getInitialGlobalStiffMatrix(self, basic_stiff):
        """Initial stiffness has no geometric contribution (N=0)."""
        kb = np.asarray(basic_stiff)
        Kg = self.__T.T @ kb @ self.__T
        return Matrix(init=Kg)

    def getCosTheta(self):
        return self.__cosTheta

    def getSinTheta(self):
        return self.__sinTheta

    def copy(self):
        return PDeltaCrdTransf2d()

    def __str__(self):
        return "PDeltaCrdTransf2d"

    def __repr__(self):
        return "PDeltaCrdTransf2d"
