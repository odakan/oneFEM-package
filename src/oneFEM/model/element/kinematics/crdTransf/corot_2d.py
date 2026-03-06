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
# CorotCrdTransf2d
#   Corotational coordinate transformation for 2D beam-column elements.
#   Captures large deformations by tracking deformed geometry each iteration.
#   OpenSees: CorotCrdTransf2d
#
#   Key idea: local axes rotate with the deformed element chord.
#   The transformation T is recomputed from deformed nodal positions.
#
#   Deformed configuration:
#     Ln = deformed chord length
#     alpha = deformed chord angle (from deformed nodal coordinates)
#     alpha0 = initial chord angle
#
#   Basic deformations (in deformed frame):
#     ub[0] = Ln - L0                  (axial elongation)
#     ub[1] = theta_i - (alpha - alpha0)  (rotation at i relative to chord)
#     ub[2] = theta_j - (alpha - alpha0)  (rotation at j relative to chord)
#
#   Global stiffness:
#     K = Tc^T * kb * Tc + K_sigma
#     where Tc (3x6) uses deformed angle alpha and deformed length Ln
#     and K_sigma is the geometric stiffness:
#       K_sigma = (N/Ln)*outer(rn,rn) - ((M1+M2)/Ln^2)*(outer(rn,sn)+outer(sn,rn))
#     rn = transverse direction in deformed frame
#     sn = axial direction in deformed frame

from .base import CrdTransf
from ....._systools.data import Vector, Matrix
from ....._systools.math_tools import _is_close
import numpy as np


class CorotCrdTransf2d(CrdTransf):
    """
    Corotational coordinate transformation for 2D beam-column elements.
    Full geometric nonlinearity: axes rotate with deformed element.
    """

    formulation = 'corotational'

    def __init__(self):
        super().__init__()
        self.__L0 = 0.0          # initial length
        self.__alpha0 = 0.0      # initial chord angle
        self.__Ln = 0.0          # deformed length
        self.__alpha = 0.0       # deformed chord angle
        self.__cosAlpha = 0.0
        self.__sinAlpha = 0.0
        self.__node_i = None
        self.__node_j = None

    def initialize(self, node_i, node_j):
        """Compute initial geometry."""
        self.__node_i = node_i
        self.__node_j = node_j

        c0 = node_i._coord
        c1 = node_j._coord

        dx = c1[0] - c0[0]
        dy = c1[1] - c0[1]
        L = (dx*dx + dy*dy) ** 0.5

        if _is_close(L, 0.0):
            raise ValueError("CorotCrdTransf2d.initialize(): Element length is zero!")

        self.__L0 = L
        self.__alpha0 = np.arctan2(dy, dx)
        self.__Ln = L
        self.__alpha = self.__alpha0
        self.__cosAlpha = dx / L
        self.__sinAlpha = dy / L

        return 0

    def update(self):
        """
        Recompute deformed geometry from current trial displacements.
        Must be called each Newton iteration (element._update() calls this).
        """
        c0 = self.__node_i._coord
        c1 = self.__node_j._coord

        u_i = self.__node_i._getTrialDisp()
        u_j = self.__node_j._getTrialDisp()

        # Deformed nodal positions
        x_i = c0[0] + u_i[0]
        y_i = c0[1] + u_i[1]
        x_j = c1[0] + u_j[0]
        y_j = c1[1] + u_j[1]

        dx = x_j - x_i
        dy = y_j - y_i
        Ln = (dx*dx + dy*dy) ** 0.5

        if _is_close(Ln, 0.0):
            raise ValueError("CorotCrdTransf2d.update(): Deformed element length is zero!")

        self.__Ln = Ln
        self.__alpha = np.arctan2(dy, dx)
        self.__cosAlpha = dx / Ln
        self.__sinAlpha = dy / Ln

        return 0

    def getInitialLength(self):
        return self.__L0

    def getDeformedLength(self):
        return self.__Ln

    def getBasicTrialDisp(self):
        """
        Basic deformations in the corotated frame.
        ub[0] = Ln - L0          (axial)
        ub[1] = theta_i - dalpha (rotation at i)
        ub[2] = theta_j - dalpha (rotation at j)
        where dalpha = alpha - alpha0 (chord rotation)
        """
        u_i = self.__node_i._getTrialDisp()
        u_j = self.__node_j._getTrialDisp()

        dalpha = self.__alpha - self.__alpha0

        ub = np.zeros(3)
        ub[0] = self.__Ln - self.__L0
        ub[1] = u_i[2] - dalpha
        ub[2] = u_j[2] - dalpha

        return Vector(list(ub))

    def getBasicTrialVel(self):
        """Transform velocities using current deformed T."""
        Tc = self._buildTc()
        v_i = self.__node_i._getTrialVel()
        v_j = self.__node_j._getTrialVel()

        vg = np.array([v_i[0], v_i[1], v_i[2],
                       v_j[0], v_j[1], v_j[2]])

        vb = Tc @ vg
        return Vector(list(vb))

    def getBasicTrialAccel(self):
        """Transform accelerations using current deformed T."""
        Tc = self._buildTc()
        a_i = self.__node_i._getTrialAccel()
        a_j = self.__node_j._getTrialAccel()

        ag = np.array([a_i[0], a_i[1], a_i[2],
                       a_j[0], a_j[1], a_j[2]])

        ab = Tc @ ag
        return Vector(list(ab))

    def _buildTc(self):
        """
        Build the corotational transformation matrix Tc (3x6)
        using deformed geometry (Ln, alpha).
        Same structure as LinearCrdTransf2d but with deformed angle/length.
        """
        c = self.__cosAlpha
        s = self.__sinAlpha
        Ln = self.__Ln
        sl = s / Ln
        cl = c / Ln

        Tc = np.array([
            [-c,  -s,  0.0,  c,   s,  0.0],
            [-sl,  cl, 1.0,  sl, -cl, 0.0],
            [-sl,  cl, 0.0,  sl, -cl, 1.0]
        ])
        return Tc

    def getGlobalResistingForce(self, basic_force, p0):
        """
        Transform basic forces to global using corotational T.
        f = Tc^T * q + p0
        """
        q = np.asarray(basic_force)
        Tc = self._buildTc()
        fg = Tc.T @ q

        if p0 is not None:
            fg = fg + np.asarray(p0)

        return Vector(list(fg))

    def getGlobalStiffMatrix(self, basic_stiff, basic_force):
        """
        Transform basic stiffness to global with geometric stiffness.
        K = Tc^T * kb * Tc + K_sigma

        K_sigma = (N/Ln)*outer(rn,rn)
                + ((M1+M2)/Ln^2)*(outer(rn,sn) + outer(sn,rn))

        rn = transverse direction in deformed frame (perpendicular to chord)
        sn = axial direction in deformed frame (along chord)
        """
        kb = np.asarray(basic_stiff)
        Tc = self._buildTc()
        Kg = Tc.T @ kb @ Tc

        # Geometric stiffness
        q = np.asarray(basic_force)
        N = q[0]      # axial force
        M1 = q[1]     # moment at i
        M2 = q[2]     # moment at j


        c = self.__cosAlpha
        s = self.__sinAlpha
        Ln = self.__Ln

        # rn: transverse to deformed chord [-sin, cos, 0, sin, -cos, 0]
        rn = np.array([-s, c, 0.0, s, -c, 0.0])
        # sn: along deformed chord [-cos, -sin, 0, cos, sin, 0]
        sn = np.array([-c, -s, 0.0, c, s, 0.0])

        # K_sigma from axial force
        if abs(N) > 1e-30:
            Kg += (N / Ln) * np.outer(rn, rn)

        # K_sigma from moments (P-delta second-order terms)
        # Sign: K_sigma = (N/Ln)*rn⊗rn − (M_sum/Ln²)*(rn⊗sn + sn⊗rn)
        # The minus comes from d²(−α)/dug² in the second variation of ub[1], ub[2].
        M_sum = M1 + M2
        if abs(M_sum) > 1e-30:
            Kg -= (M_sum / (Ln * Ln)) * (np.outer(rn, sn) + np.outer(sn, rn))

        return Matrix(init=Kg)

    def getInitialGlobalStiffMatrix(self, basic_stiff):
        """Initial stiffness: no geometric terms (N=0, M=0)."""
        kb = np.asarray(basic_stiff)
        # Use initial geometry
        c = np.cos(self.__alpha0)
        s = np.sin(self.__alpha0)
        L = self.__L0
        sl = s / L
        cl = c / L

        T0 = np.array([
            [-c,  -s,  0.0,  c,   s,  0.0],
            [-sl,  cl, 1.0,  sl, -cl, 0.0],
            [-sl,  cl, 0.0,  sl, -cl, 1.0]
        ])

        Kg = T0.T @ kb @ T0
        return Matrix(init=Kg)

    def getCosTheta(self):
        return self.__cosAlpha

    def getSinTheta(self):
        return self.__sinAlpha

    def copy(self):
        return CorotCrdTransf2d()

    def __str__(self):
        return "CorotCrdTransf2d"

    def __repr__(self):
        return "CorotCrdTransf2d"
