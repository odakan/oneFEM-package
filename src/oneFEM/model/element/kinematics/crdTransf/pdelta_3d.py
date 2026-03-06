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
# PDeltaCrdTransf3d
#   P-Delta coordinate transformation for 3D beam-column elements.
#   Adds linearized geometric stiffness to LinearCrdTransf3d.
#
#   Same basic deformations as LinearCrdTransf3d (6 DOFs).
#   Same T matrix (6x12), computed once at initialize().
#
#   Geometric stiffness:
#     K_g = (N/L) * (outer(ry,ry) + outer(rz,rz))
#   where:
#     ry = [-e_y, 0, 0, 0, e_y, 0, 0, 0] (12-element, transverse y)
#     rz = [-e_z, 0, 0, 0, e_z, 0, 0, 0] (12-element, transverse z)
#     e_y, e_z are local axis unit vectors (rows of rotation matrix R)
#     N = basic_force[0] (axial force)

from .base import CrdTransf
from ....._systools.data import Vector, Matrix
from ....._systools.math_tools import _is_close
import numpy as np


class PDeltaCrdTransf3d(CrdTransf):
    """
    P-Delta coordinate transformation for 3D beam-column elements.
    Linearized geometric stiffness on top of small-deformation kinematics.
    """

    formulation = 'pdelta'

    def __init__(self, vecxz):
        """
        :param vecxz: Vector or list defining a vector in the local x-z plane.
        """
        super().__init__()
        self.__vecxz = np.asarray(vecxz, dtype=float)
        self.__L = 0.0
        self.__R = None        # 3x3 rotation matrix
        self.__T = None        # 6x12 transformation matrix
        self.__ry = None       # 12-element transverse y vector
        self.__rz = None       # 12-element transverse z vector
        self.__node_i = None
        self.__node_j = None

    def initialize(self, node_i, node_j):
        """Compute local axes and T matrix (same as LinearCrdTransf3d)."""
        self.__node_i = node_i
        self.__node_j = node_j

        c_i = np.array([node_i._coord[k] for k in range(3)])
        c_j = np.array([node_j._coord[k] for k in range(3)])

        dx = c_j - c_i
        L = np.linalg.norm(dx)

        if _is_close(L, 0.0):
            raise ValueError("PDeltaCrdTransf3d.initialize(): Element length is zero!")

        self.__L = L
        e_x = dx / L

        y_raw = np.cross(self.__vecxz, e_x)
        y_norm = np.linalg.norm(y_raw)
        if y_norm < 1e-10:
            raise ValueError("PDeltaCrdTransf3d.initialize(): "
                             "vecxz is parallel to element axis!")
        e_y = y_raw / y_norm
        e_z = np.cross(e_x, e_y)

        self.__R = np.array([e_x, e_y, e_z])

        # Build T matrix (same as LinearCrdTransf3d)
        R = self.__R
        R_block = np.zeros((12, 12))
        for i in range(4):
            R_block[3*i:3*i+3, 3*i:3*i+3] = R

        oL = 1.0 / L
        T_local = np.zeros((6, 12))
        T_local[0, 0] = -1.0
        T_local[0, 6] = 1.0
        T_local[1, 1] = oL
        T_local[1, 5] = 1.0
        T_local[1, 7] = -oL
        T_local[2, 1] = oL
        T_local[2, 7] = -oL
        T_local[2, 11] = 1.0
        T_local[3, 2] = -oL
        T_local[3, 4] = 1.0
        T_local[3, 8] = oL
        T_local[4, 2] = -oL
        T_local[4, 8] = oL
        T_local[4, 10] = 1.0
        T_local[5, 3] = -1.0
        T_local[5, 9] = 1.0

        self.__T = T_local @ R_block

        # Transverse direction vectors for geometric stiffness
        # ry: transverse displacement in local y direction
        # node_i DOFs [0:3] = translations, [3:6] = rotations
        # node_j DOFs [6:9] = translations, [9:12] = rotations
        self.__ry = np.zeros(12)
        self.__ry[0:3] = -e_y
        self.__ry[6:9] = e_y

        self.__rz = np.zeros(12)
        self.__rz[0:3] = -e_z
        self.__rz[6:9] = e_z

        return 0

    def update(self):
        """No-op for P-Delta (axes are fixed)."""
        return 0

    def getInitialLength(self):
        return self.__L

    def getDeformedLength(self):
        return self.__L

    def getBasicTrialDisp(self):
        """Same as LinearCrdTransf3d."""
        u_i = self.__node_i._getTrialDisp()
        u_j = self.__node_j._getTrialDisp()

        nDOF = self.__node_i.getNDOF()
        ug = np.zeros(12)
        for k in range(nDOF):
            ug[k] = u_i[k]
            ug[6 + k] = u_j[k]

        ub = self.__T @ ug
        return Vector(list(ub))

    def getBasicTrialVel(self):
        v_i = self.__node_i._getTrialVel()
        v_j = self.__node_j._getTrialVel()

        nDOF = self.__node_i.getNDOF()
        vg = np.zeros(12)
        for k in range(nDOF):
            vg[k] = v_i[k]
            vg[6 + k] = v_j[k]

        vb = self.__T @ vg
        return Vector(list(vb))

    def getBasicTrialAccel(self):
        a_i = self.__node_i._getTrialAccel()
        a_j = self.__node_j._getTrialAccel()

        nDOF = self.__node_i.getNDOF()
        ag = np.zeros(12)
        for k in range(nDOF):
            ag[k] = a_i[k]
            ag[6 + k] = a_j[k]

        ab = self.__T @ ag
        return Vector(list(ab))

    def _getGlobalDisp(self):
        """Assemble the 12-element global displacement vector."""
        u_i = self.__node_i._getTrialDisp()
        u_j = self.__node_j._getTrialDisp()
        nDOF = self.__node_i.getNDOF()
        ug = np.zeros(12)
        for k in range(nDOF):
            ug[k] = u_i[k]
            ug[6 + k] = u_j[k]
        return ug

    def getGlobalResistingForce(self, basic_force, p0):
        """
        Transform basic forces to global with P-delta contribution.
        f = T^T * q + f_pdelta + p0
        """
        q = np.asarray(basic_force)
        fg = self.__T.T @ q

        # P-delta force contribution
        N = q[0]  # axial force
        if abs(N) > 1e-30:
            ug = self._getGlobalDisp()
            NoverL = N / self.__L
            delta_y = self.__ry @ ug
            delta_z = self.__rz @ ug
            fg += NoverL * (delta_y * self.__ry + delta_z * self.__rz)

        if p0 is not None:
            fg = fg + np.asarray(p0)

        return Vector(list(fg))

    def getGlobalStiffMatrix(self, basic_stiff, basic_force):
        """
        Transform basic stiffness to global with geometric stiffness.
        K = T^T * kb * T + (N/L) * (outer(ry,ry) + outer(rz,rz))
        """
        kb = np.asarray(basic_stiff)
        Kg = self.__T.T @ kb @ self.__T

        q = np.asarray(basic_force)
        N = q[0]
        if abs(N) > 1e-30:
            NoverL = N / self.__L
            Kg += NoverL * (np.outer(self.__ry, self.__ry) +
                            np.outer(self.__rz, self.__rz))

        return Matrix(init=Kg)

    def getInitialGlobalStiffMatrix(self, basic_stiff):
        """Initial stiffness (no geometric terms)."""
        kb = np.asarray(basic_stiff)
        Kg = self.__T.T @ kb @ self.__T
        return Matrix(init=Kg)

    def getRotationMatrix(self):
        return self.__R.copy()

    def copy(self):
        return PDeltaCrdTransf3d(self.__vecxz.copy())

    def __str__(self):
        return "PDeltaCrdTransf3d"

    def __repr__(self):
        return "PDeltaCrdTransf3d"
