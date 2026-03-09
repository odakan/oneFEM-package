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
# LinearCrdTransf3d
#   Small deformation coordinate transformation for 3D beam-column elements.
#   Local axes computed once at initialize() and remain fixed.
#   OpenSees: LinearCrdTransf3d
#
#   Requires a vecxz vector to define the local coordinate system orientation.
#   vecxz is any vector lying in the local x-z plane (not parallel to element axis).
#
#   Global DOFs per element: [u1,v1,w1,rx1,ry1,rz1, u2,v2,w2,rx2,ry2,rz2] (12 DOFs)
#   Basic DOFs: [axial, rz_i, rz_j, ry_i, ry_j, twist] (6 DOFs)
#
#   Local coordinate system:
#     e_x = element axis (node_i → node_j)
#     e_y = (vecxz × e_x) / |vecxz × e_x|
#     e_z = e_x × e_y
#
#   Transformation: ub = T * ug  where T (6×12) = T_local * R_block
#
#   Basic deformations:
#     ub[0] = u'_j - u'_i                       (axial)
#     ub[1] = θz'_i - (v'_j - v'_i)/L           (bending z at i)
#     ub[2] = θz'_j - (v'_j - v'_i)/L           (bending z at j)
#     ub[3] = θy'_i + (w'_j - w'_i)/L           (bending y at i)
#     ub[4] = θy'_j + (w'_j - w'_i)/L           (bending y at j)
#     ub[5] = θx'_j - θx'_i                     (torsion)

from .base import CrdTransf
from ...._systools.data import Vector, Matrix
from ...._systools.math_tools import _is_close
import numpy as np


class LinearCrdTransf3d(CrdTransf):
    """
    Linear coordinate transformation for 3D beam-column elements.
    Small deformation assumption: local axes are fixed (computed once).
    """

    formulation = 'linear'

    def __init__(self, vecxz):
        """
        :param vecxz: Vector or list defining a vector in the local x-z plane.
                       Used to orient the local y and z axes.
                       Must not be parallel to the element axis.
        """
        super().__init__()
        self.__vecxz = np.asarray(vecxz, dtype=float)
        self.__L = 0.0
        self.__R = None        # 3x3 rotation matrix (local = R * global)
        self.__T = None        # 6x12 transformation matrix (numpy)
        self.__node_i = None
        self.__node_j = None

    def initialize(self, node_i, node_j):
        """
        Compute local axes from nodal coordinates and vecxz.
        """
        self.__node_i = node_i
        self.__node_j = node_j

        c_i = np.array([node_i._coord[k] for k in range(3)])
        c_j = np.array([node_j._coord[k] for k in range(3)])

        dx = c_j - c_i
        L = np.linalg.norm(dx)

        if _is_close(L, 0.0):
            raise ValueError("LinearCrdTransf3d.initialize(): Element length is zero!")

        self.__L = L

        # Local x-axis: along element
        e_x = dx / L

        # Local y-axis: perpendicular to both e_x and vecxz
        # y = (vecxz × e_x) / |vecxz × e_x|
        y_raw = np.cross(self.__vecxz, e_x)
        y_norm = np.linalg.norm(y_raw)
        if y_norm < 1e-10:
            raise ValueError("LinearCrdTransf3d.initialize(): "
                             "vecxz is parallel to element axis!")
        e_y = y_raw / y_norm

        # Local z-axis: completes right-hand system
        e_z = np.cross(e_x, e_y)

        # Rotation matrix: local = R * global
        # Rows are local unit vectors expressed in global coords
        self.__R = np.array([e_x, e_y, e_z])

        # Build 12x12 block rotation matrix
        R = self.__R
        R_block = np.zeros((12, 12))
        for i in range(4):
            R_block[3*i:3*i+3, 3*i:3*i+3] = R

        # Build T_local (6x12): basic = T_local * local_DOFs
        #   local DOFs: [u'i, v'i, w'i, θx'i, θy'i, θz'i,
        #                u'j, v'j, w'j, θx'j, θy'j, θz'j]
        oL = 1.0 / L
        T_local = np.zeros((6, 12))

        # ub[0] = u'_j - u'_i (axial)
        T_local[0, 0] = -1.0
        T_local[0, 6] = 1.0

        # ub[1] = θz'_i + (v'_i - v'_j)/L (bending z at i)
        T_local[1, 1] = oL
        T_local[1, 5] = 1.0
        T_local[1, 7] = -oL

        # ub[2] = θz'_j + (v'_i - v'_j)/L (bending z at j)
        T_local[2, 1] = oL
        T_local[2, 7] = -oL
        T_local[2, 11] = 1.0

        # ub[3] = θy'_i + (w'_j - w'_i)/L (bending y at i)
        T_local[3, 2] = -oL
        T_local[3, 4] = 1.0
        T_local[3, 8] = oL

        # ub[4] = θy'_j + (w'_j - w'_i)/L (bending y at j)
        T_local[4, 2] = -oL
        T_local[4, 8] = oL
        T_local[4, 10] = 1.0

        # ub[5] = θx'_j - θx'_i (torsion)
        T_local[5, 3] = -1.0
        T_local[5, 9] = 1.0

        # Full transformation: T = T_local * R_block
        self.__T = T_local @ R_block

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
        Returns Vector(6): [axial, rz_i, rz_j, ry_i, ry_j, twist]
        """
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
        """Transform global trial velocities to basic velocities."""
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
        """Transform global trial accelerations to basic accelerations."""
        a_i = self.__node_i._getTrialAccel()
        a_j = self.__node_j._getTrialAccel()

        nDOF = self.__node_i.getNDOF()
        ag = np.zeros(12)
        for k in range(nDOF):
            ag[k] = a_i[k]
            ag[6 + k] = a_j[k]

        ab = self.__T @ ag
        return Vector(list(ab))

    def getGlobalResistingForce(self, basic_force, p0):
        """
        Transform basic resisting forces to global frame.
        f_global = T^T * q + p0

        :param basic_force: Vector(6) [N, Mz_i, Mz_j, My_i, My_j, T]
        :param p0: Vector(12) element load in global coords, or None
        :return: Vector(12) global resisting force
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
        For linear transformation, basic_force is unused.

        :param basic_stiff: Matrix(6,6) basic stiffness
        :param basic_force: Vector(6) basic forces (unused for linear)
        :return: Matrix(12,12) global stiffness
        """
        kb = np.asarray(basic_stiff)
        Kg = self.__T.T @ kb @ self.__T
        return Matrix(init=Kg)

    def getInitialGlobalStiffMatrix(self, basic_stiff):
        """Same as getGlobalStiffMatrix for linear (no geometric terms)."""
        kb = np.asarray(basic_stiff)
        Kg = self.__T.T @ kb @ self.__T
        return Matrix(init=Kg)

    def getRotationMatrix(self):
        """Return the 3x3 rotation matrix (local = R * global)."""
        return self.__R.copy()

    def copy(self):
        return LinearCrdTransf3d(self.__vecxz.copy())

    def __str__(self):
        return "LinearCrdTransf3d"

    def __repr__(self):
        return "LinearCrdTransf3d"
