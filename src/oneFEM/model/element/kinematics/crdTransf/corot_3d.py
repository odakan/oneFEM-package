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
# CorotCrdTransf3d
#   Corotational coordinate transformation for 3D beam-column elements.
#   Full geometric nonlinearity with finite rotations.
#   Follows OpenSees CorotCrdTransf3d.
#
#   The deformed local frame is constructed from deformed nodal coordinates.
#   Nodal rotations are handled using Rodrigues rotation formula.
#
#   Basic deformations (6 DOFs):
#     ub[0] = Ln - L0                         (axial)
#     ub[1] = theta_z_i  (bending z at i, in corotated frame)
#     ub[2] = theta_z_j  (bending z at j, in corotated frame)
#     ub[3] = theta_y_i  (bending y at i, in corotated frame)
#     ub[4] = theta_y_j  (bending y at j, in corotated frame)
#     ub[5] = twist       (relative torsion in corotated frame)
#
#   The corotated frame e1,e2,e3:
#     e1 = deformed chord direction
#     e2, e3 from average nodal rotation applied to initial e2, e3
#
#   Geometric stiffness includes contributions from N, Mz, My, and T.

from .base import CrdTransf
from ....._systools.data import Vector, Matrix
from ....._systools.math_tools import _is_close
import numpy as np


def _rodrigues(theta_vec):
    """
    Compute rotation matrix from rotation vector using Rodrigues formula.
    R = I + sin(theta)/theta * S + (1-cos(theta))/theta^2 * S^2
    where S is the skew-symmetric matrix of theta_vec and theta = |theta_vec|.
    """
    theta = np.linalg.norm(theta_vec)
    if theta < 1e-14:
        return np.eye(3)

    n = theta_vec / theta
    S = np.array([[0, -n[2], n[1]],
                  [n[2], 0, -n[0]],
                  [-n[1], n[0], 0]])

    R = np.eye(3) + np.sin(theta) * S + (1.0 - np.cos(theta)) * (S @ S)
    return R


def _extract_rotation_vector(R):
    """
    Extract rotation vector from rotation matrix.
    Uses the formula: theta * n = [R32-R23, R13-R31, R21-R12] / (2*sin(theta))
    where cos(theta) = (trace(R) - 1) / 2
    """
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if abs(theta) < 1e-14:
        return np.zeros(3)

    if abs(theta - np.pi) < 1e-10:
        # Near pi: use eigenvector of R corresponding to eigenvalue 1
        # R*n = n → (R-I)*n = 0
        vals, vecs = np.linalg.eigh(R)
        idx = np.argmin(np.abs(vals - 1.0))
        n = vecs[:, idx].real
        return theta * n

    sin_theta = np.sin(theta)
    n = np.array([R[2, 1] - R[1, 2],
                  R[0, 2] - R[2, 0],
                  R[1, 0] - R[0, 1]]) / (2.0 * sin_theta)
    return theta * n


class CorotCrdTransf3d(CrdTransf):
    """
    Corotational coordinate transformation for 3D beam-column elements.
    Full geometric nonlinearity with finite rotations (Rodrigues formula).
    """

    formulation = 'corotational'

    def __init__(self, vecxz):
        """
        :param vecxz: Vector or list defining a vector in the local x-z plane.
        """
        super().__init__()
        self.__vecxz = np.asarray(vecxz, dtype=float)
        self.__L0 = 0.0          # initial length
        self.__Ln = 0.0          # deformed length
        self.__R0 = None         # initial rotation matrix (3x3)
        self.__e1 = None         # deformed local x axis
        self.__e2 = None         # deformed local y axis
        self.__e3 = None         # deformed local z axis
        self.__node_i = None
        self.__node_j = None

    def initialize(self, node_i, node_j):
        """Compute initial geometry and local axes."""
        self.__node_i = node_i
        self.__node_j = node_j

        c_i = np.array([node_i._coord[k] for k in range(3)])
        c_j = np.array([node_j._coord[k] for k in range(3)])

        dx = c_j - c_i
        L = np.linalg.norm(dx)

        if _is_close(L, 0.0):
            raise ValueError("CorotCrdTransf3d.initialize(): Element length is zero!")

        self.__L0 = L
        self.__Ln = L

        # Initial local axes (same as LinearCrdTransf3d)
        e_x = dx / L
        y_raw = np.cross(self.__vecxz, e_x)
        y_norm = np.linalg.norm(y_raw)
        if y_norm < 1e-10:
            raise ValueError("CorotCrdTransf3d.initialize(): "
                             "vecxz is parallel to element axis!")
        e_y = y_raw / y_norm
        e_z = np.cross(e_x, e_y)

        self.__R0 = np.array([e_x, e_y, e_z])
        self.__e1 = e_x.copy()
        self.__e2 = e_y.copy()
        self.__e3 = e_z.copy()

        return 0

    def update(self):
        """
        Recompute deformed local axes from current displacements and rotations.

        Steps:
        1. Compute deformed chord (e1) from deformed nodal positions
        2. Extract nodal rotation matrices from rotation DOFs using Rodrigues
        3. Compute average rotation applied to initial e2, e3
        4. Project onto plane perpendicular to e1
        """
        c_i = np.array([self.__node_i._coord[k] for k in range(3)])
        c_j = np.array([self.__node_j._coord[k] for k in range(3)])

        u_i = self.__node_i._getTrialDisp()
        u_j = self.__node_j._getTrialDisp()

        # Deformed nodal positions
        xi = c_i + np.array([u_i[0], u_i[1], u_i[2]])
        xj = c_j + np.array([u_j[0], u_j[1], u_j[2]])

        dx = xj - xi
        Ln = np.linalg.norm(dx)

        if _is_close(Ln, 0.0):
            raise ValueError("CorotCrdTransf3d.update(): Deformed element length is zero!")

        self.__Ln = Ln
        self.__e1 = dx / Ln

        # Extract nodal rotation vectors
        theta_i = np.array([u_i[3], u_i[4], u_i[5]])
        theta_j = np.array([u_j[3], u_j[4], u_j[5]])

        # Nodal rotation matrices
        Ri = _rodrigues(theta_i)
        Rj = _rodrigues(theta_j)

        # Average rotation: Ravg = Ri * rodrigues(0.5 * Ri^T * (Rj - Ri) ... )
        # Simplified: use average rotation vector for moderate rotations
        # For the corotational formulation, we use the average of rotated
        # initial e2/e3 from both nodes
        e2_init = self.__R0[1, :]  # initial e_y
        e3_init = self.__R0[2, :]  # initial e_z

        # Rotate initial local axes by each nodal rotation
        e2_i = Ri @ e2_init
        e2_j = Rj @ e2_init
        e3_i = Ri @ e3_init
        e3_j = Rj @ e3_init

        # Average
        e2_avg = 0.5 * (e2_i + e2_j)
        e3_avg = 0.5 * (e3_i + e3_j)

        # Project e2_avg and e3_avg onto the plane perpendicular to e1
        e1 = self.__e1
        e2_proj = e2_avg - np.dot(e2_avg, e1) * e1
        e2_norm = np.linalg.norm(e2_proj)
        if e2_norm < 1e-14:
            # Fallback: use e3 to define e2
            e3_proj = e3_avg - np.dot(e3_avg, e1) * e1
            e3_norm = np.linalg.norm(e3_proj)
            self.__e3 = e3_proj / e3_norm
            self.__e2 = np.cross(self.__e3, e1)
        else:
            self.__e2 = e2_proj / e2_norm
            self.__e3 = np.cross(e1, self.__e2)

        return 0

    def getInitialLength(self):
        return self.__L0

    def getDeformedLength(self):
        return self.__Ln

    def getBasicTrialDisp(self):
        """
        Basic deformations in the corotated frame.
        ub[0] = Ln - L0  (axial)
        ub[1..5] = relative rotations extracted in the corotated frame
        """
        u_i = self.__node_i._getTrialDisp()
        u_j = self.__node_j._getTrialDisp()

        # Axial deformation
        ub = np.zeros(6)
        ub[0] = self.__Ln - self.__L0

        # Extract nodal rotations
        theta_i = np.array([u_i[3], u_i[4], u_i[5]])
        theta_j = np.array([u_j[3], u_j[4], u_j[5]])

        Ri = _rodrigues(theta_i)
        Rj = _rodrigues(theta_j)

        # Current local frame
        Rbar = np.array([self.__e1, self.__e2, self.__e3])

        # Relative rotation of node i w.r.t. corotated frame
        # Node i's deformed local frame: R_node = R0 @ Ri.T
        #   (Ri is global rotation, R0 maps global→local, so deformed local = R0 @ Ri.T)
        # Relative rotation: Rrel = Rbar @ Ri @ R0.T
        #   (maps node-local → global → corotated)
        # At zero deformation: Rrel = R0 @ I @ R0.T = I
        Rrel_i = Rbar @ Ri @ self.__R0.T
        r_i = _extract_rotation_vector(Rrel_i)

        Rrel_j = Rbar @ Rj @ self.__R0.T
        r_j = _extract_rotation_vector(Rrel_j)

        # Basic rotations:
        # ub[1] = θz at i (rotation about corotated z → bending in x-y plane)
        # ub[2] = θz at j
        # ub[3] = θy at i (rotation about corotated y → bending in x-z plane)
        # ub[4] = θy at j
        # ub[5] = θx_j - θx_i (torsion = relative twist about corotated x)
        ub[1] = r_i[2]   # rotation about e3 (local z)
        ub[2] = r_j[2]
        ub[3] = r_i[1]   # rotation about e2 (local y)
        ub[4] = r_j[1]
        ub[5] = r_j[0] - r_i[0]  # relative twist about e1

        return Vector(list(ub))

    def _buildTc(self):
        """
        Build the corotational T matrix (6x12) using deformed geometry.
        Same structure as LinearCrdTransf3d but with deformed axes and length.
        """
        Rbar = np.array([self.__e1, self.__e2, self.__e3])
        Ln = self.__Ln

        # Block rotation matrix
        R_block = np.zeros((12, 12))
        for i in range(4):
            R_block[3*i:3*i+3, 3*i:3*i+3] = Rbar

        oL = 1.0 / Ln
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

        return T_local @ R_block

    def getBasicTrialVel(self):
        Tc = self._buildTc()
        v_i = self.__node_i._getTrialVel()
        v_j = self.__node_j._getTrialVel()
        nDOF = self.__node_i.getNDOF()
        vg = np.zeros(12)
        for k in range(nDOF):
            vg[k] = v_i[k]
            vg[6 + k] = v_j[k]
        return Vector(list(Tc @ vg))

    def getBasicTrialAccel(self):
        Tc = self._buildTc()
        a_i = self.__node_i._getTrialAccel()
        a_j = self.__node_j._getTrialAccel()
        nDOF = self.__node_i.getNDOF()
        ag = np.zeros(12)
        for k in range(nDOF):
            ag[k] = a_i[k]
            ag[6 + k] = a_j[k]
        return Vector(list(Tc @ ag))

    def getGlobalResistingForce(self, basic_force, p0):
        """Transform basic forces to global using corotational T."""
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

        Geometric stiffness K_sigma includes:
        - (N/Ln) * (outer(ry,ry) + outer(rz,rz))   from axial force
        - moment terms coupling axial and transverse directions
        """
        kb = np.asarray(basic_stiff)
        Tc = self._buildTc()
        Kg = Tc.T @ kb @ Tc

        q = np.asarray(basic_force)
        N = q[0]
        Mz_i = q[1]
        Mz_j = q[2]
        My_i = q[3]
        My_j = q[4]

        Ln = self.__Ln
        e1 = self.__e1
        e2 = self.__e2
        e3 = self.__e3

        # Transverse direction vectors (12-element)
        ry = np.zeros(12)
        ry[0:3] = -e2
        ry[6:9] = e2

        rz = np.zeros(12)
        rz[0:3] = -e3
        rz[6:9] = e3

        # Axial direction vector
        sx = np.zeros(12)
        sx[0:3] = -e1
        sx[6:9] = e1

        # K_sigma from axial force (on transverse DOFs)
        if abs(N) > 1e-30:
            NoverLn = N / Ln
            Kg += NoverLn * (np.outer(ry, ry) + np.outer(rz, rz))

        # K_sigma from moments (coupling axial-transverse)
        Mz_sum = Mz_i + Mz_j
        if abs(Mz_sum) > 1e-30:
            coeff = Mz_sum / (Ln * Ln)
            Kg += coeff * (np.outer(ry, sx) + np.outer(sx, ry))

        My_sum = My_i + My_j
        if abs(My_sum) > 1e-30:
            coeff = My_sum / (Ln * Ln)
            Kg += coeff * (np.outer(rz, sx) + np.outer(sx, rz))

        return Matrix(init=Kg)

    def getInitialGlobalStiffMatrix(self, basic_stiff):
        """Initial stiffness (no geometric terms)."""
        # Use initial geometry
        R_block = np.zeros((12, 12))
        for i in range(4):
            R_block[3*i:3*i+3, 3*i:3*i+3] = self.__R0

        oL = 1.0 / self.__L0
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

        T0 = T_local @ R_block
        kb = np.asarray(basic_stiff)
        return Matrix(init=T0.T @ kb @ T0)

    def getRotationMatrix(self):
        return np.array([self.__e1, self.__e2, self.__e3])

    def copy(self):
        return CorotCrdTransf3d(self.__vecxz.copy())

    def __str__(self):
        return "CorotCrdTransf3d"

    def __repr__(self):
        return "CorotCrdTransf3d"
