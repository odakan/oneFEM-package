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
# _NonlinearContinuumBase
#   Shared math for TotalLagrangian and UpdatedLagrangian kinematics.
#   Computes F, Green-Lagrange E, nonlinear B_NL, geometric stiffness K_sigma.
#
#   Private helpers use .data for numpy indexing per Design Rule 10 amendment.

from .base import ContinuumKinematics
from ....._systools.data import Vector, Matrix
from ....._systools.data.ctensor import CTensor
from ....._systools.backend import np


class _NonlinearContinuumBase(ContinuumKinematics):
    """Shared nonlinear math for TL and UL formulations.

    Provides per-GP arrays for F, B_NL, strain, and methods to compute:
    - Displacement gradient H
    - Deformation gradient F = I + H
    - Green-Lagrange strain E = 0.5*(H + H^T + H^T @ H)
    - Nonlinear B matrix (B_NL) consistent with E
    - Geometric stiffness K_sigma from 2PK stress
    """

    def initialize(self, nGP, nDim, nNodes, dN_dX_list, **kwargs):
        super().initialize(nGP, nDim, nNodes, dN_dX_list, **kwargs)
        self._F = [None] * nGP
        self._B_NL = [None] * nGP
        self._strain = [None] * nGP

    def _computeH(self, gp, u_e):
        """Displacement gradient H = u_mat^T @ dN_dX. Returns Matrix (nDim x nDim).
        Uses .data for reshape; accepts/returns native objects."""
        u_data = u_e.data.reshape(self._nNodes, self._nDim)
        u_mat = Matrix(init=u_data)
        dN = self._dN_dX[gp]
        return u_mat.T @ dN

    def _computeGreenLagrange(self, H):
        """E = 0.5*(H + H^T + H^T @ H). Returns Matrix (nDim x nDim)."""
        return (H + H.T + H.T @ H) * 0.5

    def _EtoVoigtCOV(self, E_mat):
        """Convert Green-Lagrange Matrix -> CTensor (2nd order, COV).
        COV input expects engineering shear (2*E_ij for off-diag)."""
        E = E_mat.data
        if self._nDim == 2:
            return CTensor([E[0, 0], E[1, 1], 2.0 * E[0, 1]],
                           self._nVoigt, CTensor.COV)
        else:
            return CTensor([E[0, 0], E[1, 1], E[2, 2],
                            2.0 * E[0, 1], 2.0 * E[1, 2], 2.0 * E[0, 2]],
                           self._nVoigt, CTensor.COV)

    def _computeF(self, H):
        """Deformation gradient F = I + H. Returns Matrix (nDim x nDim)."""
        return Matrix(init=np.eye(self._nDim) + H.data)

    def _buildBNL(self, gp, H, dN_dX_gp=None):
        """Nonlinear B matrix from displacement gradient H. Returns Matrix (nVoigt x nDOF).

        F = I + H. Row convention:
        2D: [E11, E22, 2*E12] (engineering shear, matches COV input)
        B_NL[r, a*nDim+i] = F[i,j_r] * dN_a/dX[k_r] per row r's index map.

        :param dN_dX_gp: optional Matrix override for shape function derivatives.
            If None, uses self._dN_dX[gp]. UL passes dN_dX_original for
            consistency with total-reference strain/stress.

        Uses .data for numpy indexing per Design Rule 10 amendment.
        """
        nDim = self._nDim
        F = np.eye(nDim) + H.data
        dN = (dN_dX_gp if dN_dX_gp is not None else self._dN_dX[gp]).data
        B = np.zeros((self._nVoigt, self._nDOF))

        if nDim == 2:
            for a in range(self._nNodes):
                for i in range(2):
                    col = a * 2 + i
                    B[0, col] = F[i, 0] * dN[a, 0]                          # E11
                    B[1, col] = F[i, 1] * dN[a, 1]                          # E22
                    B[2, col] = F[i, 0] * dN[a, 1] + F[i, 1] * dN[a, 0]    # 2*E12
        else:
            for a in range(self._nNodes):
                for i in range(3):
                    col = a * 3 + i
                    B[0, col] = F[i, 0] * dN[a, 0]                          # E11
                    B[1, col] = F[i, 1] * dN[a, 1]                          # E22
                    B[2, col] = F[i, 2] * dN[a, 2]                          # E33
                    B[3, col] = F[i, 0] * dN[a, 1] + F[i, 1] * dN[a, 0]    # 2*E12
                    B[4, col] = F[i, 1] * dN[a, 2] + F[i, 2] * dN[a, 1]    # 2*E23
                    B[5, col] = F[i, 0] * dN[a, 2] + F[i, 2] * dN[a, 0]    # 2*E13
        return Matrix(init=B)

    def _buildKgeo(self, gp, stress_ctensor, dN_dX_gp=None):
        """Geometric stiffness from 2PK stress. Returns Matrix (nDOF x nDOF) — unweighted.

        stress_ctensor: CTensor (2nd order, CONTR) from material.
        CONTR stores engineering shear (x2 factor) -> divide by 2 for tensor shear in S_mat.

        K_sigma[a*nDim+i, b*nDim+j] = delta_ij * (dN_a)^T @ S_mat @ (dN_b)

        :param dN_dX_gp: optional Matrix override. UL passes dN_dX_original.

        Uses .data for numpy indexing per Design Rule 10 amendment.
        """
        nDim = self._nDim
        S_vec = stress_ctensor.make_vector()
        S_mat = np.zeros((nDim, nDim))

        if nDim == 2:
            S_mat[0, 0] = S_vec[0]
            S_mat[1, 1] = S_vec[1]
            S_mat[0, 1] = S_mat[1, 0] = S_vec[2] / 2.0
        else:
            S_mat[0, 0] = S_vec[0]
            S_mat[1, 1] = S_vec[1]
            S_mat[2, 2] = S_vec[2]
            S_mat[0, 1] = S_mat[1, 0] = S_vec[3] / 2.0
            S_mat[1, 2] = S_mat[2, 1] = S_vec[4] / 2.0
            S_mat[0, 2] = S_mat[2, 0] = S_vec[5] / 2.0

        dN = (dN_dX_gp if dN_dX_gp is not None else self._dN_dX[gp]).data
        Kg = np.zeros((self._nDOF, self._nDOF))
        for a in range(self._nNodes):
            for b in range(self._nNodes):
                val = dN[a] @ S_mat @ dN[b]
                for i in range(nDim):
                    Kg[a * nDim + i, b * nDim + i] = val
        return Matrix(init=Kg)
