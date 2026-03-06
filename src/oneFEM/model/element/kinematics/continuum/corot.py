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
# Version: 1.0
#
# CorotContinuumKinematics (EICR — Felippa & Haugen 2005)
#   Corotational formulation for continuum elements.
#
#   Frame extraction follows Petracca (ASDEA/OpenSees),
#   ASDShellQ4CorotationalTransformation.h, USE_POLAR_DECOMP_ALLIGN path.
#
#   2D (Quad4): exact closed-form polar decomp via atan2 formula.
#   3D (Hex8):  polar decomp of mean F via SVD.
#
#   Because G = 0 exactly for polar-decomp frame alignment (confirmed in
#   Petracca source), the tangent reduces to T'*K_mat*T + K_sigma with no
#   P/S/H/G spin machinery needed.

import math
from .nonlinear_base import _NonlinearContinuumBase
from .linear import LinearContinuumKinematics
from ....._systools.data import Vector, Matrix
from ....._systools.data.ctensor import CTensor
from ....._systools.backend import np


class CorotContinuumKinematics(_NonlinearContinuumBase):
    """Corotational (EICR) kinematics for continuum elements.

    Extracts rigid rotation R per element via polar decomposition of the
    element deformation, applies deformation in a locally corotated frame.
    Zero strain for pure rigid-body rotations (translation + rotation).

    Inherits from _NonlinearContinuumBase for:
      - _computeH(gp, u_e): displacement gradient at GP
      - _computeF(H): deformation gradient F = I + H
      - _buildKgeo(gp, stress): geometric stiffness K_sigma
      - Per-GP _F, _strain arrays

    Uses LinearContinuumKinematics._buildBMatrix() for constant linear B.
    """

    formulation = 'corotational'

    def initialize(self, nGP, nDim, nNodes, dN_dX_list, X_ref=None, **kwargs):
        """Initialize per-GP arrays, pre-build linear B matrices, init rotation.

        :param X_ref: Matrix or numpy array (nNodes, nDim) — reference node coordinates.
                      Required for the correct deformational displacement formula.
        """
        super().initialize(nGP, nDim, nNodes, dN_dX_list, **kwargs)

        if X_ref is None:
            raise ValueError(
                "CorotContinuumKinematics.initialize: X_ref (reference node "
                "coordinates) must be provided. Pass the element's reference "
                "coordinate array of shape (nNodes, nDim).")
        X_ref_np = X_ref.data if hasattr(X_ref, 'data') else np.array(X_ref, dtype=float)
        self._X_ref = np.array(X_ref_np, dtype=float)
        self._C0 = self._X_ref.mean(axis=0)

        # Pre-build constant linear B matrices per GP (reference config, never changes).
        self._B_local = []
        for gp in range(nGP):
            self._B_local.append(
                LinearContinuumKinematics._buildBMatrix(dN_dX_list[gp], nDim))

        # Rotation state stored as raw numpy arrays for efficiency.
        self._R           = np.eye(nDim)
        self._R_committed = np.eye(nDim)

    # ------------------------------------------------------------------
    # EICR frame extraction helpers
    # ------------------------------------------------------------------

    def _extract_R_2d(self, u_e_np):
        """Extract 2D corotational rotation via Petracca's exact atan2 formula.

        Direct translation of createLocalCoordinateSystem() (USE_POLAR_DECOMP_ALLIGN)
        from ASDShellQ4CorotationalTransformation.h.

        alpha = atan2(f21 - f12, f11 + f22)

        :param u_e_np: (nDOF,) numpy array of nodal displacements
        :return: (2,2) numpy rotation matrix R
        """
        aX = self._X_ref[:, 0] - self._C0[0]
        aY = self._X_ref[:, 1] - self._C0[1]

        x_cur = self._X_ref + u_e_np.reshape(self._nNodes, 2)
        C_cur = x_cur.mean(axis=0)
        bX = x_cur[:, 0] - C_cur[0]
        bY = x_cur[:, 1] - C_cur[1]

        aX1, aX2, aX3, aX4 = aX[0], aX[1], aX[2], aX[3]
        aY1, aY2, aY3, aY4 = aY[0], aY[1], aY[2], aY[3]
        bX1, bX2, bX3, bX4 = bX[0], bX[1], bX[2], bX[3]
        bY1, bY2, bY3, bY4 = bY[0], bY[1], bY[2], bY[3]

        C1 = 1.0 / (aX1*aY2 - aX2*aY1 - aX1*aY4 + aX2*aY3
                   - aX3*aY2 + aX4*aY1 + aX3*aY4 - aX4*aY3)
        C2 = (bY1 + bY2 - bY3 - bY4) / 4.0
        C3 = (bY1 - bY2 - bY3 + bY4) / 4.0
        C4 = (bX1 + bX2 - bX3 - bX4) / 4.0
        C5 = (bX1 - bX2 - bX3 + bX4) / 4.0
        C6 =  aX1 + aX2 - aX3 - aX4
        C7 =  aX1 - aX2 - aX3 + aX4
        C8 =  aY1 + aY2 - aY3 - aY4
        C9 =  aY1 - aY2 - aY3 + aY4

        f11 = 2.0 * C1 * (C5*C8 - C4*C9)
        f12 = 2.0 * C1 * (C4*C7 - C5*C6)
        f21 = 2.0 * C1 * (C3*C8 - C2*C9)
        f22 = 2.0 * C1 * (C2*C7 - C3*C6)

        alpha = math.atan2(f21 - f12, f11 + f22)
        ca, sa = math.cos(alpha), math.sin(alpha)
        return np.array([[ca, -sa], [sa,  ca]])

    def _extract_R_3d(self):
        """Extract 3D corotational rotation via SVD of mean deformation gradient.

        F_mean = (1/nGP) * sum_gp F[gp]
        F_mean = V @ diag(S) @ Wt  ->  R = V @ Wt  (polar decomposition)

        :return: (3,3) numpy rotation matrix R
        """
        F_mean = np.zeros((3, 3))
        for gp in range(self._nGP):
            f_gp = self._F[gp]
            F_mean += f_gp.data if hasattr(f_gp, 'data') else f_gp
        F_mean /= self._nGP

        V, S, Wt = np.linalg.svd(F_mean)
        R = V @ Wt
        if np.linalg.det(R) < 0:
            V[:, -1] *= -1
            R = V @ Wt
        return R

    def _get_u_local(self, u_e_np, R, x_cur=None):
        """Compute deformational local displacements in corotated frame.

        Follows Petracca's calculateLocalDisplacements():
          u_local[I] = R.T @ (x_cur[I] - C_cur) - (X_ref[I] - C0)

        :param u_e_np: (nDOF,) numpy array of global nodal displacements
        :param R: (nDim, nDim) numpy corotational rotation matrix
        :param x_cur: optional precomputed (nNodes, nDim) current positions
        :return: (nDOF,) numpy array of local deformational displacements
        """
        nDim = self._nDim
        if x_cur is None:
            x_cur = self._X_ref + u_e_np.reshape(self._nNodes, nDim)
        C_cur = x_cur.mean(axis=0)

        u_local = np.zeros(self._nNodes * nDim)
        for I in range(self._nNodes):
            X0_I = self._X_ref[I] - self._C0
            x_I  = x_cur[I]       - C_cur
            u_local[nDim*I : nDim*I+nDim] = R.T @ x_I - X0_I
        return u_local

    # ------------------------------------------------------------------
    # Main kinematics interface
    # ------------------------------------------------------------------

    def update(self, gp, u_e):
        """Cache F[gp] for later use in 3D polar decomp."""
        H = self._computeH(gp, u_e)
        self._F[gp] = self._computeF(H)

    def applyCorotFrame(self, u_e):
        """Core EICR: extract R, compute deformational local displacements,
        compute corotated linear strain at all GPs.

        Called once per Newton iteration after all per-GP update() calls.
        """
        nDim    = self._nDim
        u_e_np  = u_e.data if hasattr(u_e, 'data') else np.array(u_e)

        # Step 1 — Extract corotational rotation R
        if nDim == 2:
            R = self._extract_R_2d(u_e_np)
        else:
            R = self._extract_R_3d()
        self._R = R

        # Step 2 — Deformational local displacements (Petracca's formula with centring)
        x_cur = self._X_ref + u_e_np.reshape(self._nNodes, nDim)
        u_local_np = self._get_u_local(u_e_np, R, x_cur=x_cur)

        # Step 3 — Linear strain per GP in corotated frame
        u_local_vec = Vector(u_local_np)
        for gp in range(self._nGP):
            eps_vec = self._B_local[gp] @ u_local_vec
            self._strain[gp] = CTensor(eps_vec.data.tolist(),
                                       self._nVoigt, CTensor.COV)

    def getStrain(self, gp):
        """Return cached corotated strain CTensor (2nd order, COV)."""
        return self._strain[gp]

    def getBMatrix(self, gp):
        """Return constant linear B matrix (reference config, never changes)."""
        return self._B_local[gp]

    def getF(self, gp):
        """Return deformation gradient at GP."""
        return self._F[gp]

    def getDetJ(self, gp):
        """Corot keeps reference config — detJ never changes. Return None."""
        return None

    def getGeometricStiffness(self, gp, stress):
        """Build geometric stiffness K_sigma using stress rotated to global frame.

        The material returns stress in the corotated frame (sigma_local).
        K_sigma is computed from reference dN_dX (global frame), so stress
        must be rotated to global first.

        K_sigma is NOT rotated by transformToGlobal() — only K_mat is rotated.
        """
        S_vec = stress.make_vector()
        S_local = np.zeros((self._nDim, self._nDim))
        if self._nDim == 2:
            S_local[0, 0] = S_vec[0]
            S_local[1, 1] = S_vec[1]
            S_local[0, 1] = S_local[1, 0] = S_vec[2] / 2.0
        else:
            S_local[0, 0] = S_vec[0]
            S_local[1, 1] = S_vec[1]
            S_local[2, 2] = S_vec[2]
            S_local[0, 1] = S_local[1, 0] = S_vec[3] / 2.0
            S_local[1, 2] = S_local[2, 1] = S_vec[4] / 2.0
            S_local[0, 2] = S_local[2, 0] = S_vec[5] / 2.0

        # Rotate stress to global: sigma_global = R @ sigma_local @ R.T
        S_global = self._R @ S_local @ self._R.T

        # Wrap back as CTensor (CONTR) — engineering shear (factor 2 on off-diags)
        if self._nDim == 2:
            sig_data = [S_global[0, 0], S_global[1, 1],
                        2.0 * S_global[0, 1]]
        else:
            sig_data = [S_global[0, 0], S_global[1, 1], S_global[2, 2],
                        2.0 * S_global[0, 1],
                        2.0 * S_global[1, 2],
                        2.0 * S_global[0, 2]]
        sig_global = CTensor(sig_data, self._nVoigt, CTensor.CONTR)

        return self._buildKgeo(gp, sig_global)

    def transformToGlobal(self, K_mat, f):
        """Rotate K_mat and f from corotated frame to global frame.

        T = block_diag(R, R, ..., R)  — block-diagonal rotation, one R per node.
        K_global[I,J] = R @ K_local[I,J] @ R.T
        f_global[I]   = R @ f_local[I]
        """
        R    = self._R
        nDim = self._nDim

        f_data = f.data.copy()
        for I in range(self._nNodes):
            sl = slice(nDim * I, nDim * I + nDim)
            f_data[sl] = R @ f_data[sl]

        K_data = K_mat.data.copy()
        for I in range(self._nNodes):
            for J in range(self._nNodes):
                si = slice(nDim * I, nDim * I + nDim)
                sj = slice(nDim * J, nDim * J + nDim)
                K_data[si, sj] = R @ K_data[si, sj] @ R.T

        return Matrix(init=K_data), Vector(f_data)

    def commitState(self, **kwargs):
        """Commit rotation. Corot does NOT update reference config (unlike UL)."""
        super().commitState(**kwargs)
        self._R_committed = self._R.copy()

    def revertToLastCommit(self):
        """Revert to last committed rotation."""
        self._R = self._R_committed.copy()

    def copy(self):
        """Return a deep copy of this kinematics object, including rotation state."""
        c = CorotContinuumKinematics()
        if hasattr(self, '_nGP') and self._nGP is not None:
            c.initialize(self._nGP, self._nDim, self._nNodes,
                         self._dN_dX,
                         X_ref=self._X_ref.copy())
            c._R           = self._R.copy()
            c._R_committed = self._R_committed.copy()
        return c
