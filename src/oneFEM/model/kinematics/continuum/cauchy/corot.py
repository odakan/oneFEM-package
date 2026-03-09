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
# Date: 09/03/2026
# Version: 1.1
#
# CorotContinuumKinematics (v2 architecture — element API path)
#   EICR — Felippa & Haugen 2005.
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

    v2: When initialized with element= kwarg, uses element API for geometry
    access. Loops over range(nNodes) instead of hardcoded 4-node unpacking
    in _extract_R_2d.
    """

    formulation = 'corotational'

    def initialize(self, nGP, nDim, nNodes, dN_dX_list, X_ref=None, **kwargs):
        """Initialize per-GP arrays, pre-build linear B matrices, init rotation."""
        super().initialize(nGP, nDim, nNodes, dN_dX_list, **kwargs)

        self._element = kwargs.get('element', None)

        if self._element is not None:
            # v2 path: get reference coords from element API
            X_ref_np = self._element.get_coords_ref()
            gauss_points = self._element.get_gauss_points()
            self._gp_coords = [gp_tuple[:-1] for gp_tuple in gauss_points]
        else:
            if X_ref is None:
                raise ValueError(
                    "CorotContinuumKinematics.initialize: X_ref (reference node "
                    "coordinates) must be provided.")
            X_ref_np = X_ref.data if hasattr(X_ref, 'data') else np.array(X_ref, dtype=float)

        self._X_ref = np.array(X_ref_np, dtype=float)
        self._C0 = self._X_ref.mean(axis=0)

        # Pre-build constant linear B matrices per GP (reference config, never changes).
        self._B_local = []
        if self._element is not None:
            for xi in self._gp_coords:
                self._B_local.append(self._element.get_B(xi))
        else:
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

        Uses loop over nNodes instead of hardcoded 4-node unpacking.

        alpha = atan2(f21 - f12, f11 + f22)

        :param u_e_np: (nDOF,) numpy array of nodal displacements
        :return: (2,2) numpy rotation matrix R
        """
        nNodes = self._nNodes
        aX = self._X_ref[:, 0] - self._C0[0]
        aY = self._X_ref[:, 1] - self._C0[1]

        x_cur = self._X_ref + u_e_np.reshape(nNodes, 2)
        C_cur = x_cur.mean(axis=0)
        bX = x_cur[:, 0] - C_cur[0]
        bY = x_cur[:, 1] - C_cur[1]

        # Petracca formula requires exactly 4 nodes for the closed-form
        # For other node counts, fall back to SVD-based extraction
        if nNodes != 4:
            return self._extract_R_2d_svd(u_e_np)

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

    def _extract_R_2d_svd(self, u_e_np):
        """Fallback 2D rotation extraction via SVD for non-4-node elements."""
        F_mean = np.zeros((2, 2))
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
        """
        nDim = self._nDim
        nNodes = self._nNodes
        if x_cur is None:
            x_cur = self._X_ref + u_e_np.reshape(nNodes, nDim)
        C_cur = x_cur.mean(axis=0)

        u_local = np.zeros(nNodes * nDim)
        for I in range(nNodes):
            X0_I = self._X_ref[I] - self._C0
            x_I  = x_cur[I]       - C_cur
            u_local[nDim*I : nDim*I+nDim] = R.T @ x_I - X0_I
        return u_local

    # ------------------------------------------------------------------
    # Main kinematics interface
    # ------------------------------------------------------------------

    def update(self, gp, u_e):
        """Cache F[gp] for later use in 3D polar decomp."""
        if self._element is not None:
            xi = self._gp_coords[gp]
            u_data = u_e.data if hasattr(u_e, 'data') else np.asarray(u_e)
            F_np = self._element.get_F(xi, u_data)
            self._F[gp] = Matrix(init=F_np)
        else:
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
        """Build geometric stiffness K_sigma using stress rotated to global frame."""
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

        S_global = self._R @ S_local @ self._R.T

        if self._nDim == 2:
            sig_data = [S_global[0, 0], S_global[1, 1],
                        2.0 * S_global[0, 1]]
        else:
            sig_data = [S_global[0, 0], S_global[1, 1], S_global[2, 2],
                        2.0 * S_global[0, 1],
                        2.0 * S_global[1, 2],
                        2.0 * S_global[0, 2]]
        sig_global = CTensor(sig_data, self._nVoigt, CTensor.CONTR)

        if self._element is not None:
            xi = self._gp_coords[gp]
            dN_dX = Matrix(init=self._element.get_dN_dX(xi))
            return self._buildKgeo(gp, sig_global, dN_dX)
        return self._buildKgeo(gp, sig_global)

    def transformToGlobal(self, K_mat, f):
        """Rotate K_mat and f from corotated frame to global frame.

        T = block_diag(R, R, ..., R)  — block-diagonal rotation, one R per node.
        K_global[I,J] = R @ K_local[I,J] @ R.T
        f_global[I]   = R @ f_local[I]
        """
        R    = self._R
        nDim = self._nDim
        nNodes = self._nNodes

        f_data = f.data.copy()
        for I in range(nNodes):
            sl = slice(nDim * I, nDim * I + nDim)
            f_data[sl] = R @ f_data[sl]

        K_data = K_mat.data.copy()
        for I in range(nNodes):
            for J in range(nNodes):
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

    def getK(self, element):
        """Assemble element tangent stiffness with corotational transform.
        K = T^T K_mat T + K_geo (K_geo already in global frame)."""
        nDOF = element.get_nDOF_total()
        t = element._thickness
        K_mat = Matrix(shape=[nDOF, nDOF])
        K_geo = Matrix(shape=[nDOF, nDOF])
        for gp in range(self._nGP):
            B = self.getBMatrix(gp)
            C_mat = element.get_tangent(gp).to_matrix()
            detJ, w = element._gp_data[gp]
            dV = detJ * w * t
            K_mat += B.T @ C_mat @ B * dV
            Kg = self.getGeometricStiffness(gp, element.get_stress(gp))
            K_geo += Kg * dV
        K_mat_global, _ = self.transformToGlobal(K_mat, Vector(shape=nDOF))
        return K_mat_global + K_geo

    def get_f_int(self, element):
        """Assemble element internal force with corotational transform."""
        nDOF = element.get_nDOF_total()
        t = element._thickness
        f = Vector(shape=nDOF)
        for gp in range(self._nGP):
            B = self.getBMatrix(gp)
            sig_vec = element.get_stress(gp).to_vector()
            detJ, w = element._gp_data[gp]
            dV = detJ * w * t
            f += B.T @ sig_vec * dV
        _, f_global = self.transformToGlobal(Matrix(shape=[nDOF, nDOF]), f)
        return f_global

    def copy(self):
        """Return a fresh (uninitialized) copy."""
        return CorotContinuumKinematics()
