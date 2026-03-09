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
# Version: 0.3
#
# UpdatedLagrangianContinuumKinematics (v2 architecture — element API path)
#   Reference = last committed config. commitState() updates reference coords
#   and recomputes dN_dX.
#
#   Key design: UL computes F_incr from incremental displacement and updated
#   dN_dX, then recovers F_total = F_incr @ F_commit. All strain, B matrix,
#   and geometric stiffness are computed from F_total and dN_dX_original
#   (same as TL). This guarantees TL == UL to machine precision.
#
#   The UL advantage is that u_incr (smaller numbers) is used for the Newton
#   linearization, giving better conditioning at large deformations.

from .nonlinear_base import _NonlinearContinuumBase
from ....._systools.data import Vector, Matrix
from ....._systools.backend import np


class UpdatedLagrangianContinuumKinematics(_NonlinearContinuumBase):
    """Updated Lagrangian (Bathe) kinematics for continuum elements.

    Reference configuration is the last committed (converged) configuration.
    commitState() updates the reference coords and recomputes shape function
    derivatives dN_dX. Displacements in the next step are incremental from
    the new reference.

    Internally, F_total = F_incr @ F_commit is recovered, and all strain/stress
    quantities are computed from the original reference (ensuring TL == UL).

    v2: When element= kwarg is present, uses element API for original-reference
    geometry lookups (get_dN_dX). Still manages its own updated-reference state
    (_dN_dX, _X_ref, _F_commit) since these are kinematics state, not element
    geometry.
    """

    formulation = 'updatedLagrangian'

    @property
    def needs_incremental_u(self):
        return True

    def initialize(self, nGP, nDim, nNodes, dN_dX_list, **kwargs):
        """Extra kwargs: dN_dxi_list (list of numpy), X_ref (Matrix nNodes x nDim)."""
        super().initialize(nGP, nDim, nNodes, dN_dX_list, **kwargs)
        self._element = kwargs.get('element', None)
        self._dN_dxi_list = kwargs['dN_dxi_list']
        self._X_ref = kwargs['X_ref']

        if self._element is not None:
            # v2 path: compute original dN_dX from element API
            gauss_points = self._element.get_gauss_points()
            self._gp_coords = [gp_tuple[:-1] for gp_tuple in gauss_points]
            self._dN_dX_original = [
                Matrix(init=self._element.get_dN_dX(xi))
                for xi in self._gp_coords
            ]
        else:
            # Legacy path: copy from initial dN_dX
            self._dN_dX_original = [Matrix(init=m.data.copy()) for m in self._dN_dX]

        # Committed deformation gradient per GP (total from original config)
        self._F_commit = [Matrix(init=np.eye(nDim)) for _ in range(nGP)]
        self._F_commit_backup = [Matrix(init=np.eye(nDim)) for _ in range(nGP)]

        # Committed state for revert (deep copy — pitfall 8)
        self._dN_dX_committed = [Matrix(init=m.data.copy()) for m in self._dN_dX]
        self._X_ref_committed = Matrix(init=self._X_ref.data.copy())
        self._detJ = [None] * nGP
        self._detJ_committed = [None] * nGP

    def update(self, gp, u_e):
        """Compute F_total, E_total, B_total at this GP.

        u_e is incremental displacement from committed config.
        F_incr is computed using updated dN_dX (from committed config).
        F_total = F_incr @ F_commit recovers the total deformation gradient.
        Strain, B matrix, and geometric stiffness use F_total and dN_dX_original.

        v2 path: adds incompatible mode enrichment from element API
        (alpha_mat @ dM_dX) to H_total before computing strain and B_NL.
        """
        # Incremental quantities from updated reference
        H_incr = self._computeH(gp, u_e)
        F_incr = np.eye(self._nDim) + H_incr.data

        # Total deformation gradient from original reference
        F_total = F_incr @ self._F_commit[gp].data
        H_total = F_total - np.eye(self._nDim)

        # v2 path: add incompatible mode enrichment from element API
        if self._element is not None:
            xi = self._gp_coords[gp]
            H_enrich = self._element.get_H_enrichment(xi)
            if H_enrich is not None:
                H_total = H_total + H_enrich
                F_total = np.eye(self._nDim) + H_total

        H_total_M = Matrix(init=H_total)
        self._F[gp] = Matrix(init=F_total)
        self._strain[gp] = self._EtoVoigtCOV(self._computeGreenLagrange(H_total_M))
        self._B_NL[gp] = self._buildBNL(gp, H_total_M, self._dN_dX_original[gp])

    def getStrain(self, gp):
        return self._strain[gp]

    def getBMatrix(self, gp):
        return self._B_NL[gp]

    def getF(self, gp):
        return self._F[gp]

    def getDetJ(self, gp):
        # Return None: element keeps original detJ for volume integration
        # (consistent with B and strain from original reference)
        return None

    def getGeometricStiffness(self, gp, stress):
        return self._buildKgeo(gp, stress, self._dN_dX_original[gp])

    def commitState(self, **kwargs):
        """Update reference config to current deformed. Recompute dN_dX per GP.
        Update F_commit to F_total (current trial).

        :param X_current: Matrix (nNodes x nDim) — current nodal positions.
        """
        from ....element.continuum.base import compute_physical_derivatives

        X_current = kwargs.get('X_current', None)
        if X_current is None:
            return

        # Save committed state for revert BEFORE updating (deep copy)
        self._X_ref_committed = Matrix(init=self._X_ref.data.copy())
        self._dN_dX_committed = [Matrix(init=m.data.copy()) for m in self._dN_dX]
        self._detJ_committed = list(self._detJ)
        self._F_commit_backup = [Matrix(init=m.data.copy()) for m in self._F_commit]

        # Update F_commit to current F_total
        for gp in range(self._nGP):
            if self._F[gp] is not None:
                self._F_commit[gp] = Matrix(init=self._F[gp].data.copy())

        # Update reference to current deformed config (for next step's F_incr)
        self._X_ref = X_current
        for gp in range(self._nGP):
            dN_dX_new, detJ_new = compute_physical_derivatives(
                self._dN_dxi_list[gp], X_current.data)
            self._dN_dX[gp] = Matrix(init=dN_dX_new)
            self._detJ[gp] = detJ_new

    def revertToLastCommit(self):
        """Restore reference config to last committed."""
        self._X_ref = Matrix(init=self._X_ref_committed.data.copy())
        self._dN_dX = [Matrix(init=m.data.copy()) for m in self._dN_dX_committed]
        self._detJ = list(self._detJ_committed)
        self._F_commit = [Matrix(init=m.data.copy()) for m in self._F_commit_backup]

    def getK(self, element):
        """Assemble element tangent stiffness K = K_mat + K_geo."""
        nDOF = element.get_nDOF_total()
        t = element.get_thickness()
        K_mat = Matrix(shape=[nDOF, nDOF])
        K_geo = Matrix(shape=[nDOF, nDOF])
        for gp in range(self._nGP):
            B = self.getBMatrix(gp)
            C_mat = element.get_tangent(gp).to_matrix()
            detJ, w = element.get_gp_weight(gp)
            dV = detJ * w * t
            K_mat += B.T @ C_mat @ B * dV
            Kg = self.getGeometricStiffness(gp, element.get_stress(gp))
            K_geo += Kg * dV
        return K_mat + K_geo

    def get_f_int(self, element):
        """Assemble element internal force f = sum_gp B_NL^T sigma dV."""
        nDOF = element.get_nDOF_total()
        t = element.get_thickness()
        f = Vector(shape=nDOF)
        for gp in range(self._nGP):
            B = self.getBMatrix(gp)
            sig_vec = element.get_stress(gp).to_vector()
            detJ, w = element.get_gp_weight(gp)
            dV = detJ * w * t
            f += B.T @ sig_vec * dV
        return f

    def copy(self):
        """Return a fresh (uninitialized) copy."""
        return UpdatedLagrangianContinuumKinematics()
