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
# Date: 06/03/2026
# Version: 0.2
#
# UpdatedLagrangianContinuumKinematics (Bathe formulation)
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
from ....._systools.data import Matrix
from ....._systools.backend import np


class UpdatedLagrangianContinuumKinematics(_NonlinearContinuumBase):
    """Updated Lagrangian (Bathe) kinematics for continuum elements.

    Reference configuration is the last committed (converged) configuration.
    commitState() updates the reference coords and recomputes shape function
    derivatives dN_dX. Displacements in the next step are incremental from
    the new reference.

    Internally, F_total = F_incr @ F_commit is recovered, and all strain/stress
    quantities are computed from the original reference (ensuring TL == UL).
    """

    formulation = 'updatedLagrangian'

    @property
    def needs_incremental_u(self):
        return True

    def initialize(self, nGP, nDim, nNodes, dN_dX_list, **kwargs):
        """Extra kwargs: dN_dxi_list (list of numpy), X_ref (Matrix nNodes x nDim)."""
        super().initialize(nGP, nDim, nNodes, dN_dX_list, **kwargs)
        self._dN_dxi_list = kwargs['dN_dxi_list']
        self._X_ref = kwargs['X_ref']

        # Original reference dN_dX (never updated, used for B and K_geo)
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
        """
        # Incremental quantities from updated reference
        H_incr = self._computeH(gp, u_e)
        F_incr = np.eye(self._nDim) + H_incr.data

        # Total deformation gradient from original reference
        F_total = F_incr @ self._F_commit[gp].data
        H_total = Matrix(init=F_total - np.eye(self._nDim))

        self._F[gp] = Matrix(init=F_total)
        self._strain[gp] = self._EtoVoigtCOV(self._computeGreenLagrange(H_total))
        self._B_NL[gp] = self._buildBNL(gp, H_total, self._dN_dX_original[gp])

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
        from ...continuum.isoparametric import compute_physical_derivatives

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

    def copy(self):
        c = UpdatedLagrangianContinuumKinematics()
        if hasattr(self, '_nGP') and self._nGP is not None:
            c.initialize(self._nGP, self._nDim, self._nNodes,
                         [Matrix(init=m.data.copy()) for m in self._dN_dX],
                         dN_dxi_list=self._dN_dxi_list,
                         X_ref=Matrix(init=self._X_ref.data.copy()))
        return c
