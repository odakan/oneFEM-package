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
# UpdatedLagrangianContinuumKinematics (Bathe formulation)
#   Reference = last committed config. commitState() updates reference coords
#   and recomputes dN_dX.
#   Strain: incremental Green-Lagrange E = 0.5*(F^T F - I) relative to committed
#   Stress: 2nd Piola-Kirchhoff S (from material via C:E)

from .nonlinear_base import _NonlinearContinuumBase
from ....._systools.data import Matrix
from ....._systools.backend import np


class UpdatedLagrangianContinuumKinematics(_NonlinearContinuumBase):
    """Updated Lagrangian (Bathe) kinematics for continuum elements.

    Reference configuration is the last committed (converged) configuration.
    commitState() updates the reference coords and recomputes shape function
    derivatives dN_dX. Displacements in the next step are incremental from
    the new reference.
    """

    formulation = 'updatedLagrangian'

    def initialize(self, nGP, nDim, nNodes, dN_dX_list, **kwargs):
        """Extra kwargs: dN_dxi_list (list of numpy), X_ref (Matrix nNodes x nDim)."""
        super().initialize(nGP, nDim, nNodes, dN_dX_list, **kwargs)
        self._dN_dxi_list = kwargs['dN_dxi_list']
        self._X_ref = kwargs['X_ref']

        # Committed state for revert (deep copy — pitfall 8)
        self._dN_dX_committed = [Matrix(init=m.data.copy()) for m in self._dN_dX]
        self._X_ref_committed = Matrix(init=self._X_ref.data.copy())
        self._detJ = [None] * nGP
        self._detJ_committed = [None] * nGP

    def update(self, gp, u_e):
        """Compute incremental F, E, B_NL relative to committed reference."""
        H = self._computeH(gp, u_e)
        self._F[gp] = self._computeF(H)
        self._strain[gp] = self._EtoVoigtCOV(self._computeGreenLagrange(H))
        self._B_NL[gp] = self._buildBNL(gp, H)

    def getStrain(self, gp):
        return self._strain[gp]

    def getBMatrix(self, gp):
        return self._B_NL[gp]

    def getF(self, gp):
        return self._F[gp]

    def getDetJ(self, gp):
        return self._detJ[gp]

    def getGeometricStiffness(self, gp, stress):
        return self._buildKgeo(gp, stress)

    def commitState(self, **kwargs):
        """Update reference config to current deformed. Recompute dN_dX per GP.

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

        # Update reference to current deformed config
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

    def copy(self):
        c = UpdatedLagrangianContinuumKinematics()
        if hasattr(self, '_nGP') and self._nGP is not None:
            c.initialize(self._nGP, self._nDim, self._nNodes,
                         [Matrix(init=m.data.copy()) for m in self._dN_dX],
                         dN_dxi_list=self._dN_dxi_list,
                         X_ref=Matrix(init=self._X_ref.data.copy()))
        return c
