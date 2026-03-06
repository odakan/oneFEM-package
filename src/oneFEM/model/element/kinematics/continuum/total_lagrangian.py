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
# TotalLagrangianContinuumKinematics
#   Reference = always initial config. No state to commit/revert.
#   Strain: Green-Lagrange E = 0.5*(F^T F - I)
#   Stress: 2nd Piola-Kirchhoff S (from material via C:E)

from .nonlinear_base import _NonlinearContinuumBase
from ....._systools.data import Matrix


class TotalLagrangianContinuumKinematics(_NonlinearContinuumBase):
    """Total Lagrangian kinematics for continuum elements.

    Reference configuration is always the initial (undeformed) configuration.
    No state needs committing or reverting.
    """

    formulation = 'totalLagrangian'

    def update(self, gp, u_e):
        """Compute F, E, B_NL at this GP."""
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

    def getGeometricStiffness(self, gp, stress):
        return self._buildKgeo(gp, stress)

    def copy(self):
        c = TotalLagrangianContinuumKinematics()
        if hasattr(self, '_nGP') and self._nGP is not None:
            c.initialize(self._nGP, self._nDim, self._nNodes,
                         [m for m in self._dN_dX])
        return c
