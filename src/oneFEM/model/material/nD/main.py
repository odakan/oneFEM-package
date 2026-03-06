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
# nDMaterial base class
#   Multidimensional material interface using CTensor for strain/stress/tangent.
#   CTensor representation conventions:
#     Strain:  CTensor, 2nd order, COV  (rep=1)
#     Stress:  CTensor, 2nd order, CONTR (rep=2)
#     Tangent: CTensor, 4th order, CONTR (rep=2)

from ..main import Material
from ...._systools.data.ctensor import CTensor


class nDMaterial(Material):
    """Base class for multi-dimensional (nD) materials.

    Subclasses must implement:
      _setTrialStrain(strain)  — accept CTensor (2nd order, COV)
      getStress()              — return CTensor (2nd order, CONTR)
      getTangent()             — return CTensor (4th order, CONTR)
      getInitialTangent()      — return CTensor (4th order, CONTR)
      _commitState()           — deep copy trial → committed
      _revertToLastCommit()    — deep copy committed → trial
      _revertToStart()         — reset to initial state
      getCopy()                — return independent copy
    """

    def __init__(self, mat_id=-1):
        super().__init__(mat_id)

    def _getClassType(self):
        return "nDMaterial"

    def _setTrialStrain(self, strain):
        """Accept CTensor (2nd order, COV)."""
        raise NotImplementedError("nDMaterial._setTrialStrain()")

    def _setTrialStrainIncr(self, strain_incr):
        """Accept incremental strain CTensor (2nd order, COV).
        Default: total = committed + increment, then call _setTrialStrain."""
        if not hasattr(self, '_eps_commit'):
            raise AttributeError(
                "nDMaterial._setTrialStrainIncr() - subclass '{}' does not define "
                "_eps_commit. Override _setTrialStrainIncr() or add committed strain "
                "tracking.".format(type(self).__name__))
        total_strain = self._eps_commit + strain_incr
        self._setTrialStrain(total_strain)

    def _setTrialF(self, F):
        """Accept deformation gradient F (Matrix, nDim x nDim).
        Default: extract Green-Lagrange E = 0.5*(F^T F - I), delegate to _setTrialStrain.
        Override in hyperelastic materials (Neo-Hookean, Mooney-Rivlin) to work with F directly.
        """
        from ...._systools.backend import np
        from ...._systools.data import Matrix

        nDim = F.shape[0]
        I = Matrix(init=np.eye(nDim))
        C_right = F.T @ F
        E_mat = (C_right - I) * 0.5

        nVoigt = 3 if nDim == 2 else 6
        E_data = E_mat.data
        if nDim == 2:
            eps_list = [E_data[0, 0], E_data[1, 1], 2.0 * E_data[0, 1]]
        else:
            eps_list = [E_data[0, 0], E_data[1, 1], E_data[2, 2],
                        2.0 * E_data[0, 1], 2.0 * E_data[1, 2], 2.0 * E_data[0, 2]]
        self._setTrialStrain(CTensor(eps_list, nVoigt, CTensor.COV))

    def getStrain(self):
        """Return CTensor (2nd order, COV)."""
        raise NotImplementedError("nDMaterial.getStrain()")

    def getStress(self):
        """Return CTensor (2nd order, CONTR)."""
        raise NotImplementedError("nDMaterial.getStress()")

    def getTangent(self):
        """Return CTensor (4th order, CONTR)."""
        raise NotImplementedError("nDMaterial.getTangent()")

    def getInitialTangent(self):
        """Return CTensor (4th order, CONTR)."""
        raise NotImplementedError("nDMaterial.getInitialTangent()")

    def _commitState(self):
        raise NotImplementedError("nDMaterial._commitState()")

    def _revertToLastCommit(self):
        raise NotImplementedError("nDMaterial._revertToLastCommit()")

    def _revertToStart(self):
        raise NotImplementedError("nDMaterial._revertToStart()")

    def getCopy(self):
        raise NotImplementedError("nDMaterial.getCopy()")
