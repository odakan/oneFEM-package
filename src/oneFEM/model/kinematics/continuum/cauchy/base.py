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
# Version: 0.2
#
# ContinuumKinematics base class (Option B: single object per element, gp-indexed)
#   Strategy for geometric nonlinearity at integration points of
#   continuum (solid) elements (Quad4, Tri3, Brick8).

from ...base import Kinematics
from ....physics_family import PhysicsFamily


class ContinuumKinematics(Kinematics):
    """Base class for integration-point-level kinematic strategies.

    Option B architecture: single instance per element, methods take gp index.
    The element calls initialize() once during _domain() with all GP data.
    After that, update/getStrain/getBMatrix/getGeometricStiffness take a GP
    index and operate on that GP's cached state.

    Subclasses implement specific strain measures and B matrices for:
      Linear, Total Lagrangian, Updated Lagrangian, Corotational.
    """

    physics_family = PhysicsFamily.CONTINUUM_CAUCHY
    formulation = None

    def initialize(self, nGP, nDim, nNodes, dN_dX_list, **kwargs):
        """Pre-allocate per-GP arrays. Called once in element._domain().

        :param nGP: number of Gauss points
        :param nDim: spatial dimension (2 or 3)
        :param nNodes: nodes per element
        :param dN_dX_list: list of Matrix (nNodes x nDim), one per GP
        :param kwargs: UL passes dN_dxi_list, X_ref. Ignored by Linear/TL.
        """
        self._nGP = nGP
        self._nDim = nDim
        self._nNodes = nNodes
        self._nVoigt = 3 if nDim == 2 else 6
        self._nDOF = nNodes * nDim
        self._dN_dX = list(dN_dX_list)

    def getStrain(self, gp):
        """Return CTensor (2nd order, COV) at GP index gp."""
        raise NotImplementedError("ContinuumKinematics.getStrain()")

    def getBMatrix(self, gp):
        """Return Matrix (nVoigt x nDOF) at GP index gp."""
        raise NotImplementedError("ContinuumKinematics.getBMatrix()")

    def getF(self, gp):
        """Return deformation gradient Matrix (nDim x nDim). None for linear."""
        return None

    def getDetJ(self, gp):
        """Return updated detJ after reference config change. None for linear/TL."""
        return None

    def getGeometricStiffness(self, gp, stress):
        """Return Matrix (nDOF x nDOF) — unweighted. Element applies detJ*w*t."""
        raise NotImplementedError("ContinuumKinematics.getGeometricStiffness()")

    def update(self, gp, u_e):
        """Update GP state for current trial displacements.

        :param gp: Gauss point index
        :param u_e: Vector — element nodal displacements (flat, nDOF)
        """
        pass

    def applyCorotFrame(self, u_e):
        """Apply corotational frame extraction. No-op for non-corotational."""
        pass

    def transformToGlobal(self, K_mat, f):
        """Transform material stiffness and force to global frame.
        No-op for non-corotational. Returns (K_mat, f) unchanged."""
        return K_mat, f

    def commitState(self, **kwargs):
        """Commit current trial state. UL extracts X_current from kwargs."""
        pass

    def revertToLastCommit(self):
        """Revert trial state to last committed. Override in UL."""
        pass

    @property
    def needs_incremental_u(self):
        """Whether this formulation expects incremental displacement. Override in UL."""
        return False

    def _setMaterialStrain(self, material, strain):
        """Push strain to material. Override in UL to call _setTrialStrainIncr."""
        material._setTrialStrain(strain)

    def getK(self, element):
        """Assemble element tangent stiffness matrix from all GPs.
        Subclasses override. Default raises NotImplementedError."""
        raise NotImplementedError("ContinuumKinematics.getK()")

    def get_f_int(self, element):
        """Assemble element internal force vector from all GPs.
        Subclasses override. Default raises NotImplementedError."""
        raise NotImplementedError("ContinuumKinematics.get_f_int()")

    def copy(self):
        """Return a deep copy of this kinematics object."""
        raise NotImplementedError("ContinuumKinematics.copy()")
