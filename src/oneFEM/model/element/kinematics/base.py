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
# Kinematics base class
#   Unified base for all geometric nonlinearity strategies:
#   CrdTransf (beams), ContinuumKinematics (solids), ShellKinematics,
#   ContactKinematics.


class Kinematics(object):
    """Base class for all geometric nonlinearity strategies.

    Provides a formulation tag and trial/committed state interface.
    Subclass hierarchies:
        CrdTransf           — beam elements (Linear, PDelta, Corotational)
        ContinuumKinematics — solid elements (Linear, TL, UL, Corotational)
        ShellKinematics     — shell elements (Linear, EICR Corotational)
        ContactKinematics   — contact zero-length elements
    """

    formulation = None  # 'linear', 'pdelta', 'corotational', 'totalLagrangian', 'updatedLagrangian'

    def initialize(self, *args, **kwargs):
        """Initialize kinematics from element geometry. Called once during element _domain()."""
        pass

    def update(self, *args, **kwargs):
        """Update for current trial state."""
        pass

    def commitState(self, **kwargs):
        """Commit current trial state. Override in UL (updates reference config).
        Base accepts **kwargs so UL's X_current= passes through without branching."""
        pass

    def revertToLastCommit(self):
        """Revert trial state to last committed. Override in UL."""
        pass

    def copy(self):
        """Return a deep copy of this kinematics object."""
        raise NotImplementedError("Kinematics.copy(): must be implemented by subclasses.")
