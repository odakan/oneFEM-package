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
# CrdTransf base class
#   Maps element-local frame to global frame.
#   OpenSees: CrdTransf (Linear2d/3d, Corotational2d/3d, PDelta2d/3d).

from ....._systools.data import Vector, Matrix
from ..base import Kinematics


class CrdTransf(Kinematics):
    """
    Coordinate Transformation base class.
    Maps element-local (basic) frame to global frame for beam/frame elements.
    """

    def __init__(self):
        pass

    def initialize(self, node_i, node_j):
        """Compute initial local axes from nodal coordinates."""
        raise NotImplementedError("CrdTransf.initialize(): must be implemented by subclasses.")

    def update(self):
        """Recompute axes after deformation (needed for Corotational)."""
        raise NotImplementedError("CrdTransf.update(): must be implemented by subclasses.")

    def getInitialLength(self):
        """Return the initial (undeformed) element length."""
        raise NotImplementedError("CrdTransf.getInitialLength(): must be implemented by subclasses.")

    def getDeformedLength(self):
        """Return the deformed element length."""
        raise NotImplementedError("CrdTransf.getDeformedLength(): must be implemented by subclasses.")

    def getBasicTrialDisp(self):
        """Return displacements in basic (local, rigid-body-removed) frame."""
        raise NotImplementedError("CrdTransf.getBasicTrialDisp(): must be implemented by subclasses.")

    def getBasicTrialVel(self):
        """Return velocities in basic frame."""
        raise NotImplementedError("CrdTransf.getBasicTrialVel(): must be implemented by subclasses.")

    def getBasicTrialAccel(self):
        """Return accelerations in basic frame."""
        raise NotImplementedError("CrdTransf.getBasicTrialAccel(): must be implemented by subclasses.")

    def getGlobalResistingForce(self, basic_force, p0):
        """Transform local resisting force to global frame.
        :param basic_force: Vector of basic forces [N, Mi, Mj]
        :param p0: Vector of element load contributions in global coords (or None)
        """
        raise NotImplementedError("CrdTransf.getGlobalResistingForce(): must be implemented by subclasses.")

    def getGlobalStiffMatrix(self, basic_stiff, basic_force):
        """Transform local tangent stiffness to global frame.
        Includes geometric stiffness for Corotational/PDelta.
        :param basic_stiff: Matrix of basic stiffness (3x3)
        :param basic_force: Vector of basic forces [N, Mi, Mj]
        """
        raise NotImplementedError("CrdTransf.getGlobalStiffMatrix(): must be implemented by subclasses.")

    def getInitialGlobalStiffMatrix(self, basic_stiff):
        """Transform initial basic stiffness to global frame (no geometric terms)."""
        raise NotImplementedError("CrdTransf.getInitialGlobalStiffMatrix(): must be implemented by subclasses.")

    def copy(self):
        """Return a copy of this transformation."""
        raise NotImplementedError("CrdTransf.copy(): must be implemented by subclasses.")
