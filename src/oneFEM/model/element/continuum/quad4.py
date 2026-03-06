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
# Quad4 — 4-node bilinear quadrilateral element (2D)
#   Nodes: 4 x Node22 (2 translational DOFs each), total 8 DOFs
#   Integration: 2x2 Gauss (full)
#   Material: nDMaterial (PlaneStress or PlaneStrain)
#   Kinematics: injected, defaults to LinearContinuumKinematics

from .base import ContinuumElement
from .isoparametric import quad4_gauss_points, quad4_shape_derivatives
from ..kinematics.continuum.linear import LinearContinuumKinematics


class Quad4(ContinuumElement):
    """4-node bilinear quadrilateral element (2D).

    Node ordering (CCW):
        3 --- 2
        |     |
        0 --- 1

    :param tag: Element ID
    :param nodes: list of 4 node tags (resolved to Node objects by Domain)
    :param material: nDMaterial template (PlaneStress or PlaneStrain)
    :param kinematics: ContinuumKinematics object (default: Linear)
    :param thickness: element thickness (default: 1.0)
    """

    def __init__(self, tag, nodes, material, kinematics=None, thickness=1.0):
        if kinematics is None:
            kinematics = LinearContinuumKinematics()
        super().__init__(tag, nodes, material, kinematics, thickness)

    def _getGaussPoints(self):
        return quad4_gauss_points()

    def _getShapeDerivatives(self, xi, eta):
        return quad4_shape_derivatives(xi, eta)

    def __repr__(self):
        return "Quad4(ID={})".format(self._ID)
