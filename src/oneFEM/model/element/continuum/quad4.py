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

import warnings
from .base import ContinuumElement
from ...kinematics.continuum.cauchy.linear import LinearContinuumKinematics
from ...._systools.backend import np


def _quad4_shape_functions(xi, eta):
    """4-node bilinear quad shape functions.

    Node ordering (CCW):
        3 --- 2
        |     |
        0 --- 1

    Natural coordinates: xi, eta in [-1, 1].

    :param xi:  natural coordinate xi
    :param eta: natural coordinate eta
    :return: N array of shape (4,)
    """
    return 0.25 * np.array([
        (1.0 - xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 - eta),
        (1.0 + xi) * (1.0 + eta),
        (1.0 - xi) * (1.0 + eta),
    ])


def _quad4_shape_derivatives(xi, eta):
    """Natural derivatives dN/d(xi, eta) for 4-node bilinear quad.

    :param xi:  natural coordinate xi
    :param eta: natural coordinate eta
    :return: dN_dxi array of shape (4, 2) — rows are nodes, cols are [dN/dxi, dN/deta]
    """
    return 0.25 * np.array([
        [-(1.0 - eta), -(1.0 - xi)],
        [ (1.0 - eta), -(1.0 + xi)],
        [ (1.0 + eta),  (1.0 + xi)],
        [-(1.0 + eta),  (1.0 - xi)],
    ])


def _quad4_gauss_points():
    """2x2 Gauss quadrature points and weights for bilinear quad.

    :return: list of (xi, eta, weight) tuples, 4 points
    """
    g = 1.0 / np.sqrt(3.0)
    return [
        (-g, -g, 1.0),
        ( g, -g, 1.0),
        ( g,  g, 1.0),
        (-g,  g, 1.0),
    ]


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
        # PhysicsFamily compatibility checks
        pf = self.physics_family
        if hasattr(kinematics, 'physics_family') and kinematics.physics_family != pf:
            warnings.warn(
                "Quad4({}): kinematics physics_family={} != element physics_family={}".format(
                    tag, kinematics.physics_family.value, pf.value), stacklevel=2)
        if hasattr(material, 'physics_family') and material.physics_family != pf:
            warnings.warn(
                "Quad4({}): material physics_family={} != element physics_family={}".format(
                    tag, material.physics_family.value, pf.value), stacklevel=2)
        super().__init__(tag, nodes, material, kinematics, thickness)

    def _getGaussPoints(self):
        return _quad4_gauss_points()

    def _getShapeDerivatives(self, xi, eta):
        return _quad4_shape_derivatives(xi, eta)

    # v2 element API hooks
    def _get_shape_functions(self, xi):
        return _quad4_shape_functions(*xi)

    def _get_shape_derivatives(self, xi):
        return _quad4_shape_derivatives(*xi)

    def __repr__(self):
        return "Quad4(ID={})".format(self._ID)
