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
# Hex8 — 8-node trilinear hexahedral element (3D) with optional B-bar
#   Nodes: 8 x Node33 (3 translational DOFs each), total 24 DOFs
#   Integration: 2x2x2 Gauss (full)
#   Material: nDMaterial (type='3D')
#   Kinematics: injected, defaults to LinearContinuumKinematics(bbar=True)

from .base import ContinuumElement
from .isoparametric import hex8_gauss_points, hex8_shape_derivatives, hex8_shape_functions
from ..kinematics.continuum.linear import LinearContinuumKinematics
from ...._systools.backend import np
from ...._systools.data import Vector, Matrix


class Hex8(ContinuumElement):
    """8-node trilinear hexahedral element (3D) with optional B-bar.

    Node ordering (right-hand rule, CCW bottom then CCW top):
        7---6
       /|  /|
      4---5 |       z
      | 3-|-2       |  y
      |/  |/        | /
      0---1          x

    :param tag: Element ID
    :param nodes: list of 8 Node33 instances
    :param material: nDMaterial template (type='3D')
    :param kinematics: ContinuumKinematics (default: LinearContinuumKinematics(bbar=True))
    :param rho: mass density (default: 0.0)
    :param body_force: [bx, by, bz] array (default: None)
    """

    def __init__(self, tag, nodes, material, kinematics=None, rho=0.0, body_force=None):
        if kinematics is None:
            kinematics = LinearContinuumKinematics(bbar=True)
        # thickness=1.0 is a true no-op multiplier in ContinuumElement._domain()
        # (dV = detJ * w * t, so t=1.0 has no effect for 3D elements)
        super().__init__(tag, nodes, material, kinematics, thickness=1.0)
        self._rho = rho
        self._body_force = body_force

    def _getGaussPoints(self):
        return hex8_gauss_points()

    def _getShapeDerivatives(self, xi, eta, zeta):
        return hex8_shape_derivatives(xi, eta, zeta)

    def _domain(self):
        super()._domain()
        if self._rho > 0:
            self._buildMass()

    def _buildMass(self):
        """Consistent mass matrix: M = sum_gp rho * N_mat^T @ N_mat * detJ * w"""
        nDOF = self._nDOF_total
        M = np.zeros((nDOF, nDOF))
        gauss_pts = self._getGaussPoints()

        for gp_idx, gp_tuple in enumerate(gauss_pts):
            *coords, w = gp_tuple
            N = hex8_shape_functions(*coords)
            detJ, _ = self._gp_data[gp_idx]

            N_mat = np.zeros((3, nDOF))
            for a in range(8):
                N_mat[0, 3*a]     = N[a]
                N_mat[1, 3*a + 1] = N[a]
                N_mat[2, 3*a + 2] = N[a]

            M += self._rho * (N_mat.T @ N_mat) * detJ * w

        self._m = Matrix(init=M)

    def __repr__(self):
        return "Hex8(ID={})".format(self._ID)
