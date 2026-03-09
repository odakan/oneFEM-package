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
# Date: 09/03/2026
# Version: 0.2
#
# TotalLagrangianContinuumKinematics (v2 architecture — element API path)
#   Reference = always initial config. No state to commit/revert.
#   Strain: Green-Lagrange E = 0.5*(F^T F - I)
#   Stress: 2nd Piola-Kirchhoff S (from material via C:E)

from .nonlinear_base import _NonlinearContinuumBase
from ....._systools.data import Vector, Matrix
from ....._systools.backend import np


class TotalLagrangianContinuumKinematics(_NonlinearContinuumBase):
    """Total Lagrangian kinematics for continuum elements.

    Reference configuration is always the initial (undeformed) configuration.
    No state needs committing or reverting.

    v2: When initialized with element= kwarg, uses element API for geometry
    access (get_H, get_B_NL, get_dN_dX). Stores zero element geometry directly.
    """

    formulation = 'totalLagrangian'

    def initialize(self, nGP, nDim, nNodes, dN_dX_list, **kwargs):
        super().initialize(nGP, nDim, nNodes, dN_dX_list, **kwargs)
        self._element = kwargs.get('element', None)
        if self._element is not None:
            gauss_points = self._element.get_gauss_points()
            self._gp_coords = [gp_tuple[:-1] for gp_tuple in gauss_points]

    def update(self, gp, u_e):
        """Compute F, E, B_NL at this GP."""
        if self._element is not None:
            # v2 path: use element API
            xi = self._gp_coords[gp]
            u_data = u_e.data if hasattr(u_e, 'data') else np.asarray(u_e)
            H_np = self._element.get_H(xi, u_data)
            H = Matrix(init=H_np)
            self._F[gp] = self._computeF(H)
            self._strain[gp] = self._EtoVoigtCOV(self._computeGreenLagrange(H))
            self._B_NL[gp] = self._element.get_B_NL(xi, u_data)
        else:
            # Legacy path
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
        if self._element is not None:
            xi = self._gp_coords[gp]
            dN_dX = Matrix(init=self._element.get_dN_dX(xi))
            return self._buildKgeo(gp, stress, dN_dX)
        return self._buildKgeo(gp, stress)

    def getK(self, element):
        """Assemble element tangent stiffness K = K_mat + K_geo.
        K_mat = sum_gp B_NL^T C B_NL dV, K_geo = sum_gp K_sigma dV."""
        nDOF = element.get_nDOF_total()
        t = element._thickness
        K_mat = Matrix(shape=[nDOF, nDOF])
        K_geo = Matrix(shape=[nDOF, nDOF])
        for gp in range(self._nGP):
            B = self.getBMatrix(gp)
            C_mat = element.get_tangent(gp).to_matrix()
            detJ, w = element._gp_data[gp]
            dV = detJ * w * t
            K_mat += B.T @ C_mat @ B * dV
            Kg = self.getGeometricStiffness(gp, element.get_stress(gp))
            K_geo += Kg * dV
        return K_mat + K_geo

    def get_f_int(self, element):
        """Assemble element internal force f = sum_gp B_NL^T sigma dV."""
        nDOF = element.get_nDOF_total()
        t = element._thickness
        f = Vector(shape=nDOF)
        for gp in range(self._nGP):
            B = self.getBMatrix(gp)
            sig_vec = element.get_stress(gp).to_vector()
            detJ, w = element._gp_data[gp]
            dV = detJ * w * t
            f += B.T @ sig_vec * dV
        return f

    def copy(self):
        """Return a fresh (uninitialized) copy."""
        return TotalLagrangianContinuumKinematics()
