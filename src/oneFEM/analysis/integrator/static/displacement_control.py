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
# DisplacementControl (DispControl)
#   Static integrator that controls displacement at one or more DOFs.
#   Each analysis step increments the controlled DOFs by a prescribed amount.
#
#   The controlled DOFs must be fixed (SP constraint) so they are in the pp
#   partition. The integrator directly sets the displacement at those DOFs
#   in newStep(). Newton-Raphson only solves for the remaining free DOFs.
#
#   Usage:
#     1. Fix the controlled DOF(s) with SP constraints (node fix parameter)
#     2. Create DispControl(node, dof, incr) for single DOF, or
#        DispControl(nodes=[(node1,dof1), (node2,dof2),...], incr=incr) for multiple
#     3. Use with Newton algorithm

from ..main import Integrator
from ...._systools.data import Vector
import numpy as np


class DispControl(Integrator):
    """
    Displacement control integrator for static nonlinear analysis.
    Controls displacement at one or more DOFs.
    """

    def __init__(self, node=None, dof=0, incr=0.0, nodes=None, iID=-1):
        """
        :param node: Node object whose DOF is controlled (single DOF mode)
        :param dof: Local DOF index (0-based) within the node (single DOF mode)
        :param incr: Displacement increment per analysis step
        :param nodes: List of (node, dof) tuples for multi-DOF control
        :param iID: Integrator ID
        """
        super().__init__(iID)
        self.__incr = incr
        self.__global_dofs = None  # list of global DOF indices

        # Build list of (node, local_dof) pairs
        if nodes is not None:
            self.__node_dof_pairs = list(nodes)
        elif node is not None:
            self.__node_dof_pairs = [(node, dof)]
        else:
            self.__node_dof_pairs = []

    def newStep(self, model, dt, current_time):
        """
        Increment the controlled DOFs and commit to nodes.
        """
        self._current_time = current_time
        self._dt = dt

        # Get committed displacements
        self._U = model.getCommittedDisp()

        # Resolve global DOF indices on first call
        if self.__global_dofs is None:
            self.__global_dofs = []
            for nd, local_dof in self.__node_dof_pairs:
                node_dofs = nd.getDOFs()
                if hasattr(node_dofs, 'tolist'):
                    dof_list = node_dofs.tolist()
                elif hasattr(node_dofs, 'data'):
                    dof_list = node_dofs.data.tolist()
                else:
                    dof_list = list(node_dofs)
                self.__global_dofs.append(dof_list[local_dof])

        # Apply displacement increment at all controlled DOFs
        for gdof in self.__global_dofs:
            self._U[gdof] += self.__incr

        # Commit to model so elements see the updated displacement
        model.u = Vector(list(self._U))
        model._commit(model.F, model.u)
        for element in model.elements:
            element._update()
