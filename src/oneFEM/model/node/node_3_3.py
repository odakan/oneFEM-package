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
# NODE 3D 3-dof specialization
#   3D node with 3 degrees of freedom: ux, uy, uz

from .main import Node
from ..._systools.data import Vector

class Node33(Node):
    def __init__(self, node_id, coord=None, mass=None, fix=None):
        """
        Node Constructor
        :param node_id: Node ID (integer)
        :param coord:   3D coordinates as a list [X, Y, Z]
        :param mass:    Masses as a list [mass1, mass2, mass3]
        :param fix:     Fixity constraints as a list [fix1, fix2, fix3]
        """
        if coord is None:
            coord = []
        if mass is None:
            mass = []
        if fix is None:
            fix = []
        super().__init__(node_id)
        self._nD = 3
        self._nDOF = 3

        # private
        self._dofs = Vector([0, 0, 0], dtype=int)
        self._coord = Vector([0.0, 0.0, 0.0], dtype=float)
        self._mass = Vector([0.0, 0.0, 0.0], dtype=float)
        self._fix = Vector([False, False, False], dtype=bool)

        # Initialize solution-related variables
        self._f_trial = Vector([0.0, 0.0, 0.0], dtype=float)
        self._u_trial = Vector([0.0, 0.0, 0.0], dtype=float)
        self._v_trial = Vector([0.0, 0.0, 0.0], dtype=float)
        self._a_trial = Vector([0.0, 0.0, 0.0], dtype=float)
        self._f_commit = Vector([0.0, 0.0, 0.0], dtype=float)
        self._u_commit = Vector([0.0, 0.0, 0.0], dtype=float)
        self._v_commit = Vector([0.0, 0.0, 0.0], dtype=float)
        self._a_commit = Vector([0.0, 0.0, 0.0], dtype=float)

        if len(coord) == self._nD:
            self._coord = Vector(coord, dtype=float)
        elif len(coord) == 0:
            pass
        else:
            raise ValueError("oneFEM.Node33() - Number of coordinate entries does not match node dimension!")

        if len(mass) == self._nDOF:
            self._mass = Vector(mass, dtype=float)
        elif len(mass) == 0:
            pass
        else:
            raise ValueError("oneFEM.Node33() - Number of mass entries does not match node number of d.o.f.s!")

        if len(fix) == self._nDOF:
            self._fix = Vector(fix, dtype=bool)
        elif len(fix) == 0:
            pass
        else:
            raise ValueError("oneFEM.Node33() - Number of fix entries does not match node number of d.o.f.s!")

    def __repr__(self):
        return f"Node33(ID={self._ID}, Coord={list(self._coord)}, Fix={list(self._fix)})"

    # Node API
    def _setDOF(self, dofs):
        if len(dofs) == self._nDOF:
            self._dofs = Vector(dofs, dtype=int)
        else:
            raise ValueError("oneFEM.Node33._setDOF() - Number of DOF entries does not match node number of d.o.f.s!")

    def _update(self, force, disp, vel=None, accel=None):
        self._f_trial = force
        self._u_trial = disp
        if vel is not None:
            self._v_trial = vel
        if accel is not None:
            self._a_trial = accel
        return 0

    def _commitState(self):
        self._f_commit = Vector(self._f_trial)
        self._u_commit = Vector(self._u_trial)
        self._v_commit = Vector(self._v_trial)
        self._a_commit = Vector(self._a_trial)
        return 0

    def _revertToLastCommit(self):
        self._f_trial = Vector(self._f_commit)
        self._u_trial = Vector(self._u_commit)
        self._v_trial = Vector(self._v_commit)
        self._a_trial = Vector(self._a_commit)
        return 0

    def _revertToStart(self):
        self._dofs = Vector([0, 0, 0], dtype=int)
        self._coord = Vector([0.0, 0.0, 0.0], dtype=float)
        self._mass = Vector([0.0, 0.0, 0.0], dtype=float)
        self._fix = Vector([False, False, False], dtype=bool)
        self._f_trial = Vector([0.0, 0.0, 0.0], dtype=float)
        self._u_trial = Vector([0.0, 0.0, 0.0], dtype=float)
        self._v_trial = Vector([0.0, 0.0, 0.0], dtype=float)
        self._a_trial = Vector([0.0, 0.0, 0.0], dtype=float)
        self._f_commit = Vector([0.0, 0.0, 0.0], dtype=float)
        self._u_commit = Vector([0.0, 0.0, 0.0], dtype=float)
        self._v_commit = Vector([0.0, 0.0, 0.0], dtype=float)
        self._a_commit = Vector([0.0, 0.0, 0.0], dtype=float)
        return 0

    def _getTrialDisp(self):
        return self._u_trial

    def _getTrialVel(self):
        return self._v_trial

    def _getTrialAccel(self):
        return self._a_trial

    def _getCommitDisp(self):
        return self._u_commit

    def _getCommitVel(self):
        return self._v_commit

    def _getCommitAccel(self):
        return self._a_commit

    def _setFix(self, fix):
        if len(fix) == self._nDOF:
            self._fix = Vector(fix, dtype=bool)
        else:
            raise ValueError("oneFEM.Node33._setFix() - Number of fix entries does not match node number of d.o.f.s!")

    def _setMass(self, mass):
        if len(mass) == self._nDOF:
            self._mass = Vector(mass, dtype=float)
        else:
            raise ValueError("oneFEM.Node33._setMass() - Number of mass entries does not match node number of d.o.f.s!")

    def _getResult(self, query, dofs):
        if query in {"force", "forces", "f"}:
            return Vector([self._f_commit[d-1] for d in dofs])
        elif query in {"disp", "displacement", "displacements", "d", "u"}:
            return Vector([self._u_commit[d-1] for d in dofs])
        elif query in {"vel", "velocity", "v"}:
            return Vector([self._v_commit[d-1] for d in dofs])
        elif query in {"accel", "acceleration", "a"}:
            return Vector([self._a_commit[d-1] for d in dofs])
        else:
            raise ValueError("oneFEM.Node33._getResult() - Unknown result type!")

    def _getCoordinates(self):
        return self._coord

    def _getDOFIndices(self):
        return self._dofs

    def _getFixity(self):
        return self._fix
