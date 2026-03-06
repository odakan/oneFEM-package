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
# Date: 08/02/2025
# Version: 0.1
#
#NODE main object definition
#   definition of a FE node object and functions operating on a node
#   3D 6-dof mechanical structure node

from .main import Node
from ..._systools.data import Vector

class Node36(Node):
    def __init__(self, node_id, coord=None, mass=None, fix=None):
        """
        Node Constructor
        :param node_id: Node ID (integer)
        :param coords:  3D coordinates as a list [X, Y, Z]
        :param mass:    Masses as a list [mass1, mass2, mass3, mass4, mass5, mass6]
        :param fix:     Fixity constraints as a list [fix1, fix2, fix3, fix4, fix5, fix6]
        """
        if coord is None:
            coord = []
        if mass is None:
            mass = []
        if fix is None:
            fix = []
        super().__init__(node_id)
        self._nD = 3
        self._nDOF = 6

        # private
        self._dofs = Vector([0, 0, 0, 0, 0, 0], dtype=int)
        self._coord = Vector([0.0, 0.0, 0.0], dtype=float)
        self._mass = Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self._fix = Vector([False, False, False, False, False, False], dtype=bool)
              
        # Initialize solution-related variables
        self._f_trial = Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self._u_trial = Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self._v_trial = Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self._a_trial = Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self._f_commit = Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self._u_commit = Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self._v_commit = Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self._a_commit = Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

        if len(coord) == self._nD:
            self._coord = Vector(coord, dtype=float)
        elif len(coord) == 0:
            pass
        else:
            raise ValueError("oneFEM.Node36() - Number of coordinate entries does not match node dimension!")

        if len(mass) == self._nDOF:
            self._mass = Vector(mass, dtype=float)
        elif len(mass) == 0:
            pass
        else:
            raise ValueError("oneFEM.Node36() - Number of mass entries does not match node number of d.o.f.s!")

        if len(fix) == self._nDOF:
            self._fix = Vector(fix, dtype=bool)
        elif len(fix) == 0:
            pass
        else:
            raise ValueError("oneFEM.Node36() - Number of fix entries does not match node number of d.o.f.s!")
        
    def __repr__(self):
        return f"Node36(ID={self._ID}, Coord={list(self._coord)}, Fix={list(self._fix)})"


    # Node API
    def _setDOF(self, dofs):
        """
        Set global dofs that maps to the local dofs
        :param dofs: global degrees of freedom. list of integers e.g.: [10, 11, 12]
        """
        if len(dofs) == self._nDOF:
            self._dofs = Vector(dofs, dtype=int)
        else:
            raise ValueError("oneFEM.Node36._setDOF() - Number of DOF entries does not match node number of d.o.f.s!")

    def _update(self, force, disp, vel=None, accel=None):
        """
        Update trial displacements
        :param displacements: Displacements to commit
        :param velocities: (Optional) Velocities to commit
        :param accelerations: (Optional) Accelerations to commit
        """
        self._f_trial = force
        self._u_trial = disp
        if vel is not None:
            self._v_trial = vel
        if accel is not None:
            self._a_trial = accel

        return 0
    
    def _commitState(self):
        """
        Commit results to the node (deep copy to avoid aliasing)
        """
        self._f_commit = Vector(self._f_trial)
        self._u_commit = Vector(self._u_trial)
        self._v_commit = Vector(self._v_trial)
        self._a_commit = Vector(self._a_trial)
        return 0

    def _revertToLastCommit(self):
        """
        Revert to the last commit (deep copy to avoid aliasing)
        """
        self._f_trial = Vector(self._f_commit)
        self._u_trial = Vector(self._u_commit)
        self._v_trial = Vector(self._v_commit)
        self._a_trial = Vector(self._a_commit)
        return 0

    def _revertToStart(self):
        self._dofs = Vector([0, 0, 0, 0, 0, 0], dtype=int)
        self._coord = Vector([0.0, 0.0, 0.0], dtype=float)
        self._mass = Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self._fix = Vector([False, False, False, False, False, False], dtype=bool)
        self._f_trial = Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self._u_trial = Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self._v_trial = Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self._a_trial = Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self._f_commit = Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self._u_commit = Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self._v_commit = Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self._a_commit = Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
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
        """
        Set displacement boundary conditions
        :param fix: Fixities. list of bools e.g.: [True, False, False]
        """
        if len(fix) == self._nDOF:
            self._fix = Vector(fix, dtype=bool)
        else:
            raise ValueError("oneFEM.Node36.setFix() - Number of fix entries does not match node number of d.o.f.s!")

    def _setMass(self, mass):
        """
        Set d.o.f. masses
        :param mass: Masses. list of floats e.g.: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        """
        if len(mass) == self._nDOF:    
            self._mass = Vector(mass, dtype=float)
        else:
            raise ValueError("oneFEM.Node36.setMass() - Number of mass entries does not match node number of d.o.f.s!")
    
    def _getResult(self, query, dofs):
        """
        Retrieve results for specific DOFs
        :param query: String specifying the type of result ('disp', 'vel', or 'accel')
        :param dofs: List of DOFs to query e.g. [1, 2, 3, 4, 5, 6]
        :return: List of results for the specified DOFs
        """
        if query in {"force", "forces", "f"}:
            return Vector([self._f_commit[d-1] for d in dofs])
        elif query in {"disp", "displacement", "displacements", "d", "u"}:
            return Vector([self._u_commit[d-1] for d in dofs])
        elif query in {"vel", "velocity", "v"}:
            return Vector([self._v_commit[d-1] for d in dofs])
        elif query in {"accel", "acceleration", "a"}:
            return Vector([self._a_commit[d-1] for d in dofs])
        else:
            raise ValueError("oneFEM.Node.getResult() - Unknown result type!")

    def _getCoordinates(self):
        return self._coord
    
    def _getDOFIndices(self):
        """Returns global DOF indices for this node."""
        return self._dofs

    def _getFixity(self):
        """Returns fixity (bool list) for each DOF."""
        return self._fix
