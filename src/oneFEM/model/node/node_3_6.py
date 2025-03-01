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
    def __init__(self, node_id, coord=[], mass=[], fix=[]):
        """
        Node Constructor
        :param node_id: Node ID (integer)
        :param coords:  3D coordinates as a list [X, Y, Z]
        :param mass:    Masses as a list [mass1, mass2, mass3, mass4, mass5, mass6]
        :param fix:     Fixity constraints as a list [fix1, fix2, fix3, fix4, fix5, fix6]
        """
        super().__init__(node_id)
        self.__nD = 3
        self.__nDOF = 3
        self.__dofs = Vector([0, 0, 0, 0, 0, 0], dtype=int)
        self.__coord = Vector([0.0, 0.0, 0.0], dtype=float)
        self.__mass = Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.__fix = Vector([False, False, False, False, False, False], dtype=bool)
              
        # Initialize solution-related variables
        self.__u_trial = Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.__v_trial = Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.__a_trial = Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.__u_commit = Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.__v_commit = Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.__a_commit = Vector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

        if len(coord) == self.__nD:
            self.__coord = Vector(coord, dtype=float)
        else:
            raise ValueError("oneFEM.Node36() - Number of coordinate entries does not match node dimension!")

        if len(mass) == self.__nDOF:
            self.__mass = Vector(mass, dtype=float)
        else:
            raise ValueError("oneFEM.Node36() - Number of mass entries does not match node number of d.o.f.s!")

        if len(fix) == self.__nDOF:
            self.__fix = Vector(fix, dtype=bool)
        else:
            raise ValueError("oneFEM.Node36() - Number of fix entries does not match node number of d.o.f.s!")

    # Node API (for internal use)
    def _setDOF(self, dofs):
        """
        Set global dofs that maps to the local dofs
        :param dofs: global degrees of freedom. list of integers e.g.: [10, 11, 12]
        """
        if len(dofs) == self.__nDOF:
            self.__dofs = Vector(dofs, dtype=int)
        else:
            raise ValueError("oneFEM.Node36._setDOF() - Number of DOF entries does not match node number of d.o.f.s!")

    def _update(self, disp, vel=None, accel=None):
        """
        Update trial displacements
        :param displacements: Displacements to commit
        :param velocities: (Optional) Velocities to commit
        :param accelerations: (Optional) Accelerations to commit
        """
        self.__u_trial = Vector(disp, dtype=float)
        self.__v_trial = Vector(vel, dtype=float)
        self.__a_trial = Vector(accel, dtype=float)
    
    def _commit(self, disp, vel=None, accel=None):
        """
        Commit results to the node
        :param displacements: Displacements to commit
        :param velocities: (Optional) Velocities to commit
        :param accelerations: (Optional) Accelerations to commit
        """
        self.__u_trial = Vector(disp, dtype=float)
        self.__v_trial = Vector(vel, dtype=float)
        self.__a_trial = Vector(accel, dtype=float)

    def _getTrialDisp(self):
        return self.__u_trial
    
    def _getTrialVel(self):
        return self.__v_trial
    
    def _getTrialAccel(self):
        return self.__a_trial
    
    def _getCommitDisp(self):
        return self.__u_commit
    
    def _getCommitVel(self):
        return self.__v_commit
    
    def _getCommitAccel(self):
        return self.__a_commit
    
    # Node API (public)
    def setFix(self, fix):
        """
        Set displacement boundary conditions
        :param fix: Fixities. list of bools e.g.: [True, False, False]
        """
        if len(fix) == self.__nDOF:
            self.__fix = Vector(fix, dtype=bool)
        else:
            raise ValueError("oneFEM.Node36.setFix() - Number of fix entries does not match node number of d.o.f.s!")

    def setMass(self, mass):
        """
        Set d.o.f. masses
        :param mass: Masses. list of floats e.g.: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        """
        if len(mass) == self.__nDOF:    
            self.__mass = Vector(mass, dtype=float)
        else:
            raise ValueError("oneFEM.Node36.setMass() - Number of mass entries does not match node number of d.o.f.s!")
    
    def getResult(self, query, dofs):
        """
        Retrieve results for specific DOFs
        :param query: String specifying the type of result ('disp', 'vel', or 'accel')
        :param dofs: List of DOFs to query e.g. [1, 2, 3, 4, 5, 6]
        :return: List of results for the specified DOFs
        """
        if query in {"disp", "displacement", "d", "u"}:
            return Vector([self.__u_commit[d-1] for d in dofs])
        elif query in {"vel", "velocity", "v"}:
            return Vector([self.__v_commit[d-1] for d in dofs])
        elif query in {"accel", "acceleration", "a"}:
            return Vector([self.__a_commit[d-1] for d in dofs])
        else:
            raise ValueError("oneFEM.Node.getResult() - Unknown query type!")

    def getCoordinates(self):
        return self.__coord