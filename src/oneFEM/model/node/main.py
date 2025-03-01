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
#   definition of a FE node object and functions operating over a node
#   object. The node object is defined with the following attributes:
#   mass, fixity, and coordinate information

from ..._systools.data.vector import Vector

class Node(object):
    def __init__(self, node_id=-1):
        """
        Node Constructor
        :param node_id: Node ID (integer)
        """
        self.__ID = node_id
        self.__nD = 0
        self.__nDOF = 0
        self.__dofs = Vector(dtype=int)
        self.__coord = Vector()
        self.__mass = Vector()
        self.__fix = Vector(dtype=bool)
              
        # Initialize solution-related variables
        self.__u_trial = Vector()
        self.__v_trial = Vector()
        self.__a_trial = Vector()
        self.__u_commit = Vector()
        self.__v_commit = Vector()
        self.__a_commit = Vector()

    # Node API (for internal use)
    def _setDOF(self, dofs):
        """
        Set global dofs that maps to the local dofs
        :param dofs: global degrees of freedom. list of integers e.g.: [10, 11, 12]
        """
        if len(dofs) == self.__nDOF:
            self.__dofs = Vector(dofs, dtype=int)
        else:
            raise ValueError("oneFEM.Node._setDOF() - Number of DOF entries does not match node number of d.o.f.s!")

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
        self.__u_commit = Vector(disp, dtype=float)
        self.__v_commit = Vector(vel, dtype=float)
        self.__a_commit = Vector(accel, dtype=float)

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
            raise ValueError("oneFEM.Node.setFix() - Number of fix entries does not match node number of d.o.f.s!")

    def setMass(self, mass):
        """
        Set d.o.f. masses
        :param mass: Masses. list of floats e.g.: [1.0, 2.0, 3.0]
        """
        if len(mass) == self.__nDOF:    
            self.__mass = Vector(mass, dtype=float)
        else:
            raise ValueError("oneFEM.Node.setMass() - Number of mass entries does not match node number of d.o.f.s!")
    
    def getResult(self, query, dofs):
        """
        Retrieve results for specific DOFs
        :param query: String specifying the type of result ('disp', 'vel', or 'accel')
        :param dofs: List of DOFs to query e.g. [1, 2, 3]
        :return: List of results for the specified DOFs
        """
        if query in {"disp", "displacement", "d", "u"}:
            return [self.__u_commit[d-1] for d in dofs]
        elif query in {"vel", "velocity", "v"}:
            return [self.__v_commit[d-1] for d in dofs]
        elif query in {"accel", "acceleration", "a"}:
            return [self.__a_commit[d-1] for d in dofs]
        else:
            raise ValueError("oneFEM.Node.getResult() - Unknown query type!")

    def getCoordinates(self):
        """Return a list of coordinates"""
        return self.__coord.data
    
    def getDOFs(self):
        """Return a list of nodes"""
        return self.__dofs.data