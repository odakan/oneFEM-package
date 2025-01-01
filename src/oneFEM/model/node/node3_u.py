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
#NODE main object definition
#   definition of a FE node object and functions operating over a node
#   3D 3-dof mechanical solid node

from numpy import array
from .main import Node

class Node3U(Node):
    def __init__(self, node_id, coord=[], mass=[], fix=[]):
        """
        Node Constructor
        :param node_id: Node ID (integer)
        :param coords:  2D or 3D coordinates as a list [X, Y, Z]
        :param mass:    Masses as a list [mass1, mass2, mass3]
        :param fix:     Fixity constraints as a list [fix1, fix2, fix3]
        """
        super().__init__(node_id)
        self.__nD = int(3)
        self.__nDOF = int(3)
        self.__dofs = array([0, 0, 0], dtype=int)
        self.__coord = array([0.0, 0.0, 0.0], dtype=float)
        self.__mass = array([0.0, 0.0, 0.0], dtype=float)
        self.__fix = array([0.0, 0.0, 0.0], dtype=bool)
              
        # Initialize solution-related variables
        self.__u_trial = array([0.0, 0.0, 0.0], dtype=float)
        self.__v_trial = array([0.0, 0.0, 0.0], dtype=float)
        self.__a_trial = array([0.0, 0.0, 0.0], dtype=float)
        self.__u_commit = array([0.0, 0.0, 0.0], dtype=float)
        self.__v_commit = array([0.0, 0.0, 0.0], dtype=float)
        self.__a_commit = array([0.0, 0.0, 0.0], dtype=float)

        if len(coord) == self.__nD:
            self.__coord = array(coord, dtype=float)
        else:
            raise ValueError("oneFEM.Node() - Number of coordinate entries does not match node dimension!")

        if len(mass) == self.__nDOF:
            self.__mass = array(mass, dtype=float)
        else:
            raise ValueError("oneFEM.Node() - Number of mass entries does not match node number of d.o.f.s!")

        if len(fix) == self.__nDOF:
            self.__fix = array(fix, dtype=float)
        else:
            raise ValueError("oneFEM.Node() - Number of mass entries does not match node number of d.o.f.s!")

    # Node API (for internal use)
    def _setDOF(self, dofs):
        """
        Set global dofs that maps to the local dofs
        :param dofs: global degrees of freedom. list of integers e.g.: [10, 11, 12]
        """
        if len(dofs) == self.__nDOF:
            self.__mass = array(dofs, dtype=float)
        else:
            raise ValueError("oneFEM.Node._setDOF() - Number of mass entries does not match node number of d.o.f.s!")

    def _update(self, disp, vel=None, accel=None):
        """
        Update trial displacements
        :param displacements: Displacements to commit
        :param velocities: (Optional) Velocities to commit
        :param accelerations: (Optional) Accelerations to commit
        """
        self.__u_trial = array(disp, dtype=float)
        self.__v_trial = array(vel, dtype=float)
        self.__a_trial = array(accel, dtype=float)
    
    def _commit(self, disp, vel=None, accel=None):
        """
        Commit results to the node
        :param displacements: Displacements to commit
        :param velocities: (Optional) Velocities to commit
        :param accelerations: (Optional) Accelerations to commit
        """
        self.__u_trial = array(disp, dtype=float)
        self.__v_trial = array(vel, dtype=float)
        self.__a_trial = array(accel, dtype=float)

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
            self.__fix = array(fix, dtype=float)
        else:
            raise ValueError("oneFEM.Node.setFix() - Number of mass entries does not match node number of d.o.f.s!")

    def setMass(self, mass):
        """
        Set d.o.f. masses
        :param mass: Masses. list of bools e.g.: [True, False, False]
        """
        if len(mass) == self.__nDOF:    
            self.__mass = array(mass, dtype=float)
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
            return array([self.__u_commit[d-1] for d in dofs])
        elif query in {"vel", "velocity", "v"}:
            return array([self.__v_commit[d-1] for d in dofs])
        elif query in {"accel", "acceleration", "a"}:
            return array([self.__a_commit[d-1] for d in dofs])
        else:
            raise ValueError("oneFEM.Node.getResult() - Unknown query type!")

    def getCoordinates(self):
        return self.__coord