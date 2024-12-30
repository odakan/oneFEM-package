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

class Node(object):
    def ONE_Node(self, cmd):
        # create and return the node object with the specifications in the command 
        #check length of command
        length = len(cmd)
        if not length == 2:
            raise ValueError("NODE: Node input must be a list of length 2!\n Exit code: -1")
            #raise an error


    def __init__(self, node_id, coords):
        """
        Node Constructor
        :param node_id: Node ID (integer)
        :param n_dof: Number of Degrees of Freedom (2 or 3)
        :param coords: 2D coordinates as a list [X, Y]
        :param fix: Fixity constraints as a list [fix1, fix2, ...]
        :param loads: Loads as a list [l1, l2, ...]
        """
        if not isinstance(node_id, int) or isinstance(node_id, bool):
            raise ValueError("Node ID must be an integer!")
        
        if n_dof not in (2, 3):
            raise ValueError("Number of DOF can be either 2 or 3!")
        
        if len(coords) != 2:
            raise ValueError("Node coordinates must be in 2D!")
        
        if len(fix) != n_dof:
            raise ValueError("Number of DOF and fix constraints do not match!")
        
        if len(loads) != n_dof:
            raise ValueError("Number of DOF and loads do not match!")
        
        self.ID = node_id
        self.nDOF = n_dof
        self.coords = coords
        self.fix = [int(f) for f in fix]
        self.force = [0.0] * n_dof
        self.imposed = [0.0] * n_dof
        
        for i in range(n_dof):
            if self.fix[i] == 0:  # Free DOF
                self.force[i] = loads[i]
            elif self.fix[i] == 1:  # Fixed DOF
                self.imposed[i] = loads[i]
            else:
                raise ValueError("Fix constraints must be 0 (free) or 1 (fixed)!")
        
        # Initialize solution-related variables
        self.u_trial = [0.0] * n_dof
        self.u_commit = [0.0] * n_dof
        self.v_commit = [0.0] * n_dof
        self.a_commit = [0.0] * n_dof
    
    def update(self, displacements):
        """
        Update trial displacements
        :param displacements: List of trial displacements
        """
        if len(displacements) != self.nDOF:
            raise ValueError("Displacement length does not match DOF!")
        self.u_trial = displacements
    
    def commit(self, displacements, velocities=None, accelerations=None):
        """
        Commit results to the node
        :param displacements: Displacements to commit
        :param velocities: (Optional) Velocities to commit
        :param accelerations: (Optional) Accelerations to commit
        """
        if len(displacements) != self.nDOF:
            raise ValueError("Displacement length does not match DOF!")
        self.u_commit = displacements
        if velocities:
            self.v_commit = velocities
        if accelerations:
            self.a_commit = accelerations
    
    def result(self, query, dofs):
        """
        Retrieve results for specific DOFs
        :param query: String specifying the type of result ('disp', 'vel', or 'acc')
        :param dofs: List of DOFs to query
        :return: List of results for the specified DOFs
        """
        if query in {"disp", "displacement", "d", "u"}:
            return [self.u_commit[d] for d in dofs]
        elif query in {"vel", "velocity", "v"}:
            return [self.v_commit[d] for d in dofs]
        elif query in {"acc", "acceleration", "a"}:
            return [self.a_commit[d] for d in dofs]
        else:
            raise ValueError("Unknown query type!")
