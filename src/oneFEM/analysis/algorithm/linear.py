##-----------------------------------------------------------------------##
#                                                                         #
#      #--oneFEM--#: One 2D FEM software in a galaxy far far away         #
#                                                                         #
#                   Computational Mechanics 2022                          #
#                       University of Pavia                               #
#               Written by: Onur Deniz AKAN, IUSS Pavia                   #
#                         27 January 2022                                 #
#                                                                         #
##-----------------------------------------------------------------------##
#LINEAR solver sub-object definition
#   Linear static solver

import numpy as np
from .main import Algorithm

class Linear(Algorithm):
    def __init__(self, ID=-1, nSteps=0, dt=0.0):
        # Call the parent class constructor
        super().__init__(ID, None, None, nSteps, dt)
        self.ID = ID
        self.nSteps = nSteps
        self.dt = dt

    @staticmethod
    def add_analysis(data):
        # Validate input
        if len(data) < 5:
            raise ValueError("Linear: Analysis input must be of 3 numbers and strings!")
        
        if not isinstance(data[3], (int, float)) or not isinstance(data[4], (int, float)):
            raise ValueError("Linear: Analysis input must be numeric!")

        # Validate the ID
        if not isinstance(data[0], int):
            raise ValueError("Linear: Analysis ID must be an integer!")

        # Create and return the Linear analysis object
        return Linear(ID=data[0], nSteps=data[3], dt=data[4])

    def analyze(self, model):
        # Perform initial assembly
        model.assemble()
        
        # Organize degrees of freedom (DOFs)
        self.organize(model)
        
        # Solve for each step
        for _ in range(self.nSteps):
            model = self.solve(model)

        return model

    def solve(self, model):
        # Compute displacement increment and reaction increment
        # Displacement increment
        uu = self.uu
        pp = self.pp
        K = model.K
        F = model.F
        u = model.u
        
        # Solve for displacements at free DOFs
        u[uu] = np.linalg.solve(
            K[np.ix_(uu, uu)],
            F[uu] - K[np.ix_(uu, pp)].dot(u[pp])
        )
        
        # Compute reactions at fixed DOFs
        F[pp] = (
            K[np.ix_(pp, pp)].dot(u[pp]) +
            K[np.ix_(pp, uu)].dot(u[uu])
        )
        
        # Update the model
        model.u = u
        model.F = F
        
        return model
