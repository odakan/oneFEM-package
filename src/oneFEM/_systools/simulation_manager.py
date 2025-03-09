##-----------------------------------------------------------------------##
#                                                                         #
#      #--oneFEM--#: One FEM software in a galaxy far far away            #
#                                                                         #
#                   Computational Mechanics 2022                          #
#                       University of Pavia                               #
#               Written by: Onur Deniz AKAN, IUSS Pavia                   #
#                         18 January 2022                                 #
#                                                                         #
##-----------------------------------------------------------------------##

"""
 SimulationManager object definition
   wrapper for the FE model
   links the model domain with analysis and recorders

   one instance can handle:
       - analysis and output for a single domain 
       - domain decomposition for parallelism with single domain

   multiple instances can be created and linked for:
       - composite models with multiple domains
       - mixed-solution for a composite model 
       - multiphysics simulations
       - multiscale simulations
       - parallelism with multiple domains
"""

import os
from pathlib import Path
from ..model.main import Domain
from ..analysis.main import Analysis

# SimulationManager class definition
class SimulationManager(object):
    def __init__(self, ID, domain=Domain(), analysis=Analysis(), dt=0.0):
        """
        Initialize the SimulationManager object.
        :param path: Directory of the folder with .txt files.
        """
        # Initialize properties
        self.ID = ID
        self._dt = dt
        self.model_path = None
        self.model = domain  # Initialize the domain
        self.analysis = analysis  # Initialize the analysis

        # Display message
        print("oneFEM:SimulationManager: Model built successfully.")


    def analyze(self, nSteps, dt=-1):

        # if a new dt is specified
        if dt > 0.0:
            self._dt=dt

        """Run analysis and record results."""


        print("oneFEM:SimulationManager: Analysis completed successfully.")
        self.terminate()


    def terminate(self, loud=True):
        """
        Clean up and display termination message if required.

        :param loud: Whether to display the termination message.
        """
        if loud:
            print("oneFEM:SimulationManager: Cleaning up...")


    def wipe(self):
        pass