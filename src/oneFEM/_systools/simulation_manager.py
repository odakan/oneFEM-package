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

from ..model.main import Domain
from ..analysis.main import Analysis

# SimulationManager class definition
class SimulationManager(object):
    def __init__(self, ID, domain=Domain(), analysis=Analysis(), dt=0.0):
        self.ID = ID
        self._dt = dt
        self.model_path = None
        self.model = domain
        self.analysis = analysis

        print("oneFEM.SimulationManager: Model built successfully.")


    def analyze(self, nSteps=1, dt=-1):
        # if a new dt is specified
        if dt > 0.0:
            self._dt = dt

        # Run the analysis pipeline
        self.analysis._analyze(self.model, nSteps, self._dt)

        print("oneFEM.SimulationManager: Analysis completed successfully.")


    def terminate(self, loud=True):
        if loud:
            print("oneFEM.SimulationManager: Cleaning up...")


    def wipe(self):
        pass
