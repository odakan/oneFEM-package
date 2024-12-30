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
    def __init__(self, ID, path=None):
        """
        Initialize the SimulationManager object.
        :param path: Directory of the folder with .txt files.
        """
        # Initialize properties
        self.ID = ID
        self.model_path = None
        self.model = Domain()  # Initialize the domain
        self.analysis = Analysis()  # Initialize the analysis

        # Check if path exists
        if Path(path).is_dir():
            self.model_path = path
        else:
            raise FileNotFoundError("oneFEM:SimulationManager: Model directory does not exist!\nExit code: -1")

        # Build the model
        self._build()

        # Display message
        print("oneFEM:SimulationManager: Model built successfully.")

    def _build(self):

        # first wipe the model
        self.wipe()

        """Read the model definition."""
        self.__read("nodes.txt")
        self.__read("materials.txt")
        self.__read("elements.txt")
        self.__read("time_series.txt")
        self.__read("recorders.txt")
        self.__read("analysis.txt")

        # Compute the reference configuration
        self.model._domain()

    def run(self):
        """Run analysis and record results."""
        for analysis in self.analysis:
            self.model = analysis._analyze(self.model)
        print("oneFEM:SimulationManager: Analysis completed successfully.")
        self.terminate()

    def __read(self, file):
        """
        Read a .txt file and process its contents.

        :param file: Name of the .txt file to read.
        """
        # Change directory temporarily
        original_dir = os.getcwd()
        os.chdir(self.model_path)

        try:
            # Read the file line by line
            with open(file, "r") as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith("#"):
                        continue

                    # Split the line into tokens
                    tokens = line.split()
                    self._add(tokens)
        except FileNotFoundError:
            print(f"oneFEM:SimulationManager: File {file} not found!")
        finally:
            # Return to the original directory
            os.chdir(original_dir)

    def _add(self, cmd):
        """
        Dispatch add object command to the domain or analysis.

        :param cmd: List of tokens representing the command.
        """
        if cmd:
            if cmd[0] == "analysis":
                an = Analysis()
                if self.analysis[-1].ID == -1:  # If the latest element is invalid
                    self.analysis[-1] = an.add_analysis(cmd)
                else:  # If the latest analysis is valid
                    self.analysis.append(an.add_analysis(cmd))
            else:  # Other commands work on the Domain
                self.model = self.model.add(cmd)
        else:
            print("oneFEM:SimulationManager: Command string is empty!")

    def _terminate(self, loud=True):
        """
        Clean up and display termination message if required.

        :param loud: Whether to display the termination message.
        """
        if loud:
            print("oneFEM:SimulationManager: Cleaning up...")


    def wipe(self):
        pass