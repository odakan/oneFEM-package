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
source object definition
    wrapper for model running
    source(path_to_file) runs the model .py file and returns an interactive IPython shell.
    source is a void function, hence it does not return any value (None).
"""

import os
import sys
import importlib.util

class source(object):
    """
    Wrapper for running a model from a Python file.
    It executes the model script and then returns an interactive IPython shell.
    """

    def __init__(self, path_to_file, shell=False):
        """
        Initializes the source object with a path to a model file.
        :param path_to_file: Path to the .py model file
        :param shell:        Return IPython shell if True (optional)      
        """
        self.__path_to_file = path_to_file
        self.__shell_requested = shell
        self.__run_model()
        
    def __run_model(self, shell):
        """
        Executes the provided model file (.py) and enters interactive IPython shell.
        """

        # Ensure the file exists
        if not os.path.exists(self.__path_to_file):
            print(f"Error: File '{self.__path_to_file}' does not exist.")
            sys.exit(1)

        # Run interactive shell
        self.__start_shell()

        # Load the model script dynamically
        model_name = os.path.splitext(os.path.basename(self.__path_to_file))[0]
        spec = importlib.util.spec_from_file_location(model_name, self.__path_to_file)
        model_module = importlib.util.module_from_spec(spec)
        sys.modules[model_name] = model_module
        spec.loader.exec_module(model_module)

        print(f"Model '{model_name}' loaded successfully.")

    def __start_shell(self):
        if self.__shell_requested:
            # Start IPython interactive shell
            import IPython
            IPython.embed()