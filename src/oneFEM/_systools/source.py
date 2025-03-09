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
source function definition
    function for sourcing .py files
    source(path_to_file) runs the .py file and returns an interactive IPython shell.
    source is a void function, hence it does not return any value (None).
"""

import os
import sys
import importlib.util


def __start_shell():
    # Start IPython interactive shell
    import IPython
    IPython.embed()


def __run_file(__path_to_file, __shell_requested):
    """
    Executes the provided model file (.py) and enters interactive IPython shell.
    """

    # Ensure the file exists
    if not os.path.exists(__path_to_file):
        print(f"Error: File '{__path_to_file}' does not exist.")
        sys.exit(1)

    # Run interactive shell
    if __shell_requested:
        __start_shell()

    # Load the file script dynamically
    file_name = os.path.splitext(os.path.basename(__path_to_file))[0]
    spec = importlib.util.spec_from_file_location(file_name, __path_to_file)
    model_module = importlib.util.module_from_spec(spec)
    sys.modules[file_name] = model_module
    spec.loader.exec_module(model_module)

    print(f"File '{file_name}' loaded successfully.")


def source(path_to_file, shell=False):
    """
    Wrapper for running a model from a Python file.
    It executes the model script and then returns an interactive IPython shell.
    """
    __run_file(path_to_file, shell)