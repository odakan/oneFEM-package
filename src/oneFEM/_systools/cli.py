# oneFEM/cli.py

import os
import sys
import argparse


__opening_text = """
        oneFEM  Copyright (C) 2024  FE Implementation Library
    This program comes with ABSOLUTELY NO WARRANTY; for details type `--warranty'.
    This is free software, and you are welcome to redistribute it
    under certain conditions; type `--license' for details.
"""


def __show_warranty():
    """
    Display warranty information.
    """
    warranty_text = """
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
    """
    print(warranty_text)


def __show_license():
    """
    Display license information.
    """
    license_text = """
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    You should have received a copy of the GNU General Public License
    along with this program. If not, see <https://www.gnu.org/licenses/>.
    """
    print(license_text)


def __run_model(directory):
    """
    Simulate running a model from the given directory.
    """
    
    if not os.path.isdir(directory):
        raise RuntimeError(f"Error: '{directory}' is not a valid directory.", file=sys.stderr)

    from .source import source
    print(__opening_text)
    print(f"\nRunning model at: {directory}\n")
    source(directory, shell=True)


def _cli_main():
    """
    Command-line interface for the oneFEM package.
    """

    # Check if no arguments are passed to start IPython interactive session
    if len(sys.argv) == 1:
        # Ensure oneFEM is imported
        try:
            import oneFEM

        except ImportError:
            print("oneFEM is not installed. Please install it using 'pip install oneFEM'.")
            sys.exit(1)

        # Ensure oneFEM is imported
        try:
            import IPython
            IPython.embed()     # Start the interactive IPython shell
            
        except ImportError:
            print("IPython is not installed. Please install it using 'pip install ipython'.")
            sys.exit(1)
        
        return


    # If there are arguments, process them using argparse
    parser = argparse.ArgumentParser(
        description="Run finite element analysis using oneFEM.",
        epilog="Use --warranty or --license to display information about the program.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        prog="oneFEM"
    )
    
    parser.add_argument(
        "path_to_model",
        nargs="?",  # Optional argument (can be given or not)
        type=str,
        help="Path to the file containing the model data.",
    )
    
    parser.add_argument(
        "--warranty",
        action="store_true",
        help="Display warranty information."
    )

    parser.add_argument(
        "--license",
        action="store_true",
        help="Display license information."
    )

    # Process commands
    args = parser.parse_args()

    # Execute commands    
    if args.warranty:
        __show_warranty()
        return

    if args.license:
        __show_license()
        return

    # Load the model and perform analysis
    if args.path_to_model:
        try:
           __run_model(args.path_to_model)

        except Exception as e:
            raise RuntimeError(f"FATAL ERROR!: {e}")

    else:
        raise RuntimeError("Error: No directory provided. Use '--help' for usage instructions.", file=sys.stderr)

    # exit when all is done
    sys.exit(0)