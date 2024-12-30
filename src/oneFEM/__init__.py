# oneFEM/__init__.py

"""
oneFEM: A Python Package for Finite Element Modeling and Analysis
        Copyright (C) 2024  FE Implementation Library

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Modules:
    - analysis: Tools for running finite element analyses.
    - model: Classes and functions for building finite element models.
    - input: Functions for importing mesh from software such as CUBIT.
    - output: Functions for visualizing and exporting results.

Direct usage:
    >>> from oneFEM import source
    >>> source(path_to_model_file)

Interactive usage:
    >>> # pushover of a cantilever column
    >>> from oneFEM import SimulationManager
    >>> from oneFEM.model import Domain, Node
    >>> from oneFEM.model.element import BernoulliBeam
    >>> from oneFEM.model.constraint import Fix
    >>> from oneFEM.analysis import Analysis
    >>> from oneFEM.output import NodeRecorder, ElementRecorder
    >>> column = Domain()   # initialize an empty domain
    >>> column.builder(nD=3, nDOF=6)    # set domain properties
    >>> column.add(Node(nID_1, [xCrd, yCrd, zCrd]))
    >>> column.add(Node(nID_2, [xCrd, yCrd, zCrd]))
    >>> column.add(Node(nID_3, [xCrd, yCrd, zCrd]))
    >>> column.add(BernoulliBeam(eID_1, [nID_1, nID_2], E=1e9))
    >>> column.add(Fix(ID, [nID_1, nID_3], [1, 1, 1, 1, 1, 1]))
    >>> column.add.load(nID_1, '')
    >>> column.add.load('')
    >>> column.add.pattern(ID, horizontal_load)
    >>> column.add.tseries(ID, 'linear')
    >>> column.remove(nID_3)    # remove a node inc. load and and any constraints
    >>> base_shear = NodeRecorder(ID, nodes=[nID_1], outfile='filename', results=['reaction'], dofs=[1, 2, 3])
    >>> cord_rotation = ElementRecorder(ID, nodes=[nID_1], outfile='filename', results=['reaction'], dofs=[1, 2, 3])
    >>> outputs = [base_shear, cord_rotation]
    >>> static_analysis = Analysis([['Integrator', args], ['System', args], 
                                    ['Algorithm', args], ['Numberer'], 
                                    ['Constraints', args], ['Test', tol=1e-5, nInt=10]])
    >>> pushover = SimulationManager(column, static_analysis, outputs)
    >>> pushover.analyze(dt=0.005, nSteps=1000)
"""

from . import analysis      # Import analysis module
from . import model         # Import model module
from . import output        # Import output module
from . import input         # Import input module

# Import top-level objects
from ._systools.simulation_manager import SimulationManager # Simulation Manager tool
from ._systools import source                               # The source command
from ._systools import data                                 # The data module

# Optional: Define a version number or metadata for your package
__version__ = "0.0.1"
__author__ = "FE Implementation Library"

__all__ = [
    "analysis",
    "model",
    "output",
    "input",
    "data",
    "SimulationManager",
    "source"
]

metadata = {
    "version": "0.0.1",
    "author": "FE Implementation Library",
    "email": "fe.implement.library@gmail.com",
    "license": "GPLv3",
    "description": "A Python package for finite element modeling and analysis",
    "copyright": "Copyright (c) 2024 FE Implementation Library",
    "url": "https://feimplementlib.github.io/",
    "dependencies": ['numpy', 'scipy', 'matplotlib', 'ipython'],
    "keywords": ["finite element", "modeling", "analysis", "simulation"],
    "python_version": ">=3.6, <4.0",
}