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

Quick start::

    from oneFEM.model import Domain
    from oneFEM.model.node import Node36
    from oneFEM.model.element.truss import Truss
    from oneFEM.model.material.uniaxial import Elastic
    from oneFEM.model.pattern import Plain
    from oneFEM.model.tseries import Constant
    from oneFEM.analysis import Analysis
    from oneFEM import SimulationManager
    from oneFEM.output.recorder import NodeRecorder

    model = Domain()
    model.add(Node36(1, coord=[0, 0, 0], fix=[True]*6))
    model.add(Node36(2, coord=[1, 0, 0]))

    mat = Elastic(1, E=200e9)
    model.add(Truss(1, [1, 2], mat, A=0.01))

    ts = Constant(1, factor=1.0)
    pat = Plain(1, ts, loads=[[2, 1000, 0, 0, 0, 0, 0]])
    model.add(pat)

    rec = NodeRecorder(1, nd=2, dofs=[1, 2, 3], results=['disp'])
    model.add(rec)

    analysis = Analysis(algorithm='Linear', system='FullGeneral',
                        integrator='LoadControl', numberer='Plain',
                        constraints='Plain')
    sim = SimulationManager(model, analysis)
    sim.analyze(nSteps=1)
"""

from . import analysis      # Import analysis module
from . import model         # Import model module
from . import output        # Import output module
from . import input         # Import input module

# Import top-level objects
from ._systools.simulation_manager import SimulationManager # Simulation Manager tool
from ._systools.source import source                        # The source command
from ._systools import data                                 # The data module

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
