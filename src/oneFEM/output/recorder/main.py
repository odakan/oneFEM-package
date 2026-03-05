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
#
# RECORDER base class
#   Collects and stores simulation results at each analysis step.
#   Subclasses: NodeRecorder, ElementRecorder, ModeShapeRecorder.
#


class Recorder(object):
    def __init__(self, ID=-1):
        self._ID = ID
        self.time = []
        self.data = {}

    def record(self, time=None):
        """Record current state. Called by Domain._record() each step.

        Parameters
        ----------
        time : float, optional
            Current simulation time.
        """
        if time is not None:
            self.time.append(time)

    def clear(self):
        """Reset all recorded data."""
        self.time = []
        for key in self.data:
            self.data[key] = []

    def save(self, filepath=None):
        """Save recorded data to file. Subclasses override with format."""
        pass
