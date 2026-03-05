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
# NODE RECORDER
#   Records nodal results (displacement, velocity, acceleration, reaction)
#   at each analysis step for one or more nodes.
#
#   Usage:
#       # Single node (backward compatible):
#       rec = NodeRecorder(1, nd3, dofs=[1,2], results=['displacement'])
#
#       # Multiple nodes:
#       rec = NodeRecorder(1, [nd1, nd2, nd3], dofs=[1,2], results=['displacement'])
#
#   Data storage:
#       Single node:  rec.data['displacement'] = [Vector, Vector, ...]
#       Multi-node:   rec.data['displacement'] = [{nodeID: Vector, ...}, ...]
#
#   File output:
#       rec = NodeRecorder(1, nd3, dofs=[1], results=['displacement'],
#                          file='node3_disp.out')
#       # writes one row per step: time val1 val2 ...
#

from .main import Recorder
from ...model.node.main import Node

# Query normalization: map user-facing names to internal _getResult keys
_QUERY_MAP = {
    'displacement': 'displacement', 'disp': 'displacement',
    'd': 'displacement', 'u': 'displacement',
    'reaction': 'force', 'force': 'force', 'forces': 'force', 'f': 'force',
    'velocity': 'velocity', 'vel': 'velocity', 'v': 'velocity',
    'acceleration': 'acceleration', 'accel': 'acceleration', 'a': 'acceleration',
}


class NodeRecorder(Recorder):
    def __init__(self, recID, nd=None, dofs=None, results=None, file=None):
        """
        Parameters
        ----------
        recID : int
            Recorder ID.
        nd : Node or list of Nodes
            Node(s) to record. Single Node for backward compatibility,
            or a list for multi-node recording.
        dofs : list of int
            DOF indices to record (1-based, e.g. [1,2,3]).
        results : list of str
            Result types: 'displacement', 'velocity', 'acceleration',
            'reaction' (or short aliases 'disp', 'vel', 'accel', 'force', etc.).
        file : str, optional
            File path for incremental output. One row per step,
            space-separated: time val1 val2 ...
        """
        super().__init__(recID)

        # Handle single node or list of nodes
        if nd is None:
            self._nodes = []
            self._single = True
        elif isinstance(nd, list):
            self._nodes = nd
            self._single = False
        else:
            self._nodes = [nd]
            self._single = True

        self._dofs = dofs if dofs is not None else []
        self._results = results if results is not None else []
        self._file = file

        # Initialize data storage
        for r in self._results:
            self.data[r] = []

        # Truncate output file on creation
        if self._file:
            with open(self._file, 'w') as f:
                pass

    def record(self, time=None):
        """Record current committed state for all tracked nodes/results.

        Parameters
        ----------
        time : float, optional
            Current simulation time (stored in self.time).
        """
        super().record(time)

        for r in self._results:
            query = _QUERY_MAP.get(r)
            if query is None:
                continue

            if self._single:
                # Single-node: backward compatible — store Vector directly
                vals = self._nodes[0]._getResult(query, self._dofs)
                self.data[r].append(vals)
            else:
                # Multi-node: store dict keyed by nodeID
                step = {}
                for nd in self._nodes:
                    step[nd._ID] = nd._getResult(query, self._dofs)
                self.data[r].append(step)

        # Incremental file output
        if self._file:
            self._write_step(time)

    def _write_step(self, time):
        """Append current step to output file."""
        parts = []
        if time is not None:
            parts.append(f"{time:.10e}")

        for r in self._results:
            last = self.data[r][-1]
            if isinstance(last, dict):
                # Multi-node: write in nodeID order
                for nid in sorted(last.keys()):
                    vals = last[nid]
                    if hasattr(vals, '__iter__'):
                        parts.extend(f"{float(v):.10e}" for v in vals)
                    else:
                        parts.append(f"{float(vals):.10e}")
            else:
                if hasattr(last, '__iter__'):
                    parts.extend(f"{float(v):.10e}" for v in last)
                else:
                    parts.append(f"{float(last):.10e}")

        with open(self._file, 'a') as f:
            f.write(" ".join(parts) + "\n")

    def save(self, filepath=None):
        """Write all recorded data to file at once.

        Parameters
        ----------
        filepath : str, optional
            Output file path. Defaults to self._file.
        """
        path = filepath or self._file
        if path is None:
            raise ValueError("NodeRecorder.save() - No file path specified.")

        with open(path, 'w') as f:
            nSteps = len(self.time) if self.time else 0
            if nSteps == 0:
                # Fall back to length of first result
                for r in self._results:
                    if self.data[r]:
                        nSteps = len(self.data[r])
                        break

            for step in range(nSteps):
                parts = []
                if self.time and step < len(self.time):
                    parts.append(f"{self.time[step]:.10e}")

                for r in self._results:
                    if step >= len(self.data[r]):
                        continue
                    entry = self.data[r][step]
                    if isinstance(entry, dict):
                        for nid in sorted(entry.keys()):
                            vals = entry[nid]
                            if hasattr(vals, '__iter__'):
                                parts.extend(f"{float(v):.10e}" for v in vals)
                            else:
                                parts.append(f"{float(vals):.10e}")
                    else:
                        if hasattr(entry, '__iter__'):
                            parts.extend(f"{float(v):.10e}" for v in entry)
                        else:
                            parts.append(f"{float(entry):.10e}")

                f.write(" ".join(parts) + "\n")

    def clear(self):
        """Reset all recorded data and truncate output file."""
        super().clear()
        if self._file:
            with open(self._file, 'w') as f:
                pass
