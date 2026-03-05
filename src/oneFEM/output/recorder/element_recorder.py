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
# ELEMENT RECORDER
#   Records element-level results (strain, stress, force, tangent)
#   at each analysis step for one or more elements.
#
#   Usage:
#       # Single element (backward compatible):
#       rec = ElementRecorder(1, tr1, results=['strain', 'stress'])
#
#       # Multiple elements:
#       rec = ElementRecorder(1, [tr1, tr2], results=['strain'])
#
#   Data storage:
#       Single element:  rec.data['strain'] = [val, val, ...]
#       Multi-element:   rec.data['strain'] = [{eleID: val, ...}, ...]
#
#   File output:
#       rec = ElementRecorder(1, tr1, results=['strain'],
#                             file='ele_strain.out')
#       # writes one row per step: time val1 val2 ...
#

from .main import Recorder
from ...model.element.main import Element

# Query normalization
_QUERY_MAP = {
    'strain': 'strain', 'strains': 'strain', 'eps': 'strain',
    'stress': 'stress', 'stresses': 'stress', 'sig': 'stress',
    'force': 'force', 'forces': 'force', 'f': 'force',
    'tangent': 'tangent', 'tang': 'tangent', 'stiffness': 'tangent',
}


class ElementRecorder(Recorder):
    def __init__(self, recID, ele=None, results=None, file=None):
        """
        Parameters
        ----------
        recID : int
            Recorder ID.
        ele : Element or list of Elements
            Element(s) to record. Single Element for backward compatibility,
            or a list for multi-element recording.
        results : list of str
            Result types: 'strain', 'stress', 'force', 'tangent'
            (or aliases 'eps', 'sig', 'f', 'tang', 'stiffness').
        file : str, optional
            File path for incremental output. One row per step,
            space-separated: time val1 val2 ...
        """
        super().__init__(recID)

        # Handle single element or list
        if ele is None:
            self._elements = []
            self._single = True
        elif isinstance(ele, list):
            self._elements = ele
            self._single = False
        else:
            self._elements = [ele]
            self._single = True

        self._results = results if results is not None else []
        self._file = file

        # Initialize data storage
        for r in self._results:
            self.data[r] = []

        # Truncate output file on creation
        if self._file:
            with open(self._file, 'w') as f:
                pass

    def _query_element(self, ele, query):
        """Extract a result value from a single element.

        Parameters
        ----------
        ele : Element
            The element to query.
        query : str
            Normalized query key ('strain', 'stress', 'force', 'tangent').

        Returns
        -------
        value : float, Vector, or Matrix
            The requested result.
        """
        if query == 'strain':
            if hasattr(ele, '_section') and hasattr(ele._section, 'getStrain'):
                return ele._section.getStrain()
            return None
        elif query == 'stress':
            if hasattr(ele, '_section') and hasattr(ele._section, 'getStress'):
                return ele._section.getStress()
            return None
        elif query == 'force':
            return ele.getForce()
        elif query == 'tangent':
            if hasattr(ele, '_section') and hasattr(ele._section, 'getTangent'):
                return ele._section.getTangent()
            return None
        return None

    def record(self, time=None):
        """Record current committed state for all tracked elements/results.

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
                # Single-element: backward compatible — store value directly
                val = self._query_element(self._elements[0], query)
                self.data[r].append(val)
            else:
                # Multi-element: dict keyed by element ID
                step = {}
                for ele in self._elements:
                    step[ele.getID()] = self._query_element(ele, query)
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
                for eid in sorted(last.keys()):
                    val = last[eid]
                    self._format_value(val, parts)
            else:
                self._format_value(last, parts)

        with open(self._file, 'a') as f:
            f.write(" ".join(parts) + "\n")

    @staticmethod
    def _format_value(val, parts):
        """Format a result value into string parts."""
        if val is None:
            parts.append("nan")
        elif hasattr(val, '__iter__'):
            parts.extend(f"{float(v):.10e}" for v in val)
        else:
            parts.append(f"{float(val):.10e}")

    def save(self, filepath=None):
        """Write all recorded data to file at once.

        Parameters
        ----------
        filepath : str, optional
            Output file path. Defaults to self._file.
        """
        path = filepath or self._file
        if path is None:
            raise ValueError("ElementRecorder.save() - No file path specified.")

        with open(path, 'w') as f:
            nSteps = len(self.time) if self.time else 0
            if nSteps == 0:
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
                        for eid in sorted(entry.keys()):
                            self._format_value(entry[eid], parts)
                    else:
                        self._format_value(entry, parts)

                f.write(" ".join(parts) + "\n")

    def clear(self):
        """Reset all recorded data and truncate output file."""
        super().clear()
        if self._file:
            with open(self._file, 'w') as f:
                pass
