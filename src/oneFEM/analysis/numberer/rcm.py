##-----------------------------------------------------------------------##
#                                                                         #
#        #--oneFEM--#: One FEM software in a galaxy far far away          #
#                                                                         #
#                   Computational Mechanics 2022                          #
#                       University of Pavia                               #
#               Written by: Onur Deniz AKAN, IUSS Pavia                   #
#                                                                         #
##-----------------------------------------------------------------------##
# Reverse Cuthill-McKee DOF numberer
#   Minimizes matrix bandwidth by reordering DOFs using the RCM algorithm.
#   Free DOFs are numbered first (0..n_free-1), then constrained DOFs
#   (n_free..n_total-1), so K_uu is always the top-left block.

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from .main import Numberer


class RCM(Numberer):
    def __init__(self, nID=-1):
        super().__init__(nID)
        self._elem_dof_map = {}
        self._node_dof_map = {}
        self._uu = []
        self._pp = []

    def number(self, model):
        """Assign DOF numbers using Reverse Cuthill-McKee ordering.

        Free DOFs are numbered 0..n_free-1, constrained DOFs n_free..n_total-1.
        This places K_uu as the contiguous top-left block of the global matrix.

        Must be called before Domain._domain() resolves node tags to objects.
        Element node lists may contain integer tags or Node objects.
        """
        nodes = model.nodes
        elements = model.elements
        n_nodes = len(nodes)

        # Map node IDs to contiguous indices
        node_id_to_idx = {}
        for i, nd in enumerate(nodes):
            node_id_to_idx[nd._ID] = i

        # Build symmetric node adjacency graph from element connectivity
        # Element._nodes may hold int tags (before _domain) or Node objects
        rows = []
        cols = []
        for elem in elements:
            enodes = elem.getNodes()
            idxs = []
            for nd in enodes:
                if isinstance(nd, int):
                    idxs.append(node_id_to_idx[nd])
                else:
                    idxs.append(node_id_to_idx[nd._ID])
            for a in idxs:
                for b in idxs:
                    if a != b:
                        rows.append(a)
                        cols.append(b)

        if len(rows) > 0:
            data = np.ones(len(rows), dtype=np.int8)
            adj = csr_matrix((data, (np.array(rows, dtype=np.int32),
                                     np.array(cols, dtype=np.int32))),
                             shape=(n_nodes, n_nodes))
            perm = reverse_cuthill_mckee(adj)
        else:
            perm = np.arange(n_nodes)

        # Count total free DOFs to know where constrained numbering starts
        n_free = 0
        for node_idx in perm:
            nd = nodes[node_idx]
            fix = nd._fix
            for j in range(nd.getNDOF()):
                if not fix[j]:
                    n_free += 1

        # Now assign: free DOFs get 0..n_free-1, constrained get n_free..n_total-1
        free_counter = 0
        fixed_counter = n_free
        self._uu = []
        self._pp = []

        for node_idx in perm:
            nd = nodes[node_idx]
            nDOF = nd.getNDOF()
            fix = nd._fix
            dofs = [0] * nDOF
            for j in range(nDOF):
                if not fix[j]:
                    dofs[j] = free_counter
                    self._uu.append(free_counter)
                    free_counter += 1
                else:
                    dofs[j] = fixed_counter
                    self._pp.append(fixed_counter)
                    fixed_counter += 1
            nd._setDOF(dofs)

        # Cache DOF maps for fast assembly
        self._node_dof_map = {}
        for nd in nodes:
            dof_data = nd.getDOFs()
            if hasattr(dof_data, 'tolist'):
                self._node_dof_map[nd._ID] = np.array(dof_data.tolist(),
                                                       dtype=np.int32)
            else:
                self._node_dof_map[nd._ID] = np.array(list(dof_data),
                                                       dtype=np.int32)

        self._elem_dof_map = {}
        for elem in elements:
            edofs = []
            for nd in elem.getNodes():
                nid = nd if isinstance(nd, int) else nd._ID
                edofs.extend(self._node_dof_map[nid].tolist())
            self._elem_dof_map[elem._ID] = np.array(edofs, dtype=np.int32)

        return fixed_counter  # total DOF count
