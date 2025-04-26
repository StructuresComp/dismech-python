import typing
import numpy as np

from .spring import Springs


class TwistSprings(Springs):

    def __init__(self, N: int, dof_map_fn):
        self._dof_map_fn = dof_map_fn
        super().__init__(N)

    def _initialize_fields(self):
        self._fields = ['nodes_ind', 'edges_ind', 'GJ', 'sgn', 'voronoi_len', 'ind']
        self._data = {
            'nodes_ind': np.empty((self.N, 3), dtype=np.int32),
            'edges_ind': np.empty((self.N, 2), dtype=np.int32),
            'EI': np.empty((self.N, 2), dtype=np.float64),
            'GJ': np.empty(self.N, dtype=np.float64),
            'sgn': np.empty((self.N, 2), np.int32),
            'voronoi_len': np.empty(self.N, dtype=np.float64),
            'ind': np.empty((self.N, 11), dtype=np.int32),
        }
        super()._initialize_fields()

    @classmethod
    def from_arrays(cls,
                    nodes_edge_ind: np.ndarray,
                    sgn: np.ndarray,
                    GJ: np.ndarray,
                    ref_len_arr: np.ndarray,
                    dof_map_fn,
                    edge_map_fn):
        N = len(nodes_edge_ind)

        instance = cls(N=N, dof_map_fn=dof_map_fn)

        if N != 0:
            instance.nodes_ind = np.stack(
                (nodes_edge_ind[:, 0], nodes_edge_ind[:, 2], nodes_edge_ind[:, 4]), axis=1)
            instance.edges_ind = np.stack(
                (nodes_edge_ind[:, 1], nodes_edge_ind[:, 3]), axis=1)
            instance.GJ = GJ
            instance.sgn = sgn

            # Easier to handle dimension expansion separately
            inds0 = np.vstack([dof_map_fn(n0) for n0 in nodes_edge_ind[:, 0]])
            inds1 = np.vstack([dof_map_fn(n1) for n1 in nodes_edge_ind[:, 2]])
            inds2 = np.vstack([dof_map_fn(n2) for n2 in nodes_edge_ind[:, 4]])
            inds3 = np.vstack([edge_map_fn(e1) for e1 in nodes_edge_ind[:, 1]])
            inds4 = np.vstack([edge_map_fn(e2) for e2 in nodes_edge_ind[:, 3]])
            instance.ind = np.hstack([inds0, inds1, inds2, inds3, inds4])

            instance.voronoi_len = 0.5 * (ref_len_arr[instance.edges_ind[:, 0]] +
                                      ref_len_arr[instance.edges_ind[:, 1]])

        return instance
