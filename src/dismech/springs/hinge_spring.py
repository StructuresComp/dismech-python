import typing
import numpy as np

from .spring import Springs


class HingeSprings(Springs):
    def __init__(self, N: int, dof_map_fn):
        self._dof_map_fn = dof_map_fn
        super().__init__(N)

    def _initialize_fields(self):
        self._fields = ['nodes_ind', 'kb', 'ind']
        self._data = {
            'nodes_ind': np.empty((self.N, 4), dtype=np.int32),
            'kb': np.empty(self.N, dtype=np.float64),
            'ind': np.empty((self.N, 12), dtype=np.int32),
        }
        super()._initialize_fields()

    @classmethod
    def from_arrays(cls, nodes_ind: np.ndarray, kb: np.ndarray, dof_map_fn):
        N = kb.shape[0]

        assert nodes_ind.shape == (N, 4)

        instance = cls(N=N, dof_map_fn=dof_map_fn)
        if N != 0:
            instance.nodes_ind = nodes_ind
            instance.kb = kb

            # Easier to handle dimension expansion separately
            inds0 = np.vstack([dof_map_fn(n0) for n0 in nodes_ind[:, 0]])
            inds1 = np.vstack([dof_map_fn(n1) for n1 in nodes_ind[:, 1]])
            inds2 = np.vstack([dof_map_fn(n1) for n1 in nodes_ind[:, 2]])
            inds3 = np.vstack([dof_map_fn(n1) for n1 in nodes_ind[:, 3]])
            instance.ind = np.hstack([inds0, inds1, inds2, inds3])

        return instance
