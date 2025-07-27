from .spring import Springs

import numpy as np


class StretchSprings(Springs):
    def __init__(self, N: int, dof_map_fn):
        self._dof_map_fn = dof_map_fn
        super().__init__(N)

    def _initialize_fields(self):
        self._fields = ['nodes_ind', 'ref_len', 'EA', 'ind']
        self._data = {
            'nodes_ind': np.empty((self.N, 2), dtype=np.int32),
            'ref_len': np.empty(self.N, dtype=np.float64),
            'EA': np.empty(self.N, dtype=np.float64),
            'ind': np.empty((self.N, 6), dtype=np.int32),
        }
        super()._initialize_fields()

    @classmethod
    def from_arrays(cls, nodes_ind: np.ndarray, ref_len: np.ndarray, EA: np.ndarray, dof_map_fn):
        N = len(ref_len)

        assert nodes_ind.shape == (N, 2)
        assert EA.shape == (N,)
        
        instance = cls(N=N, dof_map_fn=dof_map_fn)

        if N != 0:
            instance.nodes_ind = nodes_ind
            instance.ref_len = ref_len
            instance.EA = EA

            # Easier to handle dimension expansion separately
            inds0 = np.vstack([dof_map_fn(n0) for n0 in nodes_ind[:, 0]])
            inds1 = np.vstack([dof_map_fn(n1) for n1 in nodes_ind[:, 1]])
            instance.ind = np.hstack([inds0, inds1])

        return instance
