import typing

import numpy as np

from .spring import Springs


class TripletSprings(Springs):
    """
    Base class for springs on rod triplets (n0, e, n1, f, n2) whose energy is a
    nonlinear function of several strains (see TripletEnergy).

    Subclasses set n_strain (number of strain components) and declare their
    material fields in _material_fields.

    Common fields per spring:
        nodes_ind, edges_ind, sgn, ind: triplet topology, as in BendSprings
        ref_len:     rest lengths of edges e and f
        stretch_w:   length of edge e/f whose pure stretch this spring carries
                     (edge length / number of springs of this type sharing the edge)
        voronoi_len: 0.5 * (ref_len_e + ref_len_f)
        triplet_ind: index of the triplet in the robot's bend/twist springs
                     (used to look up the reference twist)
    """

    n_strain: int = 1

    def __init__(self, N: int, dof_map_fn):
        self._dof_map_fn = dof_map_fn
        super().__init__(N, self.n_strain)

    def _material_fields(self) -> typing.Dict[str, np.ndarray]:
        """Subclass-specific per-spring fields as {name: empty array}."""
        return {}

    def _initialize_fields(self):
        material = self._material_fields()
        self._fields = ['nodes_ind', 'edges_ind', 'sgn', 'ref_len', 'stretch_w',
                        'voronoi_len', 'triplet_ind', 'ind'] + list(material)
        self._data = {
            'nodes_ind': np.empty((self.N, 3), dtype=np.int32),
            'edges_ind': np.empty((self.N, 2), dtype=np.int32),
            'sgn': np.empty((self.N, 2), dtype=np.int32),
            'ref_len': np.empty((self.N, 2), dtype=np.float64),
            'stretch_w': np.empty((self.N, 2), dtype=np.float64),
            'voronoi_len': np.empty(self.N, dtype=np.float64),
            'triplet_ind': np.empty(self.N, dtype=np.int32),
            'ind': np.empty((self.N, 11), dtype=np.int32),
            **material,
        }
        super()._initialize_fields()

    @classmethod
    def from_arrays(cls,
                    nodes_edge_ind: np.ndarray,
                    sgn: np.ndarray,
                    triplet_ind: np.ndarray,
                    ref_len_arr: np.ndarray,
                    dof_map_fn,
                    edge_map_fn,
                    **material):
        """Build springs on triplets [n0, e, n1, f, n2]; material fields by keyword."""
        N = len(nodes_edge_ind)

        instance = cls(N=N, dof_map_fn=dof_map_fn)

        if N != 0:
            instance.nodes_ind = np.stack(
                (nodes_edge_ind[:, 0], nodes_edge_ind[:, 2], nodes_edge_ind[:, 4]), axis=1)
            instance.edges_ind = np.stack(
                (nodes_edge_ind[:, 1], nodes_edge_ind[:, 3]), axis=1)
            instance.sgn = sgn
            instance.triplet_ind = triplet_ind
            for name, value in material.items():
                setattr(instance, name, value)

            inds0 = np.vstack([dof_map_fn(n0) for n0 in nodes_edge_ind[:, 0]])
            inds1 = np.vstack([dof_map_fn(n1) for n1 in nodes_edge_ind[:, 2]])
            inds2 = np.vstack([dof_map_fn(n2) for n2 in nodes_edge_ind[:, 4]])
            inds3 = np.vstack([edge_map_fn(e1) for e1 in nodes_edge_ind[:, 1]])
            inds4 = np.vstack([edge_map_fn(e2) for e2 in nodes_edge_ind[:, 3]])
            instance.ind = np.hstack([inds0, inds1, inds2, inds3, inds4])

            ref_len = ref_len_arr[instance.edges_ind]
            instance.ref_len = ref_len
            instance.voronoi_len = 0.5 * ref_len.sum(axis=1)

            # Each edge's pure stretch is split evenly among the springs sharing it
            edges, counts = np.unique(instance.edges_ind, return_counts=True)
            n_share = counts[np.searchsorted(edges, instance.edges_ind)]
            instance.stretch_w = ref_len / n_share

        return instance
