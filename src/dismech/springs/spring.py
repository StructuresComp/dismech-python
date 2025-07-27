import abc
import numpy as np
from typing import Any


class Springs(metaclass=abc.ABCMeta):
    """
    Abstract base class for contiguous spring blocks with shared memory.
    Viewable via .get_view(index), not list-style access.
    """

    def __init__(self, N: int, C = None):
        self.C = C
        self.N = N
        self._data = {}
        self._fields = []
        self._initialize_fields()

    def get_view(self, i: int):
        """Return a single spring view at index `i`."""
        return self._view_class(self, i)

    def __len__(self):
        return self.N

    def __iter__(self):
        for i in range(self.N):
            yield self.get_view(i)

    def __repr__(self):
        return f"<{self.__class__.__name__} with {self.N} springs: {self._fields}>"

    @property
    def _view_class(self):
        """Return the view class associated with this spring type"""
        return SpringView

    def __getattr__(self, name):
        if name in self._fields:
            return self._data[name]
        raise AttributeError(
            f"{type(self).__name__} has no attribute '{name}'")

    def __setattr__(self, name, value):
        if '_fields' in self.__dict__ and name in self._fields:
            expected = self._data[name]
            casted = np.asarray(value, dtype=expected.dtype)
            if casted.shape != expected.shape:
                raise ValueError(f"Shape mismatch for field '{name}': expected {expected.shape}, got {casted.shape}")
            self._data[name][...] = casted
        else:
            super().__setattr__(name, value)

    def __dir__(self):
        return super().__dir__() + self._fields

    @abc.abstractmethod
    def _initialize_fields(self, **kwargs):
        """Subclasses must initialize self._data and self._fields here."""
        self._fields += ['nat_strain', 'inc_strain']
        if self.C is not None:
            self._data['nat_strain'] = np.full((self.N, self.C), np.nan, dtype=np.float64)
            self._data['inc_strain'] = np.zeros((self.N, self.C), dtype=np.float64)
        else:
            self._data['nat_strain'] = np.full(self.N, np.nan, dtype=np.float64)
            self._data['inc_strain'] = np.zeros(self.N, dtype=np.float64)


class SpringView:
    """
    A 'view' of a spring within a Springs object.
    """

    def __init__(self, parent: Springs, index: int):
        if not (0 <= index < parent.N):
            raise ValueError(f"Index {index} is invalid for {type(parent)} of size {parent.N}")
        self._parent = parent
        self._index = index

    def __getattr__(self, name: str):
        if name in self._parent._fields:
            return self._parent._data[name][self._index]
        raise AttributeError(f"'SpringView' has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any):
        if name.startswith('_'):
            super().__setattr__(name, value)
        elif name in self._parent._fields:
            self._parent._data[name][self._index] = value
        else:
            raise AttributeError(f"'SpringView' has no attribute '{name}'")
