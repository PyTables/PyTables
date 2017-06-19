import sys
import numpy
from .leaf import Leaf
from tables import Atom
from tables.utils import byteorders


class Array(Leaf):
    def __init__(self, _atom=None, **kwargs):
        super().__init__(**kwargs)
        self.atom = _atom
        if _atom is None or _atom.shape == ():
            self.atom = Atom.from_dtype(self.dtype)
        self.nrow = None
        # Provisional for test
        self.flavor = 'numpy'
        # TODO iterators?

    @property
    def rowsize(self):
        "The size of the rows in bytes in dimensions orthogonal to *maindim*."
        maindim = self.maindim
        rowsize = self.atom.size
        for i, dim in enumerate(self.shape):
            if i != maindim:
                rowsize *= dim
        return rowsize

    @property
    def size_in_memory(self):
        """The size of this array's data in bytes when it is fully loaded into
        memory."""
        return self.nrows * self.rowsize

    def _g_create(self):
        """Save a new array in file."""

    def get_enum(self):
        if self.atom.kind != 'enum':
            raise TypeError("array ``%s`` is not of an enumerated type"
                            % self._v_pathname)

        return self.atom.enum

    def read(self, start=None, stop=None, step=None):
        (start, stop, step) = self._process_range_read(start, stop, step)
        if self.nrows == 1:
            return numpy.array(self[()])
        else:
            return self[start:stop:step]

    def iterrows(self, start=None, stop=None, step=None):
        start, stop, step = self._process_range(start, stop, step,
                                                warn_negstep=False)
        # Scalar dataset
        if self.nrows == 1:
            self.nrow = 0
            yield numpy.array(self[()])
        else:
            self.nrow = start
            for r in self[start:stop:step]:
                self.nrow += step
                yield r

    __iter__ = iterrows
