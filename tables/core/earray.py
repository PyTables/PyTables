import numpy as np
from .carray import CArray
from ..utils import convert_to_np_atom2

class EArray(CArray):
    def __init__(self, expectedrows=None, **kwargs):
        super(EArray, self).__init__(**kwargs)
        self._v_expectedrows = expectedrows
        """The expected number of rows to be stored in the array."""
        if 'new' in kwargs and kwargs['new']:
            zerodims = np.sum(np.array(self.shape) == 0)
            if zerodims > 0:
                if zerodims == 1:
                    self.extdim = list(self.shape).index(0)
                else:
                    raise NotImplementedError(
                        "Multiple enlargeable (0-)dimensions are not "
                        "supported.")
            else:
                raise ValueError(
                    "When creating EArrays, you need to set one of "
                    "the dimensions of the Atom instance to zero.")
        else:
            self.extdim = list(self.backend.maxshape).index(None)

    def _check_shape_append(self, nparr):
        "Test that nparr shape is consistent with underlying EArray."

        # The arrays conforms self expandibility?
        myrank = len(self.shape)
        narank = len(nparr.shape) - len(self.atom.shape)
        if myrank != narank:
            raise ValueError(("the ranks of the appended object (%d) and the "
                              "``%s`` EArray (%d) differ")
                             % (narank, self._v_pathname, myrank))
        for i in range(myrank):
            if i != self.extdim and self.shape[i] != nparr.shape[i]:
                raise ValueError(("the shapes of the appended object and the "
                                  "``%s`` EArray differ in non-enlargeable "
                                  "dimension %d") % (self._v_pathname, i))

    def append(self, sequence):
        """Add a sequence of data to the end of the dataset.

        The sequence must have the same type as the array; otherwise a
        TypeError is raised. In the same way, the dimensions of the
        sequence must conform to the shape of the array, that is, all
        dimensions must match, with the exception of the enlargeable
        dimension, which can be of any length (even 0!).  If the shape
        of the sequence is invalid, a ValueError is raised.

        """

        self._g_check_open()
        self._v_file._check_writable()

        # Convert the sequence into a NumPy object
        nparr = convert_to_np_atom2(sequence, self.atom)
        # Check if it has a consistent shape with underlying EArray
        self._check_shape_append(nparr)
        # If the size of the nparr is zero, don't do anything else
        if nparr.size > 0:
            start = self.shape[self.extdim]
            stop = start + nparr.size
            self.backend.resize(stop, axis=self.extdim)
            self[tuple(slice(None) if i != self.extdim
                        else slice(start, stop) for i in range(len(self.shape)))]

