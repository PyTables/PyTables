import sys
import numpy as np
from .leaf import Leaf
from .. import Atom
from ..utils import byteorders
from ..exceptions import ClosedNodeError


class Array(Leaf):
    def __init__(self, _atom=None, **kwargs):
        super().__init__(**kwargs)
        self.atom = _atom
        if _atom is None or _atom.shape == ():
            if self.dtype == np.dtype('O'):
                self.atom = Atom.from_dtype(np.array(self[()]).dtype)
            else:
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

    def read(self, start=None, stop=None, step=None, out=None):

        if not self._v_file._isopen or self._isopen:
            raise ClosedNodeError
        # Scalar dataset
        if self.shape == ():
            arr = self[()]
            nrowstoread = 1
        else:
            (start, stop, step) = self._process_range_read(start, stop, step)
            arr = self[start:stop:step]
            nrowstoread = len(range(start, stop, step))

        if (isinstance(arr, np.ndarray) and byteorders[arr.dtype.byteorder] != sys.byteorder):
            arr = arr.byteswap(True)
            arr.dtype = arr.dtype.newbyteorder('=')

        if out is not None:
            if self.flavor != 'numpy':
                msg = ("Optional 'out' argument may only be supplied if array "
                       "flavor is 'numpy', currently is {0}").format(self.flavor)
                raise TypeError(msg)
            bytes_required = self.rowsize * nrowstoread
            # if buffer is too small, it will segfault
            if bytes_required != out.nbytes:
                raise ValueError(('output array size invalid, got {0} bytes, '
                                  'need {1} bytes').format(out.nbytes,
                                                           bytes_required))
            if not out.flags['C_CONTIGUOUS']:
                raise ValueError('output array not C contiguous')

            # Check empty dataset
            if self.nrows != 0:
                if self.shape == ():
                    self.backend.read_direct(out)
                else:
                    self.backend.read_direct(out, np.s_[start:stop:step], np.s_[0:nrowstoread])
            return out

        return arr

    def iterrows(self, start=None, stop=None, step=None):
        start, stop, step = self._process_range(start, stop, step,
                                                warn_negstep=False)
        # Scalar dataset
        if self.shape == ():
            self.nrow = 0
            yield np.array(self[()])
        else:
            self.nrow = start
            for r in self[start:stop:step]:
                self.nrow += step
                yield r

    __iter__ = iterrows

    def copy(self, newparent=None, newname=None,
             overwrite=False, createparents=True, *,
             copyuserattrs=True,
             **kwargs):

        if not hasattr(newparent, 'create_array'):
            newparent = self.root._get_or_create_path(newparent, createparents)
        if not any(k in kwargs for k in {'start', 'stop', 'step'}):
            ret = newparent.create_array(newname, obj=self, **kwargs)
        else:
            slc = slice(*(kwargs.pop(k, None)
                          for k in ('start', 'stop', 'step')))
            tmp_data = self[slc]
            ret = newparent.create_array(newname, obj=tmp_data, **kwargs)

        if copyuserattrs:
            for k, v in self.attrs.items():
                if k.lower() == 'title' and 'title' in kwargs:
                    continue
                ret.attrs[k] = v
        return ret

    def truncate(self):
        raise TypeError('can not truncate Arrays')
