import sys
import numpy as np
from .leaf import Leaf
from .. import Atom
from ..utils import byteorders
from ..exceptions import ClosedNodeError
from ..flavor import array_of_flavor


class Array(Leaf):
    def __init__(self, title='', _atom=None, new=False, **kwargs):
        super().__init__(**kwargs)
        if new:
            self.attrs['TITLE'] = title
            self.attrs['CLASS'] = self.__class__.__name__.upper()
            self.flavor = 'numpy'
        else:
            try:
                self._flavor = self.attrs['FLAVOR']
            except KeyError:
                self._flavor = 'numpy'

        self.atom = _atom
        if _atom is None or _atom.shape == ():
            if self.dtype == np.dtype('O'):
                self.atom = Atom.from_dtype(np.array(self[()]).dtype, dflt=self.backend.fillvalue)
            else:
                self.atom = Atom.from_dtype(self.dtype, dflt=self.backend.fillvalue)
        self.nrow = None
        # Provisional for test
        if not hasattr(self, 'extdim'):
            self.extdim = -1
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

        if not self._v_file._isopen:
            raise ClosedNodeError
        # Scalar dataset
        if self.shape == ():
            arr = self[()]
            nrowstoread = 1
        else:
            (start, stop, step) = self._process_range_read(start, stop, step)
            slices = tuple(slice(start, stop, step) if i == self.maindim else slice(None)
                           for i in range(len(self.shape)))
            arr = self[slices]
            nrowstoread = len(range(start, stop, step))
            if arr.size == 0:
                try:
                    aux = list(self.backend.maxshape)
                    aux[aux.index(None)] = 0
                    arr = np.reshape(arr, aux)
                except ValueError:
                    pass

        if isinstance(arr, np.ndarray) and byteorders[arr.dtype.byteorder] != sys.byteorder:
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

            # Check empty dataset or selection
            if self.nrows != 0 and nrowstoread != 0:
                if self.shape == ():
                    self.backend.read_direct(out)
                else:
                    slices = tuple(slice(start, stop, step) if i == self.maindim else slice(None)
                                   for i in range(len(self.shape)))
                    slices2 = tuple(slice(0, nrowstoread, 1) if i == self.maindim else slice(None)
                                   for i in range(len(self.shape)))
                    self.backend.read_direct(out, np.s_[slices], np.s_[slices2])
            return out

        if self.flavor != 'numpy':
            arr = array_of_flavor(arr, self.flavor)

        return arr

    def iterrows(self, start=None, stop=None, step=None):
        start, stop, step = self._process_range(start, stop, step,
                                                warn_negstep=False)
        # Scalar dataset
        if self.shape == ():
            self.nrow = 0
            yield np.array(self[()])
        else:
            self.nrow = start-step
            aux = np.swapaxes(self[()], self.maindim, 0)
            for r in aux[start:stop:step]:
                self.nrow += step
                yield r

    __iter__ = iterrows

    def copy(self, newparent=None, newname=None,
             overwrite=False, createparents=True, *,
             copyuserattrs=True,
             **kwargs):

        if not hasattr(newparent, 'create_array'):
            newparent = self.root._get_or_create_path(newparent, createparents)
        if self.__class__.__name__ == 'Array':
            create_function = newparent.create_array
        elif self.__class__.__name__ == 'CArray':
            create_function = newparent.create_carray
        else:
            create_function = newparent.create_earray
        if not any(k in kwargs for k in {'start', 'stop', 'step'}):
            ret = create_function(newname, obj=self, **kwargs)
        else:
            slc = slice(*(kwargs.pop(k, None)
                          for k in ('start', 'stop', 'step')))
            tmp_data = self[slc]
            ret = create_function(newname, obj=tmp_data, **kwargs)

        if copyuserattrs:
            for k, v in self.attrs.items():
                if k.lower() == 'title' and 'title' in kwargs:
                    continue
                if k.lower() == 'flavor':
                    ret._flavor = v
                ret.attrs[k] = v
        return ret

    def truncate(self):
        raise TypeError('can not truncate Arrays')
