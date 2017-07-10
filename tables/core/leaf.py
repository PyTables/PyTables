from .node import Node
from tables.utils import byteorders
from ..exceptions import ClosedNodeError


class Leaf(Node):
    @property
    def byteorder(self):
        return byteorders[self.dtype.byteorder]

    @property
    def chunkshape(self):
        return self.backend.chunks

    @property
    def dtype(self):
        return self.backend.dtype

    @property
    def shape(self):
        return self.backend.shape

    @property
    def size_on_disk(self):
        return self.backend.size_on_disk

    @property
    def chunk_shape(self):
        return self.backend.chunk_shape

    def __len__(self):
        return len(self.backend)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def nrows(self):
        if len(self.shape) > 0:
            return int(self.shape[self.maindim])
        # Scalar dataset
        else:
            return 1 if self.shape == () else len(self)

    def __getitem__(self, item):
        if not self._v_file._isopen or self._isopen:
            raise ClosedNodeError
        return self.backend.__getitem__(item)

    def __setitem__(self, item, value):
        if not self._v_file._isopen or self._isopen:
            raise ClosedNodeError
        return self.backend.__setitem__(item, value)

    @property
    def maindim(self):
        return 0

    def get_attr(self, attr):
        return self.attrs[attr]

    def _process_range(self, start, stop, step, dim=None, warn_negstep=True):
        # This method is appropriate for calls to __getitem__ methods
        if dim is None:
            nrows = self.nrows
        else:
            nrows = self.shape[dim]
        if warn_negstep and step and step < 0:
            raise ValueError("slice step cannot be negative")
        return slice(start, stop, step).indices(nrows)

    def _process_range_read(self, start, stop, step, warn_negstep=True):
        # This method is appropriate for calls to read() methods
        nrows = self.nrows
        if start is not None and stop is None and step is None:
            # Protection against start greater than available records
            # nrows == 0 is a special case for empty objects
            if nrows > 0 and start >= nrows:
                raise IndexError("start of range (%s) is greater than "
                                 "number of rows (%s)" % (start, nrows))
            step = 1
            if start == -1:  # corner case
                stop = nrows
            else:
                stop = start + 1
        # Finally, get the correct values (over the main dimension)
        start, stop, step = self._process_range(start, stop, step,
                                                warn_negstep=warn_negstep)
        return (start, stop, step)

    def flush(self):
        pass

    def __str__(self):
        """The string representation for this object is its pathname in the
        HDF5 object tree plus some additional metainfo."""

        # Get this class name
        classname = self.__class__.__name__
        # The title
        title = self.title
        # The filters
        filters = ""
        if self.filters.fletcher32:
            filters += ", fletcher32"
        if self.filters.complevel:
            if self.filters.shuffle:
                filters += ", shuffle"
            if self.filters.bitshuffle:
                filters += ", bitshuffle"
            filters += ", %s(%s)" % (self.filters.complib,
                                     self.filters.complevel)
        return "%s (%s%s%s) %r" % \
               (self._v_pathname, classname, self.shape, filters, title)
