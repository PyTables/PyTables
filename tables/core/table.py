import numpy as np
from tables.path import split_path
import operator
from tables.utils import is_idx
from tables.table import _index_pathname_of_column_
from .leaf import Leaf

import numexpr
import sys
from tables.description import descr_from_dtype, Description


def all_row_selector(chunk_id, chunk):
    yield from range(len(chunk))


def dispatch(value):
    """Wrap dataset for """
    if value.attrs['CLASS'] == 'TABLE':
        return Table(value)
    return value


def dflt_row_selector_factory(condition, *, start, stop, step,
                              chnk_sz, depth=4):
    """Return a row selector depending on the format of condition."""

    def row_selection(chunkid, chunk):
        offset = chnk_sz * chunkid

        for i, r in enumerate(chunk):
            nabs = i + offset
            if not (nabs < stop):
                return
            if nabs >= start and ((nabs - start) % step == 0) and condition(r):
                yield i

    def inkernel_row_selection(chunkid, chunk):
        offset = chnk_sz * chunkid
        # Get a dictionary with the columns in chunk
        cols = dict((name, chunk[name]) for name in chunk.dtype.names)
        # Get locals and globals
        frame = sys._getframe(depth)
        cols.update(frame.f_locals)
        cols.update(frame.f_globals)
        # Evaluate the condition
        out = numexpr.evaluate(condition, local_dict=cols)
        for i, r in enumerate(out):
            nabs = i + offset
            if not (nabs < stop):
                return
            if r and nabs >= start and ((nabs - start) % step == 0):
                yield i

    if callable(condition):
        return row_selection
    elif type(condition) == str:
        return inkernel_row_selection


class RowAppender:
    def __init__(self, write_target):
        self.write_target = write_target
        self._data = np.empty(1, dtype=self.dtype)[0]

    @property
    def dtype(self):
        return self.write_target.dtype

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def __contains__(self, item):
        return item in self.dtype.names

    def __getitem__(self, key):
        if isinstance(key, slice):
            return tuple(self.data)[key]
        try:
            return self.data[key]
        except IndexError as e:
            raise TypeError from e
        except ValueError as e:
            raise KeyError from e

    def __setitem__(self, key, value):
        self.data[key] = value

    def append(self):
        d = np.rec.array(self.data)
        d.resize(1)
        self.write_target.append(d)

    def fetch_all_fields(self):
        return self.data


class Row(RowAppender):

    def __init__(self, write_target):
        super().__init__(write_target)
        self._read_src = None
        self._crow = -1
        self._data = None
        self._offset = 0

    @property
    def nrow(self):
        return self._crow + self._offset

    @property
    def crow(self):
        return self._crow

    @crow.setter
    def crow(self, value):
        self._crow = value
        self._data = None

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, value):
        self._offset = value
        self._data = None

    @property
    def read_src(self):
        return self._read_src

    @read_src.setter
    def read_src(self, value):
        self._read_src = value

    @property
    def data(self):
        if self._data is not None:
            return self._data
        if self.nrow >= 0:
            self._data = self._read_src[self._crow]
        else:
            self._data = np.empty(1, dtype=self.dtype)[0]
        return self._data

    def update(self):
        self.write_target[self.nrow] = self.data


class Column:

    def __init__(self, table, pathname):
        self.table = table
        self.name = split_path(pathname)[-1]
        self.pathname = pathname
        self.dtype = table.dtype[pathname]

    @property
    def indexpath(self):
        return _index_pathname_of_column_(self.table.pathname, self.pathname)

    @property
    def index(self):
        try:
            return self.table.parentnode[self.indexpath]
        except KeyError:
            return None

    @property
    def shape(self):
        return (self.table.nrows,) + self.dtype.shape

    @property
    def is_indexed(self):
        return self.index is not None

    def __getitem__(self, key):
        table = self.table
        if isinstance(key, tuple) and len(key) == 1:
            key = key[0]
        if is_idx(key):
            key = operator.index(key)
            # Index out of range protection
            if key >= table.nrows:
                raise IndexError("Index out of range")
            if key < 0:
                # To support negative values
                key += table.nrows
            return table.read(key, key + 1, 1, self.pathname)[0]
        elif isinstance(key, slice):
            return table.read(key.start, key.stop, key.step, self.pathname)
        else:
            raise TypeError(
                "'%s' key type is not valid in this context" % key)

    def __iter__(self):
        """Iterate through all items in the column."""
        table = self.table
        for row in table.iterrows():
            yield row[self.pathname]

    def __setitem__(self, key, value):
        table = self.table
        table._v_file._check_writable()

        # Generalized key support not there yet, but at least allow
        # for a tuple with one single element (the main dimension).
        # (key,) --> key
        if isinstance(key, tuple) and len(key) == 1:
            key = key[0]

        if is_idx(key):
            key = operator.index(key)

            # Index out of range protection
            if key >= table.nrows:
                raise IndexError("Index out of range")
            if key < 0:
                # To support negative values
                key += table.nrows
            return table.modify_column(key, key + 1, 1,
                                       [[value]], self.pathname)
        elif isinstance(key, slice):
            (start, stop, step) = table._process_range(
                key.start, key.stop, key.step)
            return table.modify_column(start, stop, step,
                                       value, self.pathname)

    def __str__(self):
        """The string representation for this object."""
        tablepathname = self.table.pathname
        pathname = self.pathname.replace('/', '.')
        classname = self.__class__.__name__
        shape = self.shape
        tcol = self.dtype
        return "%s.cols.%s (%s%s, %s, idx=%s)" % \
               (tablepathname, pathname, classname, shape, tcol, self.index)

    def __repr__(self):
        """A detailed string representation for this object."""
        return str(self)


class Cols:
    def __init__(self, table):
        self.table = table
        self.dtype = table.dtype
        for name in self.dtype.names:
            setattr(self, name, Column(table, name))

    def __len__(self):
        return len(self.dtype)

    def __getitem__(self, k):
        return self.table[k]

    def __setitem__(self, k, v):
        self.table[v] = v

    def _f_col(self, colname):
        return getattr(self, colname)


class Table(Leaf):
    ndim = 1
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._exprvars_cache = {}
        self.colinstances = {}
        self.cols = Cols(self)
        colinstances, cols = self.colinstances, self.cols
        for colpathname in self.dtype.names:
            colinstances[colpathname] = cols._f_col(colpathname)
        self.row = RowAppender(self)

        self.colpathnames = []
        """A list containing the pathnames of *bottom-level* columns in the
        table.

        These are the leaf columns obtained when walking the
        table description left-to-right, bottom-first.  Columns inside a
        nested column have slashes (/) separating name components in
        their pathname.
        """
        self.coldescrs = {}
        """Maps the name of a column to its Col description (see
        :ref:`ColClassDescr`)."""
        self.coltypes = {}
        """Maps the name of a column to its PyTables data type."""
        self.coldtypes = {}
        """Maps the name of a column to its NumPy data type."""
        self.coldflts = {}
        """Maps the name of a column to its default value."""
        self.colindexed = {}
        """Condition function and argument list for selection of values."""
        self._seqcache_key = None
        """The key under which to save a query's results (list of row indexes)
        or None to not save."""

        self.colpathnames = [
            col._v_pathname for col in self.description._f_walk()
            if not hasattr(col, '_v_names')]  # bottom-level

        # Get info about columns
        for colobj in self.description._f_walk(type="Col"):
            colname = colobj._v_pathname
            # Get the column types, types and defaults
            self.coldescrs[colname] = colobj
            self.coltypes[colname] = colobj.type
            self.coldtypes[colname] = colobj.dtype
            self.coldflts[colname] = colobj.dflt

        # Assign _v_dtype for this table
        self._v_dtype = self.description._v_dtype

    @property
    def pathname(self):
        return self.backend.name

    @property
    def description(self):
        return descr_from_dtype(self.dtype)[0]

    @property
    def colnames(self):
        return self.dtype.names

    def __getitem__(self, k):
        return np.rec.array(super().__getitem__(k))

    def _iter_rows(self, *, chunk_selector=None, row_selector=None):
        if row_selector is None:
            row_selector = all_row_selector

        row = Row(self)
        for (j, ), chunk in self.backend.iter_chunks(
                chunk_selector=chunk_selector):
            row.read_src = chunk
            row.offset = j * self.chunk_shape[0]
            for r in row_selector(j, chunk):
                row.crow = r
                yield row

    def where(self, condition, condvars=None, start=None, stop=None,
              step=None, *, row_selector_factory=None):

        if row_selector_factory is None:
            row_selector_factory = dflt_row_selector_factory
        # Adjust the slice to be used.
        start, stop, step = self._process_range_read(start, stop, step)
        if start >= stop:  # empty range, reset conditions
            return iter([])
        # TODO write numexpr -> selector code
        if not (callable(condition) or type(condition)):
            raise NotImplementedError("condition must be either a callable or a string")

        chk_sz, = self.chunk_shape
        # TODO write code to get chunk selector from index
        min_chunk = start // chk_sz
        max_chunk = stop // chk_sz

        def chunk_selector(j):
            j, = j
            if j < min_chunk or j > max_chunk:
                return False
            return True

        row_selector = row_selector_factory(condition, start=start, stop=stop,
                                            step=step, chnk_sz=chk_sz)

        yield from self._iter_rows(chunk_selector=chunk_selector,
                                   row_selector=row_selector)

    def append_where(self, dest, *args, **kwargs):
        # get the iterator for the condition
        wh_itr = self.where(*args, **kwargs)
        # get the first row
        row = next(wh_itr)
        # reset how the table that this will try to write to (this is exciting)
        row.write_target = dest
        # add the first result
        row.append()
        # add the rest of the results
        for row in wh_itr:
            row.append()

    def read_where(self, *args, field=None, **kwargs):
        wh_itr = self.where(*args, **kwargs)
        if field is None:
            return np.fromiter((r.data for r in wh_itr), dtype=self.dtype)
        return np.fromiter((r.data[field] for r in wh_itr),
                           dtype=self.dtype[field])

    def get_where_list(self, *args, sort=False, **kwargs):
        coords = np.fromiter((r.nrow for r in self.where(*args, **kwargs)),
                             dtype='i8')
        if sort:
            coords.sort()
        return coords

    def append(self, rows):
        if not hasattr(self, 'append_buffer'):
            self.append_buffer = np.empty(self.chunk_shape[0], self.dtype)
            self.append_buffer_remainder = self.chunk_shape[0]
        rows = np.rec.array(rows, self.dtype)
        # XXX FIXME
#        if len(rows) < self.append_buffer_remainder:
#            remainder = self.append_buffer_remainder
#            self.append_buffer[-remainder:len(rows)] = rows
#            self.append_buffer_remainder -= len(rows)
#        else:
#            # Append to the existing values in buffer and flush
#            remainder = self.append_buffer_remainder
#            self.append_buffer[-remainder:] = rows[:remainder]
#            buflen = len(self.append_buffer)
#            cur_count = len(self)
#            self._backend.resize((cur_count + buflen,))
#            self[cur_count:] = self.append_buffer
#            if len(rows) - remainder > buflen:
#                # Flush the remainder
#                cur_count = len(self)
#                self._backend.resize((cur_count + len(rows) - remainder,))
#                self[cur_count:] = rows[-remainder:]
#                self.append_buffer_remainder = buflen
#            else:
#                # Add the remainder to the buffer
#                self.append_buffer[:len(rows)-remainder] = rows[remainder:]
#                self.append_buffer_remainder = buflen - (len(rows) - remainder)

    def modify_rows(self, start=None, stop=None, step=None, rows=None):
        if rows is None:
            return
        self[start:stop:step] = rows

    def itersequence(self, sequence):
        from itertools import groupby, repeat
        chk_sz, = self.chunk_shape
        dm = map(divmod, sequence, repeat(chk_sz))
        # TODO cache chunks?
        row = Row(self)
        for k, g in groupby(dm, key=lambda x: x[0]):
            chunk = self[k*chk_sz: (k+1)*chk_sz]
            indx = [_[1] for _ in g]
            row.offset = chk_sz * k
            row.read_src = chunk
            for r in indx:
                row.crow = r
                yield row

    def itersorted(self, *args, **kwargs):
        raise NotImplementedError('requires index')

    def read_sorted(self, *args, **kwargs):
        raise NotImplementedError('requires index')

    def iterrows(self, start=None, stop=None, step=None):
        start, stop, step = self._process_range(start, stop, step,
                                                warn_negstep=False)
        yield from self.itersequence(range(start, stop, step))

    __iter__ = iterrows

    def read(self, start=None, stop=None, step=None, field=None, out=None):
        start, stop, step = self._process_range(start, stop, step,
                                                warn_negstep=False)
        if field is None:
            dtype = self.dtype
        else:
            dtype = self.dtype[field]

        seq = range(start, stop, step)
        if out is None:
            out = np.empty(len(seq), dtype=dtype)

        for j, r in enumerate(self.itersequence(seq)):
            if field is None:
                out[j] = r
            else:
                out[j] = r[field]
        return out

    def read_coordinates(self, coords, field=None):
        if field is None:
            return np.fromiter(self.itersequence(coords), dtype=self.dtype)
        return np.fromiter((r[field] for r in self.itersequence(coords)),
                           dtype=self.dtype[field])

    def col(self, name):
        return self.read(field=name)

    def remove_rows(self, start=None, stop=None, step=None):
        old = len(self)
        start, stop, step = self._process_range(start, stop, step)
        del self.backend[start:stop:step]
        return old - len(self)

    def remove_row(self, n):
        return self.remove_rows(start=n, stop=n+1)

    def flush(self):
        print('flushing')
        if hasattr(self, 'append_buffer'):
            cur_count = len(self)
            buflen = len(self.append_buffer) - self.append_buffer_remainder
            self._backend.resize((cur_count + buflen,))
            self[cur_count:] = self.append_buffer[:buflen]
        return self.backend.flush()

    def copy(self, newparent=None, newname=None, overwrite=False,
             createparents=False, **kwargs):
        ...

    def modify_column(self, start=None, stop=None, step=None,
                      column=None, colname=None):
        if not isinstance(colname, str):
            raise TypeError("The 'colname' parameter must be a string.")
        if column is None:      # Nothing to be done
            return 0
        return self.modify_columns(start, stop, step, [column], [colname])

    def modify_colums(self, start=None, stop=None, step=None,
                      columns=None, names=None):
        if step is None:
            step = 1
        if columns is None:      # Nothing to be done
            return 0
        if start is None:
            start = 0
        if start < 0:
            raise ValueError("'start' must have a positive value.")
        if step < 1:
            raise ValueError(
                "'step' must have a value greater or equal than 1.")
        objcols = [self.dtype[name] for name in names]
        columns = np.asarray(columns, dtype=objcols)
        if stop is None:
            # compute the stop value. start + len(rows)*step does not work
            stop = start + (len(self.table) - 1) * step + 1
        (start, stop, step) = self._process_range(start, stop, step)
        if stop > self.nrows:
            raise IndexError("This modification will exceed the length of "
                             "the table. Giving up.")
        # Compute the number of rows to read.
        nrows = len(range(0, stop - start, step))
        if len(self.table) < nrows:
            raise ValueError("The value has not enough elements to fill-in "
                             "the specified range")
        for row, v in zip(self.iterrows(start, stop, step), columns):
            for name in names:
                row[name] = v[name]
            row.update()
        return nrows
