import numpy as np
from tables import Description


def forwarder(forwarded_props, forwarded_methods,
              remapped_keys=None):
    def inner(cls):
        for k in forwarded_props:
            def bound_getter(self, k=k):
                return getattr(self._backend, k)
            setattr(cls, k, property(bound_getter))
        for k in forwarded_methods:
            def make_bound(key):
                def bound_method(self, *args, **kwargs):
                    return getattr(self._backend, key)(*args, **kwargs)
                return bound_method
            setattr(cls, k, make_bound(k))

        return cls
    return inner


class HasBackend:
    @property
    def backend(self):
        return self._backend

    def __init__(self, *, backend):
        self._backend = backend


class HasTitle:
    @property
    def title(self):
        return self.backend.attrs.get('TITLE', None)

    @title.setter
    def title(self, title):
        self.backend.attrs['TITLE'] = title


class PyTablesNode(HasTitle, HasBackend):
    @property
    def attrs(self):
        return self.backend.attrs

    def open(self):
        return self.backend.open()

    def close(self):
        return self.backend.close()


def all_row_selector(chunk_id, chunk):
    yield from range(len(chunk))


def description_to_dtype(desc):
    try:
        return desc._v_dtype
    except AttributeError:
        return np.dtype(desc, copy=True)


def dispatch(value):
    """Wrap dataset for PyTables"""
    if value.attrs['CLASS'] == 'TABLE':
        return PyTablesTable(value)
    return value


def dflt_row_selector_factory(condition):
    def row_selection(chunkid, chunk):
        for i, r in enumerate(chunk):
            if condition(r):
                yield i

    return row_selection


class Row:
    def fetch_all_fields(self):
        raise NotImplementedError()

    def __init__(self, write_target):
        self._read_src = None
        self._crow = -1
        self._data = None
        self._offset = 0
        self.write_target = write_target

    @property
    def dtype(self):
        return self.table.dtype

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
            self._data = np.empty(1, dtype=self.dtype)
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    def append(self):
        d = np.rec.array(self.data)
        d.resize(1)
        self.write_target.append(d)

    def update(self):
        self.write_target[self.nrow] = self.data

    def __contains__(self, item):
        return item in self.dtype.names

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value


class PyTablesLeaf(PyTablesNode):
    @property
    def dtype(self):
        return self.backend.dtype

    @property
    def shape(self):
        return self.backend.shape

    @property
    def chunk_shape(self):
        return self.backend.chunk_shape

    def __len__(self):
        return self.backend.__len__()

    @property
    def nrows(self):
        return int(self.shape[self.maindim])

    def __getitem__(self, item):
        return self.backend.__getitem__(item)

    def __setitem__(self, item, value):
        return self.backend.__setitem__(item, value)

    @property
    def maindim(self):
        return 0

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


class PyTablesArray(PyTablesLeaf):
    pass


class PyTablesTable(PyTablesLeaf):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._exprvars_cache = {}

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

    def where(self, condition, condvars, start=None, stop=None, step=None, *,
              row_selector_factory=None):

        if row_selector_factory is None:
            row_selector_factory = dflt_row_selector_factory
        # Adjust the slice to be used.
        start, stop, step = self._process_range_read(start, stop, step)
        if start >= stop:  # empty range, reset conditions
            return iter([])
        # TODO write numexpr -> selector code
        if not callable(condition):
            raise NotImplementedError("non lambda selection not done yet")
        # TODO write code to get chunk selector from index
        chunk_selector = None

        row_selector = row_selector_factory(condition)

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
        rows = np.rec.array(rows, self.dtype)
        cur_count = len(self)
        self._backend.resize((cur_count + len(rows), ))
        self[cur_count:] = rows

    def modify_rows(self, start=None, stop=None, step=None, rows=None):
        if rows is None:
            return
        self[start:stop:step] = rows

    def itersequence(self, sequence):
        from itertools import groupby, repeat
        chk_sz, = self.chunk_shape
        dm = map(divmod, sequence, repeat(chk_sz))
        # TODO cache chunks?
        for k, g in groupby(dm, key=lambda x: x[0]):
            chunk = self[k*chk_sz: (k+1)*chk_sz]
            indx = [_[1] for _ in g]
            yield from chunk[indx]

    def itersorted(self, *args, **kwargs):
        raise NotImplementedError('requires index')

    def read_sorted(self, *args, **kwargs):
        raise NotImplementedError('requires index')

    def iterrows(self, start, stop, step):
        yield from self.itersequence(range(start, stop, step))


class PyTableFile(PyTablesNode):
    @property
    def root(self):
        return PyTablesGroup(backend=self.backend['/'])

    def __iter__(self):
        return iter(self.root)

    def create_array(self, where, *args, **kwargs):
        return where.create_array(*args, **kwargs)

    def create_group(self, where, *args, **kwargs):
        return where.create_group(*args, **kwargs)

    def create_table(self, where, name, desc, *args, **kwargs):
        desc = Description(desc.columns)
        return where.create_table(name, desc, *args, **kwargs)


class PyTablesGroup(PyTablesNode):
    def __getitem__(self, item):
        value = self.backend[item]
        if hasattr(value, 'dtype'):
            return dispatch(value)
        # Group?
        return PyTablesGroup(backend=value)

    @property
    def parent(self):
        return PyTablesGroup(backend=self.backend.parent)

    @property
    def filters(self):
        return self.backend.attrs.get('FILTERS', None)

    @filters.setter
    def filters(self, filters):
        # TODO how we persist this? JSON?
        self.backend.attrs['FILTERS'] = filters

    @property
    def _v_pathname(self):
        return self.backend.name

    def __iter__(self):
        for child in self.backend.values():
            yield child.name

    def create_array(self, name, obj, title='', byte_order='I', **kwargs):
        obj = np.asarray(obj)
        dtype = obj.dtype.newbyteorder(byte_order)

        dataset = self.backend.create_dataset(name, data=obj,
                                              dtype=dtype,
                                              **kwargs)
        dataset.attrs['TITLE'] = title
        dataset.attrs['CLASS'] = 'ARRAY'
        return PyTablesArray(backend=dataset)

    def create_group(self, name, title=''):
        g = PyTablesGroup(backend=self.backend.create_group(name))
        g.attrs['TITLE'] = title
        return g

    def create_table(self, name, description=None, title='',
                     filters=None, expectedrows=10000,
                     byte_order='I',
                     chunk_shape=None, obj=None, **kwargs):
        """ TODO write docs"""
        if obj is None and description is not None:
            dtype = description_to_dtype(description)
            obj = np.empty(shape=(0,), dtype=dtype)
        elif obj is not None and description is not None:
            dtype = description_to_dtype(description)
            obj = np.asarray(obj)
        elif description is None:
            obj = np.asarray(obj)
            dtype = obj.dtype
        else:
            raise Exception("BOOM")
        # newbyteorder makes a copy
        # dtype = dtype.newbyteorder(byte_order)

        if chunk_shape is None:
            # chunk_shape = compute_chunk_shape_from_expected_rows(dtype, expectedrows)
            ...
        # TODO filters should inherit the ones defined at group level
        # filters = filters + self.attrs['FILTERS']

        # here the backend creates a dataset

        # TODO pass parameters kwargs?
        dataset = self.backend.create_dataset(name, data=obj,
                                              dtype=dtype,
                                              maxshape=(None,),
                                              chunk_shape=chunk_shape,
                                              **kwargs)
        dataset.attrs['TITLE'] = title
        dataset.attrs['CLASS'] = 'TABLE'
        return PyTablesTable(backend=dataset)
