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
    def row_selection(chunk):
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
        self.write_target.append(self.data)

    def update(self):
        self.write_target[self.nrow + self.offest] = self.data

    def __contains__(self, item):
        return item in self.dtype.names

    def __getitem__(self, key):
        return self.data[0][key]

    def __setitem__(self, key, value):
        self.data[0][key] = value


class PyTablesLeaf(PyTablesNode):
    @property
    def dtype(self):
        return self.backend.dtype

    @property
    def shape(self):
        return self.backend.shape

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

    def iter_rows(self, *, chunk_selector=None, row_selector=None):
        from tables.core import Row
        if row_selector is None:
            row_selector = all_row_selector

        row = Row(self)
        for (j, ), chunk in self.iter_chunks(chunk_selector=chunk_selector):
            row.read_src = chunk
            row.offset = j * self.chunk_shape[0]
            for r in row_selector(j, chunk):
                row.crow = r
                yield row

    def where(self, condition, condvars, start=None, stop=None, step=None, *,
              sub_chunk_selector_factory=None):

        if sub_chunk_selector_factory is None:
            sub_chunk_selector_factory = dflt_sub_chunk_selector_factory
        # Adjust the slice to be used.
        start, stop, step = self._process_range_read(start, stop, step)
        if start >= stop:  # empty range, reset conditions
            return iter([])
        # TODO write numexpr -> selector code
        if not callable(condition):
            raise NotImplementedError("non lambda selection not done yet")
        # TODO write code to get chunk selector from index
        selector = None

        sub_chunk_select = sub_chunk_selector_factory(condition)

        yield from self.backend.iter_with_selectors(
            chunk_selector=selector, sub_chunk_selector=sub_chunk_select)

    def _required_expr_vars(self, expression, uservars, depth=1):
        # Get the names of variables used in the expression.
        exprvarscache = self._exprvars_cache
        if expression not in exprvarscache:
            # Protection against growing the cache too much
            if len(exprvarscache) > 256:
                # Remove 10 (arbitrary) elements from the cache
                for k in list(exprvarscache.keys())[:10]:
                    del exprvarscache[k]
            cexpr = compile(expression, '<string>', 'eval')
            exprvars = [var for var in cexpr.co_names
                        if var not in ['None', 'False', 'True']
                        and var not in numexpr_functions]
            exprvarscache[expression] = exprvars
        else:
            exprvars = exprvarscache[expression]

        # Get the local and globbal variable mappings of the user frame
        # if no mapping has been explicitly given for user variables.
        user_locals, user_globals = {}, {}
        if uservars is None:
            # We use specified depth to get the frame where the API
            # callable using this method is called.  For instance:
            #
            # * ``table._required_expr_vars()`` (depth 0) is called by
            # * ``table._where()`` (depth 1) is called by
            # * ``table.where()`` (depth 2) is called by
            # * user-space functions (depth 3)
            user_frame = sys._getframe(depth)
            user_locals = user_frame.f_locals
            user_globals = user_frame.f_globals

        colinstances = self.colinstances
        tblfile, tblpath = self._v_file, self._v_pathname
        # Look for the required variables first among the ones
        # explicitly provided by the user, then among implicit columns,
        # then among external variables (only if no explicit variables).
        reqvars = {}
        for var in exprvars:
            # Get the value.
            if uservars is not None and var in uservars:
                val = uservars[var]
            elif var in colinstances:
                val = colinstances[var]
            elif uservars is None and var in user_locals:
                val = user_locals[var]
            elif uservars is None and var in user_globals:
                val = user_globals[var]
            else:
                raise NameError("name ``%s`` is not defined" % var)

            # Check the value.
            if hasattr(val, 'pathname'):  # non-nested column
                if val.shape[1:] != ():
                    raise NotImplementedError(
                        "variable ``%s`` refers to "
                        "a multidimensional column, "
                        "not yet supported in conditions, sorry" % var)
                if (val._table_file is not tblfile or
                        val._table_path != tblpath):
                    raise ValueError("variable ``%s`` refers to a column "
                                     "which is not part of table ``%s``"
                                     % (var, tblpath))
                if val.dtype.str[1:] == 'u8':
                    raise NotImplementedError(
                        "variable ``%s`` refers to "
                        "a 64-bit unsigned integer column, "
                        "not yet supported in conditions, sorry; "
                        "please use regular Python selections" % var)
            elif hasattr(val, '_v_colpathnames'):  # nested column
                raise TypeError(
                    "variable ``%s`` refers to a nested column, "
                    "not allowed in conditions" % var)
            else:  # only non-column values are converted to arrays
                # XXX: not 100% sure about this
                if isinstance(val, six.text_type):
                    val = numpy.asarray(val.encode('ascii'))
                else:
                    val = numpy.asarray(val)
            reqvars[var] = val
        return reqvars


    def append(self, rows):
        rows = np.rec.array(rows, self.dtype)
        cur_count = len(self)
        self._backend.resize((cur_count + len(rows), ))
        self[cur_count:] = rows

    def modify_rows(self, start=None, stop=None, step=None, rows=None):
        if rows is None:
            return
        self[start:stop:step] = rows


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
