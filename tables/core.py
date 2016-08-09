import numpy as np
from tables import Description


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


class PyTablesDataset(object):
    pass


class PyTablesLeaf:
    @property
    def shape(self):
        return self.backend.shape

    @property
    def nrows(self):
        return int(self.shape[self.maindim])

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


@forwarder(['attrs'], ['open', 'close'])
class PyTableNode(HasTitle, HasBackend):
    pass


@forwarder(['attrs', 'shape', 'dtype'],
           ['__len__', '__setitem__', '__getitem__'])
class PyTablesTable(PyTablesLeaf):
    def __init__(self, backend):
        self._backend = backend
        self._exprvars_cache = {}

    def __getitem__(self, k):
        return self._backend[k]

    def __setitem__(self, k, v):
        self._backend[k] = v

    def where(self, condition, condvars, start=None, stop=None, step=None):
        # Adjust the slice to be used.
        (start, stop, step) = self._process_range_read(start, stop, step)
        if start >= stop:  # empty range, reset conditions
            return iter([])
        # Compile the condition and extract usable index conditions.
        condvars = self._required_expr_vars(condition, condvars, depth=3)
        compiled = self._compile_condition(condition, condvars)

        # Can we use indexes?
        if compiled.index_expressions:
            chunkmap = _table__where_indexed(
                self, compiled, condition, condvars, start, stop, step)
            if not isinstance(chunkmap, numpy.ndarray):
                # If it is not a NumPy array it should be an iterator
                # Reset conditions
                self._use_index = False
                self._where_condition = None
                # ...and return the iterator
                return chunkmap
        else:
            chunkmap = None  # default to an in-kernel query

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

        # Get the local and global variable mappings of the user frame
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



class PyTableFile(PyTableNode):
    @property
    def root(self):
        return self['/']

    def create_table(self, where, name, desc, *args, **kwargs):
        desc = Description(desc.columns)
        return where.create_table(name, desc, *args, **kwargs)


class PyTablesGroup(PyTableNode):
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


class PyTablesDataset(PyTableNode):
    pass

