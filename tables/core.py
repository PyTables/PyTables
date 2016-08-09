from tables import abc
from tables import Description
import numpy as np


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


@forwarder(['attrs', 'shape', 'dtype'],
           ['__len__', '__setitem__', '__getitem__'])
class PyTablesTable(object):
    def __init__(self, backend):
        self._backend = backend

    def __getitem__(self, k):
        return self._backend[k]

    def __setitem__(self, k, v):
        self._backend[k] = v

    def where(self, *args, **kwargs):
        # here there be dragons
        ...

    def append(self, rows):
        rows = np.rec.array(rows, self.dtype)
        cur_count = len(self)
        self._backend.resize((cur_count + len(rows), ))
        self[cur_count:] = rows

    def modify_rows(self, start=None, stop=None, step=None, rows=None):
        if rows is None:
            return

        self[start:stop:step] = rows


@forwarder(['attrs'], ['open', 'close'])
class PyTableNode:
    @property
    def backend(self):
        return self._backend

    def __init__(self, *, backend):
        self._backend = backend

    def __getitem__(self, item):
        value = self.backend[item]
        if hasattr(value, 'dtype'):
            return dispatch(value)
        # Group?
        return PyTablesGroup(backend=value)


class PyTableFile(PyTableNode):
    @property
    def root(self):
        return self['/']

    def create_table(self, where, name, desc, *args, **kwargs):
        desc = Description(desc.columns)
        return where.create_table(name, desc, *args, **kwargs)


class PyTablesGroup(PyTableNode):

    def open(self):
        return self.backend.open()

    def close(self):
        return self.backend.close()

    @property
    def parent(self):
        return PyTablesGroup(backend=self.backend.parent)

    @property
    def title(self):
        return self.backend.attrs.get('TITLE', None)

    @title.setter
    def title(self, title):
        self.backend.attrs['TITLE'] = title

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
