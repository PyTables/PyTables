from .node import PyTablesNode
from .table import PyTablesTable
from .array import PyTablesArray
from tables import abc
from tables import Description
from tables import IsDescription

import numpy as np


def dtype_from(something):
    if isinstance(something, np.dtype):
        return something

    if isinstance(something, np.ndarray):
        return something.dtype

    if isinstance(something, dict):
        return Description(something)._v_dtype

    if issubclass(something, IsDescription):
        return Description(something().columns)._v_dtype

    raise NotImplementedError()


class HasChildren:
    def __iter__(self):
        for child in self.backend.values():
            yield child.name

    def __getitem__(self, item):
        value = self.backend[item]
        if isinstance(value, abc.Group):
            return PyTablesGroup(backend=value)

        if isinstance(value, abc.Dataset):
            if value.attrs['CLASS'] == 'TABLE':
                return PyTablesTable(backend=value)
            elif value.attrs['CLASS'] == 'ARRAY':
                return PyTablesArray(backend=value)

        raise NotImplementedError()

    def __getattr__(self, attr):
        return self.__getitem__(attr)

    def rename_node(self, old, new_name):
        if isinstance(old, PyTablesNode):
            self.backend.rename_node(old.name, new_name)
        elif isinstance(old, str):
            self.backend.rename_node(old, new_name)
        raise NotImplementedError()

    def remove_node(self, *args):
        """ This method expects one argument (node) or two arguments (where, node) """
        if len(args) == 1:
            if isinstance(args[0], PyTablesNode):
                node = args[0]
                self.backend.remove_node(node.name)
            elif isinstance(args[0], str):
                name = args[0]
                self.backend.remove_node(name)
            else:
                raise NotImplementedError()
        elif len(args) == 2:
            where, name = args
            where.remove_node(name)
        else:
            raise ValueError('This method expects one or two arguments')


class PyTablesGroup(HasChildren, PyTablesNode):
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
            dtype = dtype_from(description)
            obj = np.empty(shape=(0,), dtype=dtype)
        elif obj is not None and description is not None:
            dtype = dtype_from(description)
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


class PyTablesFile(HasChildren, PyTablesNode):
    @property
    def root(self):
        return PyTablesGroup(backend=self.backend['/'])

    def create_array(self, where, *args, **kwargs):
        return where.create_array(*args, **kwargs)

    def create_group(self, where, *args, **kwargs):
        return where.create_group(*args, **kwargs)

    def create_table(self, where, name, desc, *args, **kwargs):
        return where.create_table(name, desc, *args, **kwargs)
