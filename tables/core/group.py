import sys
from .node import Node
from .table import Table
from .array import Array
from .carray import CArray
from .earray import EArray
from .vlarray import VLArray
from .leaf import Leaf
from .. import abc
from ..atom import Atom
from .. import Description
from .. import IsDescription
from ..flavor import flavor_of, array_as_internal
from ..utils import np_byteorders, byteorders, correct_byteorder
from .. import lrucacheextension
from ..filters import Filters
from ..exceptions import (PerformanceWarning, ClosedFileError,
                          ClosedNodeError, NoSuchNodeError)
from ..path import join_path
from ..registry import get_class_by_name
import weakref
import warnings
import numpy as np
from h5py import special_dtype
import six

def _checkfilters(filters):
    if not (filters is None or
            isinstance(filters, Filters)):
        raise TypeError("filter parameter has to be None or a Filter "
                        "instance and the passed type is: '%s'" %
                        type(filters))

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
            return Group(backend=value, parent=self)
        elif isinstance(value, abc.Dataset):
            try:
                class_str = value.attrs['CLASS']
            except KeyError:
                class_str = value._infer_class()
            if class_str == 'TABLE':
                return Table(backend=value, parent=self)
            elif class_str == 'ARRAY':
                return Array(backend=value, parent=self)
            elif class_str == 'CARRAY':
                return CArray(backend=value, parent=self)
            elif class_str == 'EARRAY':
                return EArray(backend=value, parent=self)
            elif class_str == 'VLARRAY':
                return VLArray(backend=value, parent=self)

        raise NotImplementedError()

    def __getattr__(self, attr):
        return self.__getitem__(attr)

    def rename_node(self, old, new_name):
        if isinstance(old, Node):
            self.backend.rename_node(old.name, new_name)
        elif isinstance(old, str):
            self.backend.rename_node(old, new_name)
        else:
            raise TypeError(
                "Expecting either the name of the node to rename or "
                "the node itself")

    def remove_node(self, *args):
        """ This method expects one argument (node) or two arguments (where, node)
        """
        if len(args) == 1:
            if isinstance(args[0], Node):
                node = args[0]
                self.backend.remove_node(node.name)
            elif isinstance(args[0], str):
                name = args[0]
                self.backend.remove_node(name)
            else:
                raise TypeError("Expecting either the name of the node "
                                "to rename or the node itself when called "
                                "with one argument")
        elif len(args) == 2:
            where, name = args
            where.remove_node(name)
        else:
            raise ValueError('This method expects one or two arguments')

    def set_node_attr(self, where, attrname, attrvalue, name=None):
        n = self.get_node(where, name=name)
        n.attrs[attrname] = attrvalue

    def get_node_attr(self, where, attrname, attrvalue, name=None):
        n = self.get_node(where, name=name)
        return n.attrs[attrname]

    def get_node(self, where, name=None, classname=None):
        if isinstance(where, Node):
            node = where
        else:
            node = self[where]
        if name is not None:
            node = node[name]
        return node


class Group(HasChildren, Node):
    @property
    def filters(self):
        # TODO properly de-serialize
        ret = self.backend.attrs.get('FILTERS', None)
        if ret is None:
            return self.parent.filters

    @filters.setter
    def filters(self, filters):
        # TODO how we persist this? JSON?
        self.backend.attrs['FILTERS'] = filters

    def create_array(self, name, obj=None, title='',
                     byteorder=None, atom=None, shape=None,
                     **kwargs):
        byteorder = correct_byteorder(type(obj), byteorder)
        if byteorder is None:
            _byteorder = np_byteorders['irrelevant']
        else:
            _byteorder = np_byteorders[byteorder]

        if obj is None:
            if atom is None or shape is None:
                raise TypeError('if the obj parameter is not specified '
                                '(or None) then both the atom and shape '
                                'parametes should be provided.')
            else:
                # Making strides=(0,...) below is a trick to create the
                # array fast and without memory consumption
                dflt = np.zeros((), dtype=atom.dtype)
                obj = np.ndarray(shape, dtype=atom.dtype, buffer=dflt,
                                 strides=(0,) * len(shape))
        else:
            flavor = flavor_of(obj)
            # Use a temporary object because converting obj at this stage
            # breaks some test. This fix performs a double,
            # potentially expensive, conversion of the obj parameter.
            _obj = array_as_internal(obj, flavor)
            if shape is not None and shape != _obj.shape:
                raise TypeError('the shape parameter do not match obj.shape')

            if atom is not None and atom.dtype != _obj.dtype:
                raise TypeError('the atom parameter is not consistent with '
                                'the data type of the obj parameter')

        dtype = None
        if hasattr(obj, 'dtype'):
            dtype = obj.dtype
            if _byteorder != '|' and obj.dtype.byteorder != '|':
                if byteorders[_byteorder] != byteorders[obj.dtype.byteorder]:
                    obj = obj.byteswap()
                    obj.dtype = obj.dtype.newbyteorder()
                    dtype = obj.dtype

        dataset = self.backend.create_dataset(name, data=obj, dtype=dtype,
                                              ** kwargs)

        return Array(backend=dataset, parent=self, title=title, _atom=atom, new=True)

    def create_carray(self, name, atom=None, shape=None, title="",
                      filters=None, chunkshape=None,
                      byteorder=None, obj=None, expectedrows=None,
                      **kwargs):
        fillvalue = None
        dtype = None
        extdim = 0
        if obj is not None:
            if hasattr(obj, 'chunkshape') and chunkshape is None:
                chunkshape = obj.chunkshape
            flavor = flavor_of(obj)
            obj = array_as_internal(obj, flavor)
            if hasattr(obj, 'dtype'):
                dtype = obj.dtype

            if expectedrows is None:
                if shape is not None and shape != obj.shape:
                    raise TypeError('the shape parameter do not match obj.shape')
                else:
                    shape = obj.shape
            else:  # EArray
                earray_shape = (0,) + obj.shape[1:]
                if shape is not None and shape != earray_shape:
                    raise TypeError('the shape parameter is not compatible '
                                    'with obj.shape.')
                shape = obj.shape

            if atom is not None and atom.dtype != obj.dtype:
                raise TypeError('the atom parameter is not consistent with '
                                'the data type of the obj parameter')
        else:
            if atom is None or shape is None:
                raise TypeError('if the obj parameter is not specified '
                                '(or None) then both the atom and shape '
                                'parametes should be provided.')
            else:
                if len(atom.shape) > 0:
                    aux = list(shape)
                    for i in range(len(atom.shape)):
                        aux.append(atom.shape[i])
                    shape = tuple(aux)
                dtype = atom.dtype.base
                fillvalue = atom.dflt
                atom = Atom.from_dtype(dtype, dflt=fillvalue)

        if shape is not None and 0 in shape:
            extdim = shape.index(0)


        _checkfilters(filters)
        compression = None
        compression_opts = None
        shuffle = None
        fletcher32 = None
        if filters is not None:
            compression = filters.get_h5py_compression
            compression_opts = filters.get_h5py_compression_opts
            shuffle = filters.get_h5py_shuffle
            fletcher32 = filters.fletcher32

        byteorder = correct_byteorder(type(obj), byteorder)
        if byteorder is None:
            _byteorder = np_byteorders['irrelevant']
        else:
            _byteorder = np_byteorders[byteorder]

        if _byteorder != '|' and dtype.byteorder != '|':
            if byteorders[_byteorder] != byteorders[dtype.byteorder]:
                if obj is not None:
                    obj = obj.byteswap()
                    obj.dtype = obj.dtype.newbyteorder()
                    dtype = obj.dtype
                else:
                    dtype = dtype.newbyteorder()
        if chunkshape is None:
            chunkshape = True
            maxshape = shape
        else:
            maxshape = [shape[i] if shape[i] is None or shape[i] >= chunkshape[i]
                        else chunkshape[i]
                        for i in range(len(shape))]
        # EArray
        if expectedrows is not None:
            aux = list(shape)
            aux[extdim] = None
            maxshape = tuple(aux)
        dataset = self.backend.create_dataset(name, data=obj, dtype=dtype, shape=shape,
                                              compression=compression,
                                              compression_opts=compression_opts,
                                              shuffle=shuffle,
                                              fletcher32=fletcher32,
                                              chunks=chunkshape, maxshape=maxshape,
                                              fillvalue=fillvalue, **kwargs)
        if expectedrows is None:
            return CArray(filters=filters, backend=dataset, parent=self, title=title, atom=atom, new=True)
        else:
            return EArray(filters=filters, expectedrows=expectedrows, backend=dataset, parent=self,
                          title=title, atom=atom, new=True)

    def create_earray(self, name, atom=None, shape=None, title="",
                      filters=None, expectedrows=1000, chunkshape=None,
                      byteorder=None, obj=None,
                      **kwargs):
        return self.create_carray(name, atom, shape, title, filters,
                                  chunkshape, byteorder, obj,
                                  expectedrows, **kwargs)

    def create_vlarray(self, name, atom=None, title="", filters=None,
                       expectedrows=1000, chunkshape=None, byteorder=None,
                       obj=None, **kwargs):

        if obj is not None:
            flavor = flavor_of(obj)
            obj = array_as_internal(obj, flavor)

            if atom is not None and atom.dtype != obj.dtype:
                raise TypeError('the atom parameter is not consistent with '
                                'the data type of the obj parameter')
            if atom is None:
                atom = Atom.from_dtype(obj.dtype)
        elif atom is None:
            raise ValueError('atom parameter cannot be None')

        if hasattr(atom, 'dtype'):
            vlen = atom.dtype
        byteorder = correct_byteorder(type(obj), byteorder)
        if byteorder is None:
            _byteorder = np_byteorders['irrelevant']
        else:
            _byteorder = np_byteorders[byteorder]

        if _byteorder != '|' and vlen.byteorder != '|':
            if byteorders[_byteorder] != byteorders[vlen.byteorder]:
                vlen = vlen.newbyteorder()
                if obj is not None:
                    obj = obj.byteswap()
                    obj.dtype = obj.dtype.newbyteorder()

        if not hasattr(atom, 'size'):
            dtype = special_dtype(vlen=bytes)
        else:
            dtype = special_dtype(vlen=vlen)

        _checkfilters(filters)
        compression = None
        compression_opts = None
        shuffle = None
        fletcher32 = None
        if filters is not None:
            compression = filters.get_h5py_compression
            compression_opts = filters.get_h5py_compression_opts
            shuffle = filters.get_h5py_shuffle
            fletcher32 = filters.fletcher32

        dataset = self.backend.create_dataset(name, data=obj, dtype=dtype, shape=(0,),
                                              compression=compression,
                                              compression_opts=compression_opts,
                                              shuffle=shuffle,
                                              fletcher32=fletcher32,
                                              chunks=True, maxshape=(None,),
                                              **kwargs)
        ptobj = VLArray(backend=dataset, parent=self, atom=atom, title=title, filters=filters,
                        expectedrows=expectedrows, new=True)

        if obj is not None:
            ptobj.append(obj)

        return ptobj


    def create_group(self, name, title=''):
        g = Group(backend=self.backend.create_group(name), parent=self)
        g.attrs['TITLE'] = title
        return g

    def create_table(self, name, description=None, title='',
                     byteorder=None,
                     filters=None, expectedrows=10000,
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
        return Table(backend=dataset, parent=self)

format_version = "2.1"  # Numeric and numarray flavors are gone.

class File(HasChildren, Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO (re) make this configurable
        # node_cache_slots = params['NODE_CACHE_SLOTS']
        # TODO only show Filters the inputs it wants
        self._filters = Filters(**self.backend.params)
        # Bootstrap the _file attribute for nodes
        self._file = self


    def close(self):
        # Flush the nodes prior to close
        super().close()

    def flush(self):
        self.backend.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def reopen(self, **kwargs):
        # Flush the nodes prior to close
        self.backend.flush()
        self.backend.close()
        self.backend.open(**kwargs)

    def __contains__(self, item):
        return True if self[item] else False

    @property
    def root(self):
        return self['/']

    def create_array(self, where, *args, createparents=False, **kwargs):
        if not hasattr(where, 'create_array'):
            where = self._get_or_create_path(where, createparents)
        return where.create_array(*args, **kwargs)

    def create_carray(self, where, name, atom=None, shape=None, title="",
                      filters=None, chunkshape=None,
                      byteorder=None, createparents=False, obj=None, **kwargs):
        if not hasattr(where, 'create_carray'):
            where = self._get_or_create_path(where, createparents)
        return where.create_carray(name, atom, shape, title,
                      filters, chunkshape,
                      byteorder, obj, **kwargs)

    def create_earray(self, where, name, atom=None, shape=None, title="",
                      filters=None, expectedrows=1000, chunkshape=None,
                      byteorder=None, createparents=False, obj=None, **kwargs):
        if not hasattr(where, 'create_earray'):
            where = self._get_or_create_path(where, createparents)
        return where.create_earray(name, atom, shape, title,
                      filters, expectedrows, chunkshape,
                      byteorder, obj, **kwargs)

    def create_vlarray(self, where, name, atom=None, title="",
                       filters=None, expectedrows=None,
                       chunkshape=None, byteorder=None,
                       createparents=False, obj=None, **kwargs):
        if not hasattr(where, 'create_vlarray'):
            where = self._get_or_create_path(where, createparents)
        return where.create_vlarray(name, atom, title, filters,
                                    expectedrows, chunkshape,
                                    byteorder, obj, **kwargs)

    def create_group(self, where, *args, createparents=False, **kwargs):
        if not hasattr(where, 'create_group'):
            where = self._get_or_create_path(where, createparents)
        return where.create_group(*args, **kwargs)

    def create_table(self, where, name, desc, *args,
                     createparents=False, **kwargs):
        if not hasattr(where, 'create_table'):
            where = self._get_or_create_path(where, createparents)
        return where.create_table(name, desc, *args, **kwargs)

    def get_node(self, where, name=None, classname=None):
        """Get the node under where with the given name.

        Parameters
        ----------
        where : str or Node
            This can be a path string leading to a node or a Node instance (see
            :ref:`NodeClassDescr`). If no name is specified, that node is
            returned.

            .. note::

                If where is a Node instance from a different file than the one
                on which this function is called, the returned node will also
                be from that other file.

        name : str, optional
            If a name is specified, this must be a string with the name of
            a node under where.  In this case the where argument can only
            lead to a Group (see :ref:`GroupClassDescr`) instance (else a
            TypeError is raised). The node called name under the group
            where is returned.
        classname : str, optional
            If the classname argument is specified, it must be the name of
            a class derived from Node (e.g. Table). If the node is found but it
            is not an instance of that class, a NoSuchNodeError is also raised.

        If the node to be returned does not exist, a NoSuchNodeError is
        raised. Please note that hidden nodes are also considered.

        """

        self._check_open()

        if isinstance(where, Group):
            where._g_check_open()

            node = where[name]
        elif isinstance(where, (six.string_types, np.str_)):
            if not where.startswith('/'):
                raise NameError("``where`` must start with a slash ('/')")

            basepath = where
            nodepath = join_path(basepath, name or '') or '/'
            node = self.root[nodepath]
        else:
            raise TypeError(
                "``where`` must be a string or a node: %r" % (where,))

        # Finally, check whether the desired node is an instance
        # of the expected class.
        if classname is not None:
            class_ = get_class_by_name(classname)
            if not isinstance(node, class_):
                npathname = node._v_pathname
                nclassname = node.__class__.__name__
                # This error message is right since it can never be shown
                # for ``classname in [None, 'Node']``.
                raise NoSuchNodeError(
                    "could not find a ``%s`` node at ``%s``; "
                    "instead, a ``%s`` node has been found there"
                    % (classname, npathname, nclassname))

        return node

    def _check_writable(self):
        return self.backend._check_writable()


    def _get_or_create_path(self, path, create):
        if create:
            return self._create_path(path)
        else:
            return self.get_node(path)

    def _create_path(self, path):
        if not hasattr(path, 'split'):
            raise TypeError("when creating parents, parent must be a path")

        if path == '/':
            return self.root

        parent, create_group = self.root, self.create_group
        for pcomp in path.split('/')[1:]:
            try:
                child = parent[pcomp]
            except NoSuchNodeError:
                child = create_group(parent, name=pcomp)
            parent = child
        return parent

    def _check_open(self):
        """Check the state of the file.

        If the file is closed, a `ClosedFileError` is raised.

        """

        if not self._v_isopen:
            raise ClosedFileError("the file object is closed")


# A dumb class that doesn't keep nothing at all
class _NoCache(object):
    def __len__(self):
        return 0

    def __contains__(self, key):
        return False

    def __iter__(self):
        return iter([])

    def __setitem__(self, key, value):
        pass

    __marker = object()

    def pop(self, key, d=__marker):
        if d is not self.__marker:
            return d
        raise KeyError(key)


class _DictCache(dict):
    def __init__(self, nslots):
        if nslots < 1:
            raise ValueError("Invalid number of slots: %d" % nslots)
        self.nslots = nslots
        super(_DictCache, self).__init__()

    def __setitem__(self, key, value):
        # Check if we are running out of space
        if len(self) > self.nslots:
            warnings.warn(
                "the dictionary of node cache is exceeding the recommended "
                "maximum number (%d); be ready to see PyTables asking for "
                "*lots* of memory and possibly slow I/O." % (
                    self.nslots), PerformanceWarning)
        super(_DictCache, self).__setitem__(key, value)


class NodeManager:
    def __init__(self, nslots=64, node_factory=None):
        super().__init__()

        self.registry = weakref.WeakValueDictionary()

        if nslots > 0:
            cache = lrucacheextension.NodeCache(nslots)
        elif nslots == 0:
            cache = _NoCache()
        else:
            # nslots < 0
            cache = _DictCache(-nslots)

        self.cache = cache

        # node_factory(node_path)
        self.node_factory = node_factory

    def register_node(self, node, key):
        if key is None:
            key = node._v_pathname

        if key in self.registry:
            if not self.registry[key]._v_isopen:
                del self.registry[key]
            elif self.registry[key] is not node:
                raise RuntimeError('trying to register a node with an '
                                   'existing key: ``%s``' % key)
        else:
            self.registry[key] = node

    def cache_node(self, node, key=None):
        if key is None:
            key = node._v_pathname

        self.register_node(node, key)
        if key in self.cache:
            oldnode = self.cache.pop(key)
            if oldnode is not node and oldnode._v_isopen:
                raise RuntimeError('trying to cache a node with an '
                                   'existing key: ``%s``' % key)

        self.cache[key] = node

    def get_node(self, key):
        node = self.cache.pop(key, None)
        if node is not None:
            if node._v_isopen:
                self.cache_node(node, key)
                return node
            else:
                # this should not happen
                warnings.warn("a closed node found in the cache: ``%s``" % key)

        if key in self.registry:
            node = self.registry[key]
            if node is None:
                # this should not happen since WeakValueDictionary drops all
                # dead weakrefs
                warnings.warn("None is stored in the registry for key: "
                              "``%s``" % key)
            elif node._v_isopen:
                self.cache_node(node, key)
                return node
            else:
                # this should not happen
                warnings.warn("a closed node found in the registry: "
                              "``%s``" % key)
                del self.registry[key]
                node = None

        if self.node_factory:
            node = self.node_factory(key)
            self.cache_node(node, key)

        return node

    def rename_node(self, oldkey, newkey):
        for cache in (self.cache, self.registry):
            if oldkey in cache:
                node = cache.pop(oldkey)
                cache[newkey] = node

    def drop_from_cache(self, nodepath):
        '''Remove the node from cache'''

        # Remove the node from the cache.
        self.cache.pop(nodepath, None)

    def drop_node(self, node, check_unregistered=True):
        """Drop the `node`.

        Remove the node from the cache and, if it has no more references,
        close it.

        """

        # Remove all references to the node.
        nodepath = node._v_pathname

        self.drop_from_cache(nodepath)

        if nodepath in self.registry:
            if not node._v_isopen:
                del self.registry[nodepath]
        elif check_unregistered:
            # If the node is not in the registry (this should never happen)
            # we close it forcibly since it is not ensured that the __del__
            # method is called for object that are still alive when the
            # interpreter is shut down
            if node._v_isopen:
                warnings.warn("dropping a node that is not in the registry: "
                              "``%s``" % nodepath)

                node._g_pre_kill_hook()
                node._f_close()

    def flush_nodes(self):
        # Only iter on the nodes in the registry since nodes in the cache
        # should always have an entry in the registry
        closed_keys = []
        for path, node in list(self.registry.items()):
            if not node._v_isopen:
                closed_keys.append(path)
            elif '/_i_' not in path:  # Indexes are not necessary to be flushed
                if isinstance(node, Leaf):
                    node.flush()

        for path in closed_keys:
            # self.cache.pop(path, None)
            if path in self.cache:
                warnings.warn("closed node the cache: ``%s``" % path)
                self.cache.pop(path, None)
            self.registry.pop(path)

    @staticmethod
    def _close_nodes(nodepaths, get_node):
        for nodepath in nodepaths:
            try:
                node = get_node(nodepath)
            except KeyError:
                pass
            else:
                if not node._v_isopen or node._v__deleting:
                    continue

                try:
                    # Avoid descendent nodes to also iterate over
                    # their descendents, which are already to be
                    # closed by this loop.
                    if hasattr(node, '_f_get_child'):
                        node._g_close()
                    else:
                        node._f_close()
                    del node
                except ClosedNodeError:
                    #import traceback
                    #type_, value, tb = sys.exc_info()
                    # exception_dump = ''.join(
                    #    traceback.format_exception(type_, value, tb))
                    # warnings.warn(
                    #    "A '%s' exception occurred trying to close a node "
                    #    "that was supposed to be open.\n"
                    #    "%s" % (type_.__name__, exception_dump))
                    pass

    def close_subtree(self, prefix='/'):
        if not prefix.endswith('/'):
            prefix = prefix + '/'

        cache = self.cache
        registry = self.registry

        # Ensure tables are closed before their indices
        paths = [
            path for path in cache
            if path.startswith(prefix) and '/_i_' not in path
        ]
        self._close_nodes(paths, cache.pop)

        # Close everything else (i.e. indices)
        paths = [path for path in cache if path.startswith(prefix)]
        self._close_nodes(paths, cache.pop)

        # Ensure tables are closed before their indices
        paths = [
            path for path in registry
            if path.startswith(prefix) and '/_i_' not in path
        ]
        self._close_nodes(paths, registry.pop)

        # Close everything else (i.e. indices)
        paths = [path for path in registry if path.startswith(prefix)]
        self._close_nodes(paths, registry.pop)

    def shutdown(self):
        registry = self.registry
        cache = self.cache

        # self.close_subtree('/')

        keys = list(cache)  # copy
        for key in keys:
            node = cache.pop(key)
            if node._v_isopen:
                registry.pop(node._v_pathname, None)
                node._f_close()

        while registry:
            key, node = registry.popitem()
            if node._v_isopen:
                node._f_close()
