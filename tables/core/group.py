from .node import Node
from .table import Table
from .array import Array
from .leaf import Leaf
from tables import abc
from tables import Description
from tables import IsDescription
from .. import lrucacheextension
from ..filters import Filters
from ..exceptions import PerformanceWarning, ClosedFileError, ClosedNodeError
import weakref
import warnings
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
        # Try cache first
        nmanager = self._file._node_manager
        node = nmanager.get_node(item)
        if node:
            return node
        # No luck, so use the backend to lookup the item
        value = self.backend[item]
        if isinstance(value, abc.Group):
            return Group(backend=value, parent=self)
        elif isinstance(value, abc.Dataset):
            if value.attrs['CLASS'] == 'TABLE':
                return Table(backend=value, parent=self)
            elif value.attrs['CLASS'] == 'ARRAY':
                return Array(backend=value, parent=self)

        raise NotImplementedError()

    def __getattr__(self, attr):
        return self.__getitem__(attr)

    def rename_node(self, old, new_name):
        if isinstance(old, Node):
            self.backend.rename_node(old.name, new_name)
        elif isinstance(old, str):
            self.backend.rename_node(old, new_name)
        else:
            raise TypeError("Expecting either the name of the node to rename or the node itself")

    def remove_node(self, *args):
        """ This method expects one argument (node) or two arguments (where, node) """
        if len(args) == 1:
            if isinstance(args[0], Node):
                node = args[0]
                self.backend.remove_node(node.name)
            elif isinstance(args[0], str):
                name = args[0]
                self.backend.remove_node(name)
            else:
                raise TypeError("Expecting either the name of the node "
                        "to rename or the node itself when called with "
                        "one argument")
        elif len(args) == 2:
            where, name = args
            where.remove_node(name)
        else:
            raise ValueError('This method expects one or two arguments')


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

    def create_array(self, name, obj, title='', byte_order='I', **kwargs):
        obj = np.asarray(obj)
        dtype = obj.dtype.newbyteorder(byte_order)
        dataset = self.backend.create_dataset(name, data=obj,
                                              dtype=dtype,
                                              **kwargs)
        dataset.attrs['TITLE'] = title
        dataset.attrs['CLASS'] = 'ARRAY'
        return Array(backend=dataset, parent=self)

    def create_group(self, name, title=''):
        g = Group(backend=self.backend.create_group(name), parent=self)
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
        return Table(backend=dataset, parent=self)


class File(HasChildren, Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO (re) make this configurable
        # node_cache_slots = params['NODE_CACHE_SLOTS']
        node_cache_slots = 10
        self._node_manager = NodeManager(nslots=node_cache_slots)
        # TODO only show Filters the inputs it wants
        self._filters = Filters(**self.backend.params)
        # Bootstrap the _file attribute for nodes
        self._file = self

    def close(self):
        # Flush the nodes prior to close
        self._node_manager.flush_nodes()
        super().close()

    def __enter__(self):
        self.open()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def reopen(self, **kwargs):
        # Flush the nodes prior to close
        self._node_manager.flush_nodes()
        self.backend.close()
        self.backend.open(**kwargs)

    def __contains__(self, item):
        return True if self[item] else False

    @property
    def root(self):
        return self['/']

    def create_array(self, where, *args, **kwargs):
        return where.create_array(*args, **kwargs)

    def create_group(self, where, *args, **kwargs):
        return where.create_group(*args, **kwargs)

    def create_table(self, where, name, desc, *args, **kwargs):
        return where.create_table(name, desc, *args, **kwargs)

    def get_node(self, where):
        return self.root[where]


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
                    #exception_dump = ''.join(
                    #    traceback.format_exception(type_, value, tb))
                    #warnings.warn(
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

        #self.close_subtree('/')

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
