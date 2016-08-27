from .attributes import Attributes
from .mixins import HasTitle, HasBackend


class Node(HasTitle, HasBackend):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._filters = None
        self._isopen = True
        if self._parent is not None:
            # Set the _file attr for nodes that are not File
            self._file = self._parent._file
            nmanager = self._file._node_manager
            node = nmanager.get_node(self._v_pathname)
            if not node:
                # Put this node in cache
                nmanager.cache_node(self, self._v_pathname)

    @property
    def name(self):
        return self.backend.name

    @property
    def _v_pathname(self):
        if self._parent:
            if self._parent._v_pathname != '/':
                return self._parent._v_pathname + '/' + self.name
            else:
                return '/' + self.name
        else:
            return '/'

    @property
    def attrs(self):
        return Attributes(backend=self.backend.attrs, parent=self)

    # for backward compatibility
    _v_attrs = attrs

    def open(self):
        self._isopen = True
        return self.backend.open()

    def close(self):
        self._isopen = False
        return self.backend.close()

    @property
    def _v_isopen(self):
        return self._isopen

    @property
    def filters(self):
        if self._filters is not None:
            return self._filters
        else:
            return self.parent.filters

    @filters.setter
    def filters(self, value):
        self._filters = value

    @property
    def _v_parent(self):
        return self._parent

    @property
    def _v_file(self):
        return self._file
