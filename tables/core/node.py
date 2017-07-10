from .attributes import Attributes
from .mixins import HasTitle, HasBackend
from ..exceptions import ClosedNodeError


class Node(HasTitle, HasBackend):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._filters = None
        self._isopen = True

        if self.parent is not None:
            # Set the _file attr for nodes that are not File
            self._file = self.parent._file

    @property
    def name(self):
        return self.backend.name

    @property
    def _v_pathname(self):
        if self.parent:
            if self.parent._v_pathname != '/':
                return self.parent._v_pathname + '/' + self.name
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
        return self.parent

    @property
    def _v_file(self):
        return self._file

    @property
    def _v_isopen(self):
        return self._isopen

    def _g_check_open(self):
        """Check that the node is open.

        If the node is closed, a `ClosedNodeError` is raised.

        """

        if not self._v_isopen:
            raise ClosedNodeError("the node object is closed")
        assert self._v_file._v_isopen, "found an open node in a closed file"
