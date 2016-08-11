from .mixins import HasTitle, HasBackend, Attributes


class Node(HasTitle, HasBackend):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._filters = None

    @property
    def name(self):
        return self.backend.name

    @property
    def attrs(self):
        return Attributes(backend=self.backend.attrs, parent=self)

    # for backward compatibility
    _v_attrs = attrs

    def open(self):
        return self.backend.open()

    def close(self):
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
