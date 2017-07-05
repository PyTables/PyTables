class HasBackend:
    @property
    def root(self):
        return self._file

    @property
    def backend(self):
        return self._backend

    @property
    def parent(self):
        return self._parent

    def __init__(self, *, backend, parent, **kwargs):
        super().__init__(**kwargs)
        self._backend = backend
        self._parent = parent


class HasTitle:
    @property
    def title(self):
        return self.backend.attrs.get('TITLE', None)

    @title.setter
    def title(self, title):
        self.backend.attrs['TITLE'] = title
