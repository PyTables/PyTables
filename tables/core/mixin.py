

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


class Attributes(HasBackend):

    def __getitem__(self, item):
        return self.backend.__getitem__(item)

    def __setitem__(self, item, value):
        return self.backend.__setitem__(item, value)

    def __getattr__(self, attr):
        return self.__getitem__(attr)
