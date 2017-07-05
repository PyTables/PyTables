from .mixins import HasBackend


class Attributes(HasBackend):

    def __getitem__(self, item):
        return self.backend.__getitem__(item)

    def __setitem__(self, item, value):
        return self.backend.__setitem__(item, value)

    def __getattr__(self, attr):
        return self.__getitem__(attr)

    def __setattr__(self, attr, value):
        return self.__setitem__(attr, value)

    def keys(self):
        return self.backend.keys()

    def items(self):
        return self.backend.items()

    def values(self):
        return self.backend.values()
