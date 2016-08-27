from .mixins import HasBackend

class Attributes(HasBackend):

    def __getitem__(self, item):
        return self.backend.__getitem__(item)

    def __setitem__(self, item, value):
        return self.backend.__setitem__(item, value)

    def __getattr__(self, attr):
        return self.__getitem__(attr)

