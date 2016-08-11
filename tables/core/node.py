from .mixins import HasTitle, HasBackend, Attributes


class Node(HasTitle, HasBackend):
    @property
    def name(self):
        return self.backend.name

    @property
    def attrs(self):
        return Attributes(backend=self.backend.attrs)

    # for backward compatibility
    _v_attrs = attrs

    def open(self):
        return self.backend.open()

    def close(self):
        return self.backend.close()
