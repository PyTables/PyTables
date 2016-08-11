from .mixins import HasTitle, HasBackend, PyTablesAttributes


class PyTablesNode(HasTitle, HasBackend):
    @property
    def name(self):
        return self.backend.name

    @property
    def attrs(self):
        return PyTablesAttributes(backend=self.backend.attrs)

    # for backward compatibility
    _v_attrs = attrs

    def open(self):
        return self.backend.open()

    def close(self):
        return self.backend.close()
