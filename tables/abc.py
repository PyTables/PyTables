"""PyTables Abstract Base Classes."""
from abc import abstractmethod, abstractproperty
from collections.abc import MutableSequence, MutableMapping


class Dataset(MutableSequence):

    @abstractproperty
    def dtype(self):
        ...

    @abstractproperty
    def attrs(self):
        ...

    @abstractproperty
    def shape(self):
        ...

    @abstractproperty
    def params(self):
        ...

    @property
    def chunk_shape(self):
        return None

    @property
    def parent(self):
        return None

    @abstractmethod
    def iter_chunks(self):
        ...


class Group(MutableMapping):

    @abstractproperty
    def attrs(self):
        ...

    @abstractmethod
    def create_dataset(self, **kwargs):
        ...

    @abstractmethod
    def create_group(self, **kwargs):
        ...

    @abstractmethod
    def open(self):
        ...

    @abstractmethod
    def close(self):
        ...

    @property
    def parent(self):
        return None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()
        return


class Attributes(MutableMapping):
    ...
