import h5py
import os.path
from . import abc
from .exceptions import NoSuchNodeError


def set_complex_names():
    cfg = h5py.get_config()
    cfg.complex_names = ('real', 'imag')

set_complex_names()


class HasChildren:
    def __getitem__(self, k):
        ret = super().__getitem__(k)
        if isinstance(ret, h5py.Group):
            return Group(ret.id)
        elif isinstance(ret, h5py.Dataset):
            return Dataset(ret.id)
        raise NotImplementedError()

    def remove_node(self, name):
        try:
            self.__delitem__(name)
        except KeyError:
            raise NoSuchNodeError


class Attributes(h5py.AttributeManager, abc.Attributes):
    def __getattr__(self, item):
        return self.__getitem__(item)


class Resource(HasChildren, h5py.File, abc.Resource):
    def __init__(self, name, **kwargs):
        self._name = name
        self._kwargs = kwargs

    def open(self, **kwargs):
        if kwargs:
            self._kwargs.update(kwargs)
        super().__init__(self._name, **self._kwargs)
        return self

    @property
    def params(self):
        return dict(self._kwargs)


class Group(HasChildren, h5py.Group, abc.Group):
    @property
    def name(self):
        return os.path.basename(super().name)

    @property
    def parent(self):
        return Group(super().parent.id)

    def flush(self):
        self.file.flush()

    def open(self):
        ...

    def close(self):
        if self.id.valid:
            self.id.close()

    def create_group(self, name, **kwargs):
        ret = super().create_group(name, **kwargs)
        return Group(ret.id)

    def create_dataset(self, name, *, chunk_shape=None, **kwargs):
        kwargs['chunks'] = chunk_shape
        ret = super().create_dataset(name, **kwargs)
        return Dataset(ret.id)

    def rename_node(self, old_name, new_name):
        self.move(old_name, new_name)


class Dataset(h5py.Dataset, abc.Dataset):
    @property
    def name(self):
        return os.path.basename(super().name)

    def __delitem__(self, k):
        if isinstance(k, slice):
            n = 0
            for start, stop in abc.anti_slice(k, len(self)):
                asl = stop - start
                self[n:n+asl] = self[start:stop]
                n += asl
            self.resize((n,))
        else:
            raise KeyError('cannot remove key of type: {0}'.format(type(k)))

    @property
    def chunk_shape(self):
        return self.chunks

    @property
    def params(self):
        return {}

    @property
    def parent(self):
        return Group(super().parent.id)

    def flush(self):
        super().flush()

    @property
    def size_on_disk(self):
        return self.id.get_storage_size()
