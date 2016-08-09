import h5py
from tables import abc


class PTShim:
    def __getitem__(self, k):
        ret = super().__getitem__(k)
        if isinstance(ret, h5py.Group):
            return Group(ret.id)
        elif isinstance(ret, h5py.Dataset):
            return Dataset(ret.id)
        return ret

    def open(self):
        ...

    def close(self):
        ...

    @property
    def parent(self):
        return Group(super().parent.id)

    def create_dataset(self, name, *, chunk_shape=None, **kwargs):
        kwargs['chunks'] = chunk_shape
        return super().create_dataset(name, **kwargs)


class Group(PTShim, h5py.Group, abc.Group):
    ...


class Dataset(h5py.Dataset, abc.Dataset):
    def __delitem__(self, k):
        raise NotImplementedError()

    @property
    def chunk_shape(self):
        return self.chunks

    @property
    def params(self):
        return {}

    @property
    def parent(self):
        return Group(super().parent.id)


class File(h5py.File, Group):
    ...


def open(*args, **kwargs):
    f = h5py.File(*args, **kwargs)
    return Group(f['/'].id)
