import h5py
from tables import abc


class Group(h5py.Group, abc.Group):
    def __getitem__(self, k):
        ret = super().__getitem__(k)
        if isinstance(ret, h5py.Group):
            return Group(ret.id)
        elif isinstance(ret, h5py.Dataset):
            return Dataset(ret.id)
        return ret

    def open():
        ...

    def close():
        ...

    @property
    def parent(self):
        return Group(super().parent.id)


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
