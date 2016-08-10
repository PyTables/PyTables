import h5py
from tables import abc


class Group(h5py.Group, abc.Group):
    @property
    def parent(self):
        return Group(super().parent.id)

    @property
    def file(self):
        ret = super().file
        return File(ret.id)

    def flush(self):
        self.file.flush()

    def open(self):
        ...

    def close(self):
        ...

    def __getitem__(self, k):
        ret = super().__getitem__(k)
        if isinstance(ret, h5py.Group):
            return Group(ret.id)
        elif isinstance(ret, h5py.Dataset):
            return Dataset(ret.id)
        raise NotImplementedError()

    def create_group(self, name, **kwargs):
        ret = super().create_group(name, **kwargs)
        return Group(ret.id)

    def create_dataset(self, name, *, chunk_shape=None, **kwargs):
        kwargs['chunks'] = chunk_shape
        ret = super().create_dataset(name, **kwargs)
        return Dataset(ret.id)


class Dataset(h5py.Dataset, abc.Dataset):
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
        self.file.flush()


def open(*args, **kwargs):
    f = h5py.File(*args, **kwargs)
    return Group(f['/'].id)
