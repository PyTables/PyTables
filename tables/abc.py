"""PyTables Abstract Base Classes."""
from abc import abstractmethod, abstractproperty, ABCMeta
from collections.abc import MutableMapping
import itertools
import numpy as np


def all_chunk_selector(x):
    return True


class Dataset(metaclass=ABCMeta):

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
    def __getitem__(self, k):
        ...

    @abstractmethod
    def __delitem__(self, k):
        ...

    @abstractmethod
    def __setitem__(self, k, v):
        ...

    def iter_chunks(self, *, chunk_selector=None):
        if self.chunk_shape is None:
            yield (1, ), np.rec.array(self[:])
            return
        chunk_count = tuple(sz // ck + min(1, sz % ck)
                            for sz, ck in
                            zip(self.shape, self.chunk_shape))
        if chunk_selector is None:
            chunk_selector = all_chunk_selector
        for chunk_id in itertools.product(*(range(cc) for cc in chunk_count)):
            if not chunk_selector(chunk_id):
                continue

            slc = tuple(slice(j*sz, (j+1)*sz)
                        for j, sz in zip(chunk_id, self.chunk_shape))
            yield chunk_id, np.rec.array(self[slc])

    def iter_with_selectors(self, *, chunk_selector, sub_chunk_selector):
        for chunk in self.iter_chunks(chunk_selector=chunk_selector):
            yield from sub_chunk_selector(chunk)


class Table(Dataset):
    ...


class Group(MutableMapping):

    @abstractproperty
    def attrs(self):
        ...

    @abstractmethod
    def create_dataset(self, data, dtype, maxshape, chunk_shape,
                       **kwargs):
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
