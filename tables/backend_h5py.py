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

    def _infer_class(self):
        class_str = 'UNSUPPORTED' # default value
        class_id = self.id.get_type().get_class()
        layout = self.id.get_create_plist().get_layout()
        # Check if this a dataset of supported classtype for ARRAY
        if (class_id == h5py.h5t.INTEGER or
            class_id == h5py.h5t.FLOAT or
            class_id == h5py.h5t.BITFIELD or
            class_id == h5py.h5t.TIME or
            class_id == h5py.h5t.ENUM or
            class_id == h5py.h5t.STRING or
            class_id == h5py.h5t.ARRAY or
            class_id == h5py.h5t.REFERENCE):
            if layout == h5py.h5d.CHUNKED:
                class_str = 'CARRAY'
                maxdims = self.maxshape
                for i in range(len(self.maxshape)):
                    if maxdims[i] == -1:
                        class_str = "EARRAY"
            else:
                class_str = 'ARRAY'
        elif class_id == h5py.h5t.COMPOUND:
            # check whether the type is complex or not
            is_complex = False
            type_id = self.id.get_type()
            nfields = type_id.get_nmembers()
            if nfields == 2:
                field_name1 = type_id.get_member_name(0)
                field_name2 = type_id.get_member_name(1)
                # The pair ("r", "i") is for PyTables. ("real", "imag") for Octave.
                if (field_name1 == "real" and field_name2 == "imag" or
                    field_name1 == "r" and field_name2 == "i"):
                    is_complex = True
            if layout == h5py.h5d.CHUNKED:
                if is_complex:
                    class_str = "CARRAY"
                else:
                    class_str = "TABLE"
            else:  # Not chunked case
                # Octave saves complex arrays as non-chunked tables
                # with two fields: "real" and "imag"
                # Francesc Alted 2005-04-29
                # Get number of records
                if is_complex:
                    class_str = "ARRAY"  # It is probably an Octave complex array
                else:
                    # Added to support non-chunked tables
                    class_str = "TABLE"  # A test for supporting non-growable tables
        elif class_id == h5py.h5t.VLEN:
            if layout == h5py.h5d.CHUNKED:
                class_str = "VLARRAY"
        # Fallback
        return class_str




