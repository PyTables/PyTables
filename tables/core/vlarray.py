from .earray import EArray
from ..atom import VLStringAtom, VLUnicodeAtom, ObjectAtom
from ..utils import convert_to_np_atom2
from ..exceptions import HDF5ExtError


class VLArray(EArray):
    def __init__(self, expectedrows=None, **kwargs):
        super().__init__(**kwargs)


    def append(self, sequence):
        """Add a sequence of data to the end of the dataset.

        This method appends the objects in the sequence to a *single row* in
        this array. The type and shape of individual objects must be compliant
        with the atoms in the array. In the case of serialized objects and
        variable length strings, the object or string to append is itself the
        sequence.

        """

        self._g_check_open()
        self._v_file._check_writable()

        # Prepare the sequence to convert it into a NumPy object
        atom = self.atom
        if not hasattr(atom, 'size'):  # it is a pseudo-atom
            sequence = atom.toarray(sequence)
            statom = atom.base
        else:
            try:  # fastest check in most cases
                len(sequence)
            except TypeError:
                raise TypeError("argument is not a sequence")
            statom = atom

        if len(sequence) > 0:
            # The sequence needs to be copied to make the operation safe
            # to in-place conversion.
            nparr = convert_to_np_atom2(sequence, statom)
        else:
            nparr = []

        self.backend.resize(self.shape[self.extdim] + 1, axis=self.extdim)
        self[-1] = nparr

    def get_row_size(self, row):
        if row >= self.nrows:
            raise HDF5ExtError(
                "Asking for a range of rows exceeding the available ones!.",
                h5bt=False)
        return self[row].size * self[row].dtype.itemsize
