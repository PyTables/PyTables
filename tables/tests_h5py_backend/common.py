import os
import tables.backend_h5py as tbh
import tables.core as tcore
import tempfile


class TempFileMixin(object):
    open_mode = 'w'
    open_kwargs = {}

    def open_file(self, filename, mode="r", title="", **kwargs):
        backend = tbh.open(filename, mode)
        return tcore.PyTablesFile(backend=backend, **kwargs)

    def setUp(self):
        """Set ``h5file`` and ``h5fname`` instance attributes.

        * ``h5fname``: the name of the temporary HDF5 file.
        * ``h5file``: the writable, empty, temporary HDF5 file.

        """

        super().setUp()
        self.h5fname = tempfile.mktemp(prefix='file', suffix='.h5')
        self.h5file = self.open_file(self.h5fname, self.open_mode)
        self.root = self.h5file.root

    def tearDown(self):
        """Close ``h5file`` and remove ``h5fname``."""

        self.h5file.close()
        self.h5file = None
        os.remove(self.h5fname)   # comment this for debugging purposes only
        super().tearDown()

    def _reopen(self, mode='r', **kwargs):
        """Reopen ``h5file`` in the specified ``mode``.

        Returns a true or false value depending on whether the file was
        reopenend or not.  If not, nothing is changed.

        """

        self.h5file.close()
        self.h5file = self.open_file(self.h5fname, mode, **kwargs)
