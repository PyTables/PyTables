:source: http://www.pytables.org/moin/UserDocuments/CustomDataTypes
:revision: 1
:date: 2009-07-14 21:51:07
:author: KennethArnold


================================
Using your own custom data types
================================

You can make your own data types by subclassing Table (or other PyTables types,
such as :class:`tables.Leaf`).
This can be useful for storing a specialized type of data or presenting a
customized API.

Submitted by Kevin R. Thornton.

::

    from __future__ import print_function
    import numpy as np

    import tables
    from tables import File, Table
    from tables.file import _checkfilters

    from tables.parameters import EXPECTED_ROWS_TABLE


    class DerivedFromTable(Table):
        _c_classId = 'DerivedFromTable'

        def __init__(self, parentNode, name, description=None,
                     title="", filters=None,
                     expectedrows=EXPECTED_ROWS_TABLE,
                     chunkshape=None, byteorder=None, _log=True):
            super(DerivedFromTable, self).__init__(parentNode, name,
                                              description=description, title=title,
                                              filters=filters,
                                              expectedrows=expectedrows,
                                              chunkshape=chunkshape, byteorder=byteorder,
                                              _log=_log)

        def read(self, start=None, stop=None, step=None, field=None):
            print("HERE!")
            data = Table.read(self, start=start, stop=stop, step=step,
                              field=field)
            return data


    def createDerivedFromTable(self, where, name, data, title="",
                               filters=None, expectedrows=10000,
                               chunkshape=None, byteorder=None,
                               createparents=False):
        parentNode = self._get_or_create_path(where, createparents)

        _checkfilters(filters)
        return DerivedFromTable(parentNode, name, data,
                                title=title, filters=filters,
                                expectedrows=expectedrows,
                                chunkshape=chunkshape, byteorder=byteorder)


    File.createDerivedFromTable = createDerivedFromTable


    if __name__ == '__main__':
        x = np.array(np.random.rand(100))
        x=np.reshape(x,(50,2))
        x.dtype=[('x',np.float),('y',np.float)]
        h5file = tables.open_file('tester.hdf5', 'w')
        mtab = h5file.createDerivedFromTable(h5file.root, 'random', x)

        h5file.flush()
        print(type(mtab))
        mtab_read = mtab.read()
        h5file.close()
        h5file = tables.open_file('tester.hdf5', 'r')
        mtab = h5file.root.random

        print(type(mtab))
        mtab_read2 = mtab.read()
        print(np.array_equal(mtab_read, mtab_read2))


There is an issue that the DerivedFromTable read function will not be called
when the file is re-opened. The notion that the H5 file contains a derived
object gets lost. The output shows that the read function is only called before
the function is closed:

::
        <class '__main__.DerivedFromTable'>
        HERE!
        <class 'tables.table.Table'>
        True
        Closing remaining open files:tester.hdf5...done


I ran into this because I wanted a custom read that returned a more complex
object implemented in C++. Using pybind11, I'm easily able to write to a
Table via a record array. I was hoping that I could read back in, construct the
correct C++-based type, and return it. The example seems to suggest that this
is not possible.
