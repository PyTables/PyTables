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

Here's one way to do it, taken from
http://sourceforge.net/mailarchive/message.php?msg_id=200805250042.50653.pgmdevlist%40gmail.com

::

    from __future__ import print_function
    import numpy as np
    import numpy.ma as ma

    import tables
    from tables import File, Table
    from tables.file import _checkfilters

    from tables.parameters import EXPECTED_ROWS_TABLE

    class MaskedTable(Table):
        _c_classId = 'MaskedTable'
        def __init__(self, parentNode, name, description=None,
                     title="", filters=None,

                     expectedrows=EXPECTED_ROWS_TABLE,
                     chunkshape=None, byteorder=None, _log=True):
            new = description is None
            if not new:
                maskedarray = description
                description = np.array(zip(maskedarray.filled().flat,

                                       ma.getmaskarray(maskedarray).flat),
                                       dtype=[('_data',maskedarray.dtype),
                                              ('_mask',bool)])
            Table.__init__(self, parentNode, name,
                           description=description, title=title,
                           filters=filters,

                           expectedrows=expectedrows,
                           chunkshape=chunkshape, byteorder=byteorder,
                           _log=_log)
            if not new:
                self.attrs.shape = maskedarray.shape

        def read(self, start=None, stop=None, step=None, field=None):
            data = Table.read(self, start=start, stop=stop, step=step,
                              field=field)
            newshape = self.attrs.shape
            return ma.array(data['_data'],
                            mask=data['_mask']).reshape(newshape)


    def createMaskedTable(self, where, name, maskedarray, title="",
                          filters=None, expectedrows=10000,
                          chunkshape=None, byteorder=None,
                          createparents=False):
        parentNode = self._getOrCreatePath(where, createparents)

        _checkfilters(filters)
        return MaskedTable(parentNode, name, maskedarray,
                           title=title, filters=filters,
                           expectedrows=expectedrows,
                           chunkshape=chunkshape, byteorder=byteorder)


    File.createMaskedTable = createMaskedTable


    if __name__ == '__main__':
        x = ma.array(np.random.rand(100),mask=(np.random.rand(100) > 0.7))
        h5file = tables.openFile('tester.hdf5','w')
        mtab = h5file.createMaskedTable('/','random',x)

        h5file.flush()
        print(type(mtab))
        print(mtab.read())
        h5file.close()
        h5file = tables.openFile('tester.hdf5','r')
        mtab = h5file.root.random

        print(type(mtab))
        print(mtab.read())

