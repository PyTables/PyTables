:source: http://www.pytables.org/moin/UserDocuments/SimpleTable
:revision: 3
:date: 2010-04-20 16:44:41
:author: FrancescAlted


===================================================
SimpleTable: simple wrapper around the Table object
===================================================

Here it is yet another example on how to inherit from the :class:`tables.Table`
object so as to build an easy-to-use Table object.
Thanks to Brent Pedersen for this one (taken from
https://pypi.python.org/pypi/simpletable).

::

    """

    SimpleTable: simple wrapper around pytables hdf5
    ------------------------------------------------------------------------------

    Example Usage::

      >>> from simpletable import SimpleTable
      >>> import tables

      # define the table as a subclass of simple table.
      >>> class ATable(SimpleTable):
      ...     x = tables.Float32Col()
      ...     y = tables.Float32Col()
      ...     name = tables.StringCol(16)

      # instantiate with: args: filename, tablename
      >>> tbl = ATable('test_docs.h5', 'atable1')

      # insert as with pytables:
      >>> row = tbl.row
      >>> for i in range(50):
      ...    row['x'], row['y'] = i, i * 10
      ...    row['name'] = "name_%i" % i
      ...    row.append()
      >>> tbl.flush()

      # there is also insert_many() method() with takes an iterable
      # of dicts with keys matching the colunns (x, y, name) in this
      # case.

      # query the data (query() alias of tables' readWhere()
      >>> tbl.query('(x > 4) & (y < 70)') #doctest: +NORMALIZE_WHITESPACE
      array([('name_5', 5.0, 50.0), ('name_6', 6.0, 60.0)],
            dtype=[('name', '|S16'), ('x', '<f4'), ('y', '<f4')])

    """

    import tables
    _filter = tables.Filters(complib="lzo", complevel=1, shuffle=True)

    class SimpleTable(tables.Table):
        def __init__(self, file_name, table_name, description=None,
                     group_name='default', mode='a', title="", filters=_filter,
                     expectedrows=512000):

            f = tables.openFile(file_name, mode)
            self.uservars = None

            if group_name is None: group_name = 'default'
            parentNode = f._getOrCreatePath('/' + group_name, True)

            if table_name in parentNode: # existing table
                description = None
            elif description is None: # pull the description from the attrs
                description = dict(self._get_description())

            tables.Table.__init__(self, parentNode, table_name,
                           description=description, title=title,
                           filters=filters,
                           expectedrows=expectedrows,
                           _log=False)
            self._c_classId = self.__class__.__name__

        def _get_description(self):
            # pull the description from the attrs
            for attr_name in dir(self):
                if attr_name[0] == '_': continue
                try:
                    attr = getattr(self, attr_name)
                except:
                    continue
                if isinstance(attr, tables.Atom):
                    yield attr_name, attr

        def insert_many(self, data_generator, attr=False):
            row = self.row
            cols = self.colnames
            if not attr:
                for d in data_generator:
                    for c in cols:
                        row[c] = d[c]
                    row.append()
            else:
                for d in data_generator:
                    for c in cols:
                        row[c] = getattr(d, c)
                    row.append()
            self.flush()

        query = tables.Table.readWhere

    # convience sublcass that i use a lot.
    class BlastTable(SimpleTable):
          query      = tables.StringCol(5)
          subject    = tables.StringCol(5)

          pctid      = tables.Float32Col()
          hitlen     = tables.UInt16Col()
          nmismatch  = tables.UInt16Col()
          ngaps      = tables.UInt16Col()

          qstart     = tables.UInt32Col()
          qstop      = tables.UInt32Col()
          sstart     = tables.UInt32Col()
          sstop      = tables.UInt32Col()

          evalue     = tables.Float64Col()
          score      = tables.Float32Col()


    if __name__ == '__main__':
        import doctest
        doctest.testmod()
        import os
        os.unlink('test_docs.h5')

