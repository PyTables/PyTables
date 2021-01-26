import unittest

import tables as tb

verbose = 0


class Test(tb.IsDescription):
    ngroup = tb.Int32Col(pos=1)
    ntable = tb.Int32Col(pos=2)
    nrow = tb.Int32Col(pos=3)
    #string = StringCol(itemsize=500, pos=4)


class WideTreeTestCase(unittest.TestCase):

    def test00_Leafs(self):

        # Open a new empty HDF5 file
        filename = "test_widetree.h5"
        ngroups = 10
        ntables = 300
        nrows = 10
        complevel = 0
        complib = "lzo"

        print("Writing...")
        # Open a file in "w"rite mode
        fileh = tb.open_file(filename, mode="w", title="PyTables Stress Test")

        for k in range(ngroups):
            # Create the group
            group = fileh.create_group("/", 'group%04d' % k, "Group %d" % k)

        fileh.close()

        # Now, create the tables
        rowswritten = 0
        for k in range(ngroups):
            print("Filling tables in group:", k)
            fileh = tb.open_file(filename, mode="a", root_uep='group%04d' % k)
            # Get the group
            group = fileh.root
            for j in range(ntables):
                # Create a table
                table = fileh.create_table(group, 'table%04d' % j, Test,
                                           'Table%04d' % j,
                                           tb.Filters(complevel, complib), nrows)
                # Get the row object associated with the new table
                row = table.row
                # Fill the table
                for i in range(nrows):
                    row['ngroup'] = k
                    row['ntable'] = j
                    row['nrow'] = i
                    row.append()

                rowswritten += nrows
                table.flush()

            # Close the file
            fileh.close()

        # read the file
        print("Reading...")
        rowsread = 0
        for ngroup in range(ngroups):
            fileh = tb.open_file(filename, mode="r", root_uep='group%04d' %
                              ngroup)
            # Get the group
            group = fileh.root
            ntable = 0
            if verbose:
                print("Group ==>", group)
            for table in fileh.list_nodes(group, 'Table'):
                if verbose > 1:
                    print("Table ==>", table)
                    print("Max rows in buf:", table.nrowsinbuf)
                    print("Rows in", table._v_pathname, ":", table.nrows)
                    print("Buffersize:", table.rowsize * table.nrowsinbuf)
                    print("MaxTuples:", table.nrowsinbuf)

                nrow = 0
                for row in table:
                    try:
                        assert row["ngroup"] == ngroup
                        assert row["ntable"] == ntable
                        assert row["nrow"] == nrow
                    except:
                        print("Error in group: %d, table: %d, row: %d" %
                              (ngroup, ntable, nrow))
                        print("Record ==>", row)
                    nrow += 1

                assert nrow == table.nrows
                rowsread += table.nrows
                ntable += 1

            # Close the file (eventually destroy the extended type)
            fileh.close()


#----------------------------------------------------------------------
def suite():
    theSuite = unittest.TestSuite()
    theSuite.addTest(unittest.makeSuite(WideTreeTestCase))

    return theSuite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
