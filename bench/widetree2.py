import unittest

import tables as tb

verbose = 0


class Test(tb.IsDescription):
    ngroup = tb.Int32Col(pos=1)
    ntable = tb.Int32Col(pos=2)
    nrow = tb.Int32Col(pos=3)
    # string = StringCol(itemsize=500, pos=4)


class WideTreeTestCase(unittest.TestCase):
    def test00_leafs(self):

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
            group = fileh.create_group("/", f"group{k:04d}", f"Group {k}")

        fileh.close()

        # Now, create the tables
        rowswritten = 0
        for k in range(ngroups):
            print("Filling tables in group:", k)
            fileh = tb.open_file(filename, mode="a", root_uep=f"group{k:04d}")
            # Get the group
            group = fileh.root
            for j in range(ntables):
                # Create a table
                table = fileh.create_table(
                    group,
                    f"table{j:04d}",
                    Test,
                    f"Table{j:04d}",
                    tb.Filters(complevel, complib),
                    nrows,
                )
                # Get the row object associated with the new table
                row = table.row
                # Fill the table
                for i in range(nrows):
                    row["ngroup"] = k
                    row["ntable"] = j
                    row["nrow"] = i
                    row.append()

                rowswritten += nrows
                table.flush()

            # Close the file
            fileh.close()

        # read the file
        print("Reading...")
        rowsread = 0
        for ngroup in range(ngroups):
            fileh = tb.open_file(
                filename, mode="r", root_uep=f"group{ngroup:04d}"
            )
            # Get the group
            group = fileh.root

            if verbose:
                print("Group ==>", group)
            for ntable, table in enumerate(fileh.list_nodes(group, "Table")):
                if verbose > 1:
                    print("Table ==>", table)
                    print("Max rows in buf:", table.nrowsinbuf)
                    print("Rows in", table._v_pathname, ":", table.nrows)
                    print("Buffersize:", table.rowsize * table.nrowsinbuf)
                    print("MaxTuples:", table.nrowsinbuf)

                for nrow, row in enumerate(table):
                    try:
                        assert row["ngroup"] == ngroup
                        assert row["ntable"] == ntable
                        assert row["nrow"] == nrow
                    except Exception:
                        print(
                            f"Error in group: {ngroup}, table: {ntable}, "
                            f"row: {nrow}"
                        )
                        print("Record ==>", row)

                assert nrow == table.nrows
                rowsread += table.nrows

            # Close the file (eventually destroy the extended type)
            fileh.close()


# ----------------------------------------------------------------------
def suite():
    suite_ = unittest.TestSuite()
    from tests.common import make_suite

    suite_.addTest(make_suite(WideTreeTestCase))

    return suite_


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
