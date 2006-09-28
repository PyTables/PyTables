#import sys
import warnings
import unittest
from tables import *
import numarray

try:
    import Numeric
    numeric = 1
except:
    numeric = 0

import common
from common import verbose, cleanup, allequal, testFilename
# To delete the internal attributes automagically
unittest.TestCase.tearDown = cleanup

# Check read Tables from pytables version 0.5 (ucl-nrv2e), and 0.7 (ucl-nvr2d)
class BackCompatTablesTestCase(unittest.TestCase):

    #----------------------------------------

    def test01_readTable(self):
        """Checking backward compatibility of old formats of tables"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_readTable..." % self.__class__.__name__

        # Create an instance of an HDF5 Table
        warnings.filterwarnings("ignore", category=UserWarning)
        self.fileh = openFile(self.file, "r")
        warnings.filterwarnings("default", category=UserWarning)

        table = self.fileh.getNode("/tuple0")

        # Read the 100 records
        result = [ rec['var2'] for rec in table]
        if verbose:
            print "Nrows in", table._v_pathname, ":", table.nrows
            print "Last record in table ==>", rec
            print "Total selected records in table ==> ", len(result)

        assert len(result) == 100
        self.fileh.close()

class Table1_0UCL(BackCompatTablesTestCase):
    file = testFilename("Table1_0_ucl_nrv2e.h5")  # pytables 0.5.1 and before

class Table2_0UCL(BackCompatTablesTestCase):
    file = testFilename("Table2_0_ucl_nrv2d.h5")  # pytables 0.7.x versions

class Table2_1LZO(BackCompatTablesTestCase):
    file = testFilename("Table2_1_lzo_nrv2e_shuffle.h5")  # pytables 0.8.x versions and after

class Tables_LZO1(BackCompatTablesTestCase):
    file = testFilename("Tables_lzo1.h5")  # files compressed with LZO1

class Tables_LZO1_shuffle(BackCompatTablesTestCase):
    file = testFilename("Tables_lzo1_shuffle.h5")  # files compressed with LZO1 and shuffle

class Tables_LZO2(BackCompatTablesTestCase):
    file = testFilename("Tables_lzo2.h5")  # files compressed with LZO2

class Tables_LZO2_shuffle(BackCompatTablesTestCase):
    file = testFilename("Tables_lzo2_shuffle.h5")  # files compressed with LZO2 and shuffle

# Check read attributes from PyTables >= 1.0 properly
class BackCompatAttrsTestCase(unittest.TestCase):
    file = testFilename("zerodim-attrs-%s.h5")

    def test01_readAttr(self):
        """Checking backward compatibility of old formats for attributes"""

        if verbose:
            print '\n', '-=' * 30
            print "Running %s.test01_readAttr..." % self.__class__.__name__

        # Read old formats
        self.fileh = openFile(self.file % self.format, "r")
        a = self.fileh.getNode("/a")
        scalar = numarray.array(1, type="Int32")
        vector = numarray.array([1], type="Int32")
        if self.format == "1.3":
            assert allequal(a.attrs.arrdim1, vector)
            assert allequal(a.attrs.arrscalar, scalar)
            assert a.attrs.pythonscalar == 1
        elif self.format == "1.4":
            assert allequal(a.attrs.arrdim1, vector)
            assert allequal(a.attrs.arrscalar, scalar)
            assert allequal(a.attrs.pythonscalar, scalar)

        self.fileh.close()

class Attrs_1_3(BackCompatAttrsTestCase):
    format = "1.3"    # pytables 1.0.x versions and earlier

class Attrs_1_4(BackCompatAttrsTestCase):
    format = "1.4"    # pytables 1.1.x versions and later

class VLArrayTestCase(common.PyTablesTestCase):

    def test01_backCompat(self):
        """Checking backward compatibility with old flavors of VLArray"""

        # Open a PYTABLES_FORMAT_VERSION=1.6 file
        fileh = openFile(testFilename("flavored_vlarrays-format1.6.h5"), "r")
        # Check that we can read the contents without problems (nor warnings!)
        vlarray1 = fileh.root.vlarray1
        assert vlarray1.flavor == "numeric"
        if numeric:
            assert allequal(vlarray1[1], Numeric.array([5, 6, 7], typecode='i'),
                            "numeric")
        vlarray2 = fileh.root.vlarray2
        assert vlarray2.flavor == "python"
        assert vlarray2[1] == ['5', '6', '77']

        fileh.close()


#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()
    niter = 1

    ucl_avail = whichLibVersion("ucl") is not None
    lzo_avail = whichLibVersion("lzo") is not None
    for n in range(niter):
        theSuite.addTest(unittest.makeSuite(VLArrayTestCase))
        if ucl_avail:
            theSuite.addTest(unittest.makeSuite(Table1_0UCL))
            theSuite.addTest(unittest.makeSuite(Table2_0UCL))
            theSuite.addTest(unittest.makeSuite(Attrs_1_3))
            theSuite.addTest(unittest.makeSuite(Attrs_1_4))
        if lzo_avail:
            theSuite.addTest(unittest.makeSuite(Table2_1LZO))
            theSuite.addTest(unittest.makeSuite(Tables_LZO1))
            theSuite.addTest(unittest.makeSuite(Tables_LZO1_shuffle))
            theSuite.addTest(unittest.makeSuite(Tables_LZO2))
            theSuite.addTest(unittest.makeSuite(Tables_LZO2_shuffle))

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
