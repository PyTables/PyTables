#import sys
import warnings
import unittest
#import os
#import tempfile

# import numarray
# from numarray import *
# #import recarray
# import numarray.records as records
from tables import *

from test_all import verbose, cleanup
# To delete the internal attributes automagically
unittest.TestCase.tearDown = cleanup

# Check read Tables from pytables version 0.5 (ucl-nrv2e), and 0.7 (ucl-nvr2d)
class BackCompatTestCase(unittest.TestCase):

    #----------------------------------------

    def test01_readTable(self):
        """Checking backward compatibility of old formats"""

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
        
class Table1_0UCL(BackCompatTestCase):
    file = "Table1_0_ucl_nrv2e.h5"  # pytables 0.5.1 and before

class Table2_0UCL(BackCompatTestCase):
    file = "Table2_0_ucl_nrv2d.h5"  # pytables 0.7.x versions

class Table2_1UCL(BackCompatTestCase):
    file = "Table2_1_ucl_nrv2e_shuffle.h5"  # pytables 0.8.x versions and after


#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()
    niter = 1

    #theSuite.addTest(unittest.makeSuite(Table1_0UCL))
    #theSuite.addTest(unittest.makeSuite(Table2_0UCL))
    #theSuite.addTest(unittest.makeSuite(Table2_1UCL))

    ucl_avail = whichLibVersion("ucl")[0]
    for n in range(niter):
        if ucl_avail:
            theSuite.addTest(unittest.makeSuite(Table1_0UCL))
            theSuite.addTest(unittest.makeSuite(Table2_0UCL))
            theSuite.addTest(unittest.makeSuite(Table2_1UCL))
            
    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
