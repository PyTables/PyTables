# -*- coding: utf-8 -*-

import sys
import warnings
import unittest
import os
import tempfile

import numpy

from tables import *
# Next imports are only necessary for this test suite
from tables import Group, Leaf, Table, Array

from tables.tests import common
from tables.utils import quantize


# To delete the internal attributes automagically
unittest.TestCase.tearDown = common.cleanup


class QuantizeTestCase(unittest.TestCase):
    mode = "w"
    title = "This is the table title"
    expectedrows = 10
    appendrows = 5

    def setUp(self):
        # Create a temporary file
        #self.file = tempfile.mktemp(".h5")
        # Create an instance of HDF5 Table
        #self.h5file = open_file(self.file, self.mode, self.title)
        #self.populateFile()
        #self.h5file.close()
        self.data = numpy.linspace(-5., 5., 41)
        self.randomdata = numpy.random.random_sample(1000000)

    def populateFile(self):
        group = self.h5file.root
        maxshort = 1 << 15
        # maxint   = 2147483647   # (2 ** 31 - 1)
        for j in range(3):
            # Create a table
            table = self.h5file.create_table(group, 'table'+str(j), Record,
                                             title=self.title,
                                             filters=None,
                                             expectedrows=self.expectedrows)
            # Get the record object associated with the new table
            d = table.row
            # Fill the table
            for i in xrange(self.expectedrows):
                d['var1'] = '%04d' % (self.expectedrows - i)
                d['var2'] = i
                d['var3'] = i % maxshort
                d['var4'] = float(i)
                d['var5'] = float(i)
                d.append()      # This injects the Record values
            # Flush the buffer for this table
            table.flush()

            # Create a couple of arrays in each group
            var1List = [x['var1'] for x in table.iterrows()]
            var4List = [x['var4'] for x in table.iterrows()]

            self.h5file.create_array(group, 'var1', var1List, "1")
            self.h5file.create_array(group, 'var4', var4List, "4")

            # Create a new group (descendant of group)
            group2 = self.h5file.create_group(group, 'group'+str(j))
            # Iterate over this new group (group2)
            group = group2

    def tearDown(self):
        # Close the file
        #if self.h5file.isopen:
        #    self.h5file.close()

        #os.remove(self.file)
        #common.cleanup(self)
        pass

    #----------------------------------------

    def test00_quantizeData(self):
        "Checking the _quantize() function"
        quantized_0 = quantize(self.data, 0)
        quantized_1 = quantize(self.data, 1)
        quantized_2 = quantize(self.data, 2)
        quantized_m1 = quantize(self.data, -1)
        numpy.testing.assert_array_equal(quantized_0,
                numpy.asarray([-5.] * 2 + [-4.] * 5 + [-3.] * 3 + [-2.]
                    * 5 + [-1.] * 3 + [0.] * 5 + [1.] * 3 + [2.] * 5 +
                    [3.] * 3 + [4.] * 5 + [5.] * 2))
        numpy.testing.assert_array_equal(quantized_1, self.data)
        numpy.testing.assert_array_equal(quantized_2, self.data)
        numpy.testing.assert_array_equal(quantized_m1,
                numpy.asarray([-8.] * 4 + [0.] * 33 + [8.] * 4))

        quantized_0 = quantize(self.randomdata, 0)
        quantized_1 = quantize(self.randomdata, 1)
        quantized_2 = quantize(self.randomdata, 2)
        quantized_m1 = quantize(self.randomdata, -1)
        assert(numpy.abs(quantized_0 - self.randomdata).max() < 0.5)
        assert(numpy.abs(quantized_1 - self.randomdata).max() < 0.05)
        assert(numpy.abs(quantized_2 - self.randomdata).max() < 0.005)
        assert(numpy.abs(quantized_m1 - self.randomdata).max() < 1.)


#----------------------------------------------------------------------
def suite():
    theSuite = unittest.TestSuite()
    # This counter is useful when detecting memory leaks
    niter = 1

    for i in range(niter):
        theSuite.addTest(unittest.makeSuite(QuantizeTestCase))

    return theSuite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
