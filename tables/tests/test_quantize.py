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
        self.data = numpy.linspace(-5., 5., 41)
        self.randomdata = numpy.random.random_sample(1000000)
        self.randomints = numpy.random.random_integers(-1000000, 1000000,
                1000000)
        # Create a temporary file
        self.file = tempfile.mktemp(".h5")
        # Create an instance of HDF5 Table
        self.h5file = open_file(self.file, self.mode, self.title)
        self.populateFile()
        self.h5file.close()
        self.quantizeddata_0 = numpy.asarray([-5.] * 2 + [-4.] * 5 +
                [-3.] * 3 + [-2.] * 5 + [-1.] * 3 + [0.] * 5 + [1.] * 3
                + [2.] * 5 + [3.] * 3 + [4.] * 5 + [5.] * 2)
        self.quantizeddata_m1 = numpy.asarray([-8.] * 4 + [0.] * 33 +
                [8.] * 4)

    def populateFile(self):
        root = self.h5file.root
        filters = Filters(complevel=1, complib="blosc",
                least_significant_digit=1)
        ints = self.h5file.create_carray(root, "integers", Int64Atom(),
                (1000000, ), filters=filters)
        ints[:] = self.randomints
        floats = self.h5file.create_carray(root, "floats", Float32Atom(),
                (1000000, ), filters=filters)
        floats[:] = self.randomdata
        data1 = self.h5file.create_carray(root, "data1", Float64Atom(),
                (41, ), filters=filters)
        data1[:] = self.data
        filters = Filters(complevel=1, complib="blosc",
                least_significant_digit=0)
        data0 = self.h5file.create_carray(root, "data0", Float64Atom(),
                (41, ), filters=filters)
        data0[:] = self.data
        filters = Filters(complevel=1, complib="blosc",
                least_significant_digit=2)
        data2 = self.h5file.create_carray(root, "data2", Float64Atom(),
                (41, ), filters=filters)
        data2[:] = self.data
        filters = Filters(complevel=1, complib="blosc",
                least_significant_digit=-1)
        datam1 = self.h5file.create_carray(root, "datam1", Float64Atom(),
                (41, ), filters=filters)
        datam1[:] = self.data

    def tearDown(self):
        # Close the file
        if self.h5file.isopen:
            self.h5file.close()

        os.remove(self.file)
        common.cleanup(self)

    #----------------------------------------

    def test00_quantizeData(self):
        "Checking the quantize() function"
        quantized_0 = quantize(self.data, 0)
        quantized_1 = quantize(self.data, 1)
        quantized_2 = quantize(self.data, 2)
        quantized_m1 = quantize(self.data, -1)
        numpy.testing.assert_array_equal(quantized_0, self.quantizeddata_0)
        numpy.testing.assert_array_equal(quantized_1, self.data)
        numpy.testing.assert_array_equal(quantized_2, self.data)
        numpy.testing.assert_array_equal(quantized_m1, self.quantizeddata_m1)

    def test01_quantizeDataMaxError(self):
        "Checking the maximum error introduced by the quantize() function"
        quantized_0 = quantize(self.randomdata, 0)
        quantized_1 = quantize(self.randomdata, 1)
        quantized_2 = quantize(self.randomdata, 2)
        quantized_m1 = quantize(self.randomdata, -1)
        assert(numpy.abs(quantized_0 - self.randomdata).max() < 0.5)
        assert(numpy.abs(quantized_1 - self.randomdata).max() < 0.05)
        assert(numpy.abs(quantized_2 - self.randomdata).max() < 0.005)
        assert(numpy.abs(quantized_m1 - self.randomdata).max() < 1.)

    def test02_array(self):
        "Checking quantized data as written to disk"
        h5file = open_file(self.file, "r")
        numpy.testing.assert_array_equal(h5file.root.data1[:],
                self.data)
        numpy.testing.assert_array_equal(h5file.root.data2[:],
                self.data)
        numpy.testing.assert_array_equal(h5file.root.data0[:],
                self.quantizeddata_0)
        numpy.testing.assert_array_equal(h5file.root.datam1[:],
                self.quantizeddata_m1)
        numpy.testing.assert_array_equal(h5file.root.integers[:],
                self.randomints)
        assert(h5file.root.integers[:].dtype == self.randomints.dtype)
        assert(numpy.abs(h5file.root.floats[:] - self.randomdata).max()
                < 0.05)


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
