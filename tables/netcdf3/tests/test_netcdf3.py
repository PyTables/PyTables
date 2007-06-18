import unittest
import os
import tempfile
import numpy

from tables import netcdf3 as NetCDF
from tables.tests import common

# To delete the internal attributes automagically
unittest.TestCase.tearDown = common.cleanup


class NetCDFFileTestCase(common.PyTablesTestCase):

    def setUp(self):
        # Create an HDF5 with the NetCDF interface.
        self.file = tempfile.mktemp(".h5")
        file = NetCDF.NetCDFFile(self.file, mode = "w", history="created today")
        # create some dimensions.
        file.createDimension('t',None)
        file.createDimension('lat',3)
        file.createDimension('lon',7)
        file.createDimension('nchar',2)
        # create some variables.
        # foo has an unlimited dimension
        foo = file.createVariable('foo', 'f', ('t', 'lat', 'lon'))
        # bar does not
        bar = file.createVariable('bar', 'd', ('lat', 'lon'))
        # coordinate variables.
        times = file.createVariable('time', 's', ('t',))
        lats = file.createVariable('lat', '1', ('lat',))
        lons = file.createVariable('lon', 'c', ('nchar','lon'))
        # add some data.
        self.latdata = numpy.arange(100,400,100, dtype='int8')
        lats[:] = self.latdata
        self.londata = [['a','b','c','d','e','f','g'],
                        ['a','b','c','d','e','f','g']]
        lons[:] = self.londata
        for i in numpy.arange(bar.shape[0]):
            for j in numpy.arange(bar.shape[1]):
                bar[i, j] = i * j
        self.bardata = bar[:]
        # append data along unlimited dimension.
        nmax = 4
        for n in range(nmax):
            foo.append(n*numpy.ones(bar.shape,'f'))
            if n != nmax-1: # don't fill in last time
                times.append(10*(n+1))
        file.sync() # fill in timedata with _FillValue
        self.timedata = times[:]
        self.foodata = foo[:]
        # some file attributes
        file.title = 'unit test'
        file.magicNumbers= [42.,3.145,-1.]
        # some variable attributes
        bar.units = 'Ergs'
        bar.missing_value = -999
        foo.units = 'Watts'
        foo.missing_value = -999
        times.units = 'Years'
        lats.units = 'degrees north'
        lons.units = 'degrees east'
        file.close()

    def tearDown(self):
        # Remove the temporary file
        os.remove(self.file)
        common.cleanup(self)

    def test_repr(self):
        """open file and examine the __repr__ method"""

        f = NetCDF.NetCDFFile(self.file)
        outstring = "%s {\ndimensions:\n    lat = 3 ;\n    lon = 7 ;\n    t = UNLIMITED ; // (4 currently)\n    nchar = 2 ;\nvariables:\n    byte lat('lat',) ;\n        lat:units = 'degrees north' ;\n    character lon('nchar', 'lon') ;\n        lon:units = 'degrees east' ;\n    float foo('t', 'lat', 'lon') ;\n        foo:missing_value = -999 ;\n        foo:units = 'Watts' ;\n    double bar('lat', 'lon') ;\n        bar:missing_value = -999 ;\n        bar:units = 'Ergs' ;\n    short time('t',) ;\n        time:units = 'Years' ;\n// global attributes:\n        :history = 'created today' ;\n        :magicNumbers = [42.0, 3.145, -1.0] ;\n        :title = 'unit test' ;\n}" % self.file
        assert f.__repr__() == outstring
        f.close()

    def test_fileattrs(self):
        """check global attributes"""

        f = NetCDF.NetCDFFile(self.file)
        fileattrs = f.ncattrs()
        assert fileattrs == ['history', 'magicNumbers', 'title']
        assert f.history == 'created today'
        assert f.magicNumbers == [42.0, 3.145, -1.0]
        assert f.title == 'unit test'
        f.close()

    def test_dimensions(self):
        """check dimensions names and sizes"""

        f = NetCDF.NetCDFFile(self.file)
        dims = f.dimensions
        assert dims == {'lat':3, 'lon':7, 'nchar': 2, 't':None}
        f.close()

    def test_variabledata(self):
        """check data in variables"""

        f = NetCDF.NetCDFFile(self.file)
        latdata = f.variables['lat'][:]
        londata = f.variables['lon'][:]
        foodata = f.variables['foo'][:]
        bardata = f.variables['bar'][:]
        timedata = f.variables['time'][:]
        assert latdata.tolist() == self.latdata.tolist()
        assert londata.tolist() == self.londata
        assert foodata.tolist() == self.foodata.tolist()
        assert bardata.tolist() == self.bardata.tolist()
        assert timedata.tolist() == self.timedata.tolist()
        f.close()

    def test_appendata(self):
        """test appending data to an existing file"""

        f = NetCDF.NetCDFFile(self.file,'a')
        timedata = f.variables['time']
        timedata[3] = 40.
        timedata.append(50.)
        self.timedata = timedata[:]
        f.close()
        # close the file, re-open it
        f = NetCDF.NetCDFFile(self.file)
        timedata = f.variables['time']
        # check that data was actually appended
        assert timedata.shape == (5,)
        assert timedata[:].tolist() == self.timedata.tolist()
        # make sure that foo now contains _FillValue at the end of unlim dim
        foodata = f.variables['foo']
        assert foodata[-1,-1,-1] > 9.9+36
        f.close()

    def test_varttrs(self):
        """check variable names and variable attributes"""

        f = NetCDF.NetCDFFile(self.file)
        vars = f.variables.keys()
        assert vars == ['lat', 'lon', 'foo', 'bar', 'time']
        lats = f.variables['lat']
        assert lats.units == 'degrees north'
        lons = f.variables['lon']
        assert lons.units == 'degrees east'
        times = f.variables['time']
        assert times.units == 'Years'
        foo = f.variables['foo']
        assert foo.units == 'Watts'
        assert foo.missing_value == -999
        bar = f.variables['bar']
        assert bar.units == 'Ergs'
        assert bar.missing_value == -999
        f.close()

class NetCDFFileTestCase2(NetCDFFileTestCase):
# run this if Scientific.IO.NetCDF installed
# Extra tests to exercise h5tonc and nctoh5

    def test_h5tonc(self):
        """check h5 <--> netCDF conversion"""

        self.filenc = tempfile.mktemp(".nc")
        self.fileh5 = tempfile.mktemp(".h5")
        # convert to netCDF
        f = NetCDF.NetCDFFile(self.file)
        f.h5tonc(self.filenc)
        f.close()
        # convert back to HDF5
        f = NetCDF.NetCDFFile(self.fileh5,'w')
        nobjects, nbytes = f.nctoh5(self.filenc)
        # check to see that correct number of objects and bytes converted.
        assert (nobjects, nbytes) == (5, 529)
        # check that __repr__() on new HDF5 file is same as original
        outstring = "%s {\ndimensions:\n    lat = 3 ;\n    lon = 7 ;\n    t = UNLIMITED ; // (4 currently)\n    nchar = 2 ;\nvariables:\n    byte lat('lat',) ;\n        lat:units = 'degrees north' ;\n    character lon('nchar', 'lon') ;\n        lon:units = 'degrees east' ;\n    float foo('t', 'lat', 'lon') ;\n        foo:missing_value = -999 ;\n        foo:units = 'Watts' ;\n    double bar('lat', 'lon') ;\n        bar:missing_value = -999 ;\n        bar:units = 'Ergs' ;\n    short time('t',) ;\n        time:units = 'Years' ;\n// global attributes:\n        :history = 'created today' ;\n        :magicNumbers = [42.0, 3.145, -1.0] ;\n        :title = 'unit test' ;\n}" % self.fileh5
        assert f.__repr__() == outstring
        f.close()
        # check that size of new HDF5 file is same as original
        # (not a good idea, filesizes should not be expected to be the same)
#         file1size = os.stat(self.file).st_size
#         file2size = os.stat(self.fileh5).st_size
#         if common.verbose:
#             print "file1size-->", file1size
#             print "file2size-->", file2size
#         assert file1size == file2size
        os.remove(self.filenc)
        os.remove(self.fileh5)

#----------------------------------------------------------------------

def suite():
    theSuite = unittest.TestSuite()
    niter = 1

    for i in range(niter):
        if NetCDF.ScientificIONetCDF_imported:
            theSuite.addTest(unittest.makeSuite(NetCDFFileTestCase2))
        else:
            #print 'not testing hdf5 <--> netCDF conversion since Scientific.IO.NetCDF not installed'
            theSuite.addTest(unittest.makeSuite(NetCDFFileTestCase))

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )

## Local Variables:
## mode: python
## End:
