########################################################################
#
#	License: BSD
#	Created: December 15, 2004
#	Author:  Ivan Vilata i Balaguer - reverse:com.carabos@ivilata
#
#	$Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/test/test_timetype.py,v $
#	$Id$
#
########################################################################

"Unit test for the Time datatypes."

import unittest, tempfile, os
import tables, numarray
from test_all import verbose, allequal, heavy, cleanup
# To delete the internal attributes automagically
unittest.TestCase.tearDown = cleanup


__revision__ = '$Id$'



class LeafCreationTestCase(unittest.TestCase):
	"Tests creating Tables, VLArrays an EArrays with Time data."

	def setUp(self):
		"""setUp() -> None

		This method sets the following instance attributes:
		  * 'h5fname', the name of the temporary HDF5 file
		  * 'h5file', the writable, empty, temporary HDF5 file
		"""

		self.h5fname = tempfile.mktemp(suffix = '.h5')
		self.h5file = tables.openFile(
			self.h5fname, 'w', title = "Test for creating a time leaves")


	def tearDown(self):
		"""tearDown() -> None

		Closes 'h5file'; removes 'h5fname'.
		"""

		self.h5file.close()
		self.h5file = None
		os.remove(self.h5fname)


	def test00_UnidimLeaves(self):
		"Creating new nodes with unidimensional time elements."

		# Table creation.
		class MyTimeRow(tables.IsDescription):
			intcol = tables.IntCol()
			t32col = tables.Time32Col()
			t64col = tables.Time64Col()
		self.h5file.createTable('/', 'table', MyTimeRow)

		# VLArray creation.
		self.h5file.createVLArray('/', 'vlarray4', tables.Time32Atom())
		self.h5file.createVLArray('/', 'vlarray8', tables.Time64Atom())

		# EArray creation.
		self.h5file.createEArray('/', 'earray4', tables.Time32Atom(shape=(0,)))
		self.h5file.createEArray('/', 'earray8', tables.Time64Atom(shape=(0,)))


	def test01_MultidimLeaves(self):
		"Creating new nodes with multidimensional time elements."

		# Table creation.
		class MyTimeRow(tables.IsDescription):
			intcol = tables.IntCol(shape = (2, 1))
			t32col = tables.Time32Col(shape = (2, 1))
			t64col = tables.Time64Col(shape = (2, 1))
		self.h5file.createTable('/', 'table', MyTimeRow)

		# VLArray creation.
		self.h5file.createVLArray(
			'/', 'vlarray4', tables.Time32Atom(shape = (2, 1)))
		self.h5file.createVLArray(
			'/', 'vlarray8', tables.Time64Atom(shape = (2, 1)))

		# EArray creation.
		self.h5file.createEArray(
			'/', 'earray4', tables.Time32Atom(shape=(0, 2, 1)))
		self.h5file.createEArray(
			'/', 'earray8', tables.Time64Atom(shape=(0, 2, 1)))



class OpenTestCase(unittest.TestCase):
	"Tests opening a file with Time nodes."

	# The description used in the test Table.
	class MyTimeRow(tables.IsDescription):
		t32col = tables.Time32Col(shape = (2, 1))
		t64col = tables.Time64Col(shape = (2, 1))

	# The atoms used in the test VLArrays.
	myTime32Atom = tables.Time32Atom(shape = (2, 1))
	myTime64Atom = tables.Time64Atom(shape = (2, 1))


	def setUp(self):
		"""setUp() -> None

		This method sets the following instance attributes:
		  * 'h5fname', the name of the temporary HDF5 file with '/table',
		    '/vlarray4' and '/vlarray8' nodes.
		"""

		self.h5fname = tempfile.mktemp(suffix = '.h5')

		h5file = tables.openFile(
			self.h5fname, 'w', title = "Test for creating time leaves")

		# Create test Table.
		h5file.createTable('/', 'table', self.MyTimeRow)

		# Create test VLArrays.
		h5file.createVLArray('/', 'vlarray4', self.myTime32Atom)
		h5file.createVLArray('/', 'vlarray8', self.myTime64Atom)

		h5file.close()


	def tearDown(self):
		"""tearDown() -> None

		Removes 'h5fname'.
		"""

		os.remove(self.h5fname)


	def test00_OpenFile(self):
		"Opening a file with Time nodes."

		h5file = tables.openFile(self.h5fname)

		# Test the Table node.
		tbl = h5file.root.table
		self.assertEqual(
			tbl.coltypes['t32col'], self.MyTimeRow.columns['t32col'].type,
			"Column types do not match.")
		self.assertEqual(
			tbl.colshapes['t32col'], self.MyTimeRow.columns['t32col'].shape,
			"Column shapes do not match.")
		self.assertEqual(
			tbl.coltypes['t64col'], self.MyTimeRow.columns['t64col'].type,
			"Column types do not match.")
		self.assertEqual(
			tbl.colshapes['t64col'], self.MyTimeRow.columns['t64col'].shape,
			"Column shapes do not match.")

		# Test the VLArray nodes.
		vla4 = h5file.root.vlarray4
		self.assertEqual(
			vla4.atom.type, self.myTime32Atom.type,
			"Atom types do not match.")
		self.assertEqual(
			vla4.atom.shape, self.myTime32Atom.shape,
			"Atom shapes do not match.")

		vla8 = h5file.root.vlarray8
		self.assertEqual(
			vla8.atom.type, self.myTime64Atom.type,
			"Atom types do not match.")
		self.assertEqual(
			vla8.atom.shape, self.myTime64Atom.shape,
			"Atom shapes do not match.")

		h5file.close()


	def test01_OpenFileStype(self):
		"Opening a file with Time nodes, comparing Atom.stype."

		h5file = tables.openFile(self.h5fname)

		# Test the Table node.
		tbl = h5file.root.table
		self.assertEqual(
			tbl.colstypes['t32col'], self.MyTimeRow.columns['t32col'].stype,
			"Column types do not match.")
		self.assertEqual(
			tbl.colstypes['t64col'], self.MyTimeRow.columns['t64col'].stype,
			"Column types do not match.")

		# Test the VLArray nodes.
		vla4 = h5file.root.vlarray4
		self.assertEqual(
			vla4.atom.stype, self.myTime32Atom.stype,
			"Atom types do not match.")

		vla8 = h5file.root.vlarray8
		self.assertEqual(
			vla8.atom.stype, self.myTime64Atom.stype,
			"Atom types do not match.")

		h5file.close()



class CompareTestCase(unittest.TestCase):
	"Tests whether stored and retrieved time data is kept the same."

	# The description used in the test Table.
	class MyTimeRow(tables.IsDescription):
		t32col = tables.Time32Col(pos = 0)
		t64col = tables.Time64Col(shape = (2,), pos = 1)

	# The atoms used in the test VLArrays.
	myTime32Atom = tables.Time32Atom()
	myTime64Atom = tables.Time64Atom()


	def setUp(self):
		"""setUp() -> None

		This method sets the following instance attributes:
		  * 'h5fname', the name of the temporary HDF5 file
		"""

		self.h5fname = tempfile.mktemp(suffix = '.h5')


	def tearDown(self):
		"""tearDown() -> None

		Removes 'h5fname'.
		"""

		os.remove(self.h5fname)


	def test00_Compare32VLArray(self):
		"Comparing written 32-bit time data with read data in a VLArray."

		wtime = 1234567890

		# Create test VLArray with data.
		h5file = tables.openFile(
			self.h5fname, 'w', title = "Test for comparing 32-bit times")
		vla = h5file.createVLArray('/', 'test', self.myTime32Atom)
		vla.append([wtime])
		h5file.close()

		# Check the written data.
		h5file = tables.openFile(self.h5fname)
		rtime = h5file.root.test.read()[0][0]
		h5file.close()
		self.assertEqual(
			rtime, wtime, "Stored and retrieved values do not match.")


	def test01_Compare64VLArray(self):
		"Comparing written 64-bit time data with read data in a VLArray."

		wtime = 1234567890.123456

		# Create test VLArray with data.
		h5file = tables.openFile(
			self.h5fname, 'w', title = "Test for comparing 64-bit times")
		vla = h5file.createVLArray('/', 'test', self.myTime64Atom)
		vla.append([wtime])
		h5file.close()

		# Check the written data.
		h5file = tables.openFile(self.h5fname)
		rtime = h5file.root.test.read()[0][0]
		h5file.close()
		self.assertEqual(
			rtime, wtime, "Stored and retrieved values do not match.")


	def test02_CompareTable(self):
		"Comparing written time data with read data in a Table."

		wtime = 1234567890.123456

		# Create test Table with data.
		h5file = tables.openFile(
			self.h5fname, 'w', title = "Test for comparing 64-bit times")
		tbl = h5file.createTable('/', 'test', self.MyTimeRow)
		row = tbl.row
		row['t32col'] = int(wtime)
		row['t64col'] = (wtime, wtime)
		row.append()
		h5file.close()

		# Check the written data.
		h5file = tables.openFile(self.h5fname)
		recarr = h5file.root.test.read(0)
		h5file.close()

		self.assertEqual(
			recarr.field('t32col')[0], int(wtime),
			"Stored and retrieved values do not match.")

		comp = (recarr.field('t64col')[0] == numarray.array((wtime, wtime)))
		self.assert_(
			numarray.alltrue(comp),
			"Stored and retrieved values do not match.")

	def test02b_CompareTable(self):
		"Comparing written time data with read data in a Table (several vals)."

		# Create test Table with data.
		h5file = tables.openFile(
			self.h5fname, 'w', title = "Test for comparing 64-bit times")
		tbl = h5file.createTable('/', 'test', self.MyTimeRow)
		row = tbl.row
		#nrows = tbl._v_maxTuples + 1034  # add some more rows than buffer
		nrows = 10  # only for home checks... The value above should check
		            # better the I/O with multiple buffers
		for i in xrange(nrows):
			row['t32col'] = i
			row['t64col'] = (i+0.012, i+0.012)
			row.append()
		h5file.close()

		# Check the written data.
		h5file = tables.openFile(self.h5fname)
		tbl = h5file.root.test
		recarr = h5file.root.test.read()
		h5file.close()

		orig_val = numarray.arange(nrows,type=numarray.Int32)
		if verbose:
			print "Original values:", orig_val
			print "Saved values:", recarr.field('t32col')[:]

		self.assert_(
			numarray.alltrue(recarr.field('t32col')[:] == orig_val),
			"Stored and retrieved values do not match.")

		orig_val = numarray.arange(0,nrows,0.5, type=numarray.Int32,
								   shape=(nrows,2)) + 0.012
		if verbose:
			print "Original values:", orig_val
			print "Saved values:", recarr.field('t64col')[:]

		self.assert_(
            allequal(recarr.field('t64col')[:], orig_val, numarray.Float64),
			"Stored and retrieved values do not match.")


	def test03_Compare64EArray(self):
		"Comparing written 64-bit time data with read data in an EArray."

		wtime = 1234567890.123456

		# Create test EArray with data.
		h5file = tables.openFile(
			self.h5fname, 'w', title = "Test for comparing 64-bit times")
		vla = h5file.createEArray('/', 'test', tables.Time64Atom(shape=(0,)))
		vla.append([wtime])
		h5file.close()

		# Check the written data.
		h5file = tables.openFile(self.h5fname)
		rtime = h5file.root.test.read()[0]
		h5file.close()
		self.assertEqual(
			rtime, wtime, "Stored and retrieved values do not match.")



#----------------------------------------------------------------------

def suite():
	"""suite() -> test suite

	Returns a test suite consisting of all the test cases in the module.
	"""

	theSuite = unittest.TestSuite()

	theSuite.addTest(unittest.makeSuite(LeafCreationTestCase))
	theSuite.addTest(unittest.makeSuite(OpenTestCase))
	theSuite.addTest(unittest.makeSuite(CompareTestCase))
# More tests are needed so as to checking atributes, compression, exceptions,
# etc... Francesc Altet 2005-01-06


	return theSuite


if __name__ == '__main__':
	unittest.main(defaultTest = 'suite')



## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## End:
