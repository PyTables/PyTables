########################################################################
#
#	License: BSD
#	Created: October 2, 2004
#	Author:  Ivan Vilata i Balaguer - reverse:net.selidor@ivan
#
#	$Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/test/test_filenode.py,v $
#	$Id: test_filenode.py,v 1.2 2004/10/29 10:05:05 falted Exp $
#
########################################################################

"Unit test for the FileNode module."

import unittest, tempfile, os, sys
import tables
from tables.nodes import FileNode
import warnings

from test_all import verbose


__revision__ = '$Id: test_filenode.py,v 1.2 2004/10/29 10:05:05 falted Exp $'



class NewFileTestCase(unittest.TestCase):
	"Tests creating a new file node with the newNode() function."

	def setUp(self):
		"""setUp() -> None

		This method sets the following instance attributes:
		  * 'h5fname', the name of the temporary HDF5 file
		  * 'h5file', the writable, empty, temporary HDF5 file
		"""

		self.h5fname = tempfile.mktemp(suffix = '.h5', dir = '.')
		self.h5file = tables.openFile(
			self.h5fname, 'w', title = "Test for creating a new file node")


	def tearDown(self):
		"""tearDown() -> None

		Closes 'h5file'; removes 'h5fname'.
		"""

		self.h5file.close()
		self.h5file = None
		os.remove(self.h5fname)


	def test00_NewFile(self):
		"Creation of a brand new file node."

		try:
			fnode = FileNode.newNode(self.h5file, where = '/', name = 'test')
			node = self.h5file.getNode('/test')
		except LookupError:
			self.fail("FileNode.newNode() failed to create a new node.")
		else:
			self.assertEqual(
				fnode.node, node,
				"FileNode.newNode() created a node in the wrong place.")


	def test01_NewFileTooFewArgs(self):
		"Creation of a new file node without arguments for node creation."

		self.assertRaises(TypeError, FileNode.newNode, self.h5file)


	def test02_NewFileWithExpectedSize(self):
		"Creation of a new file node with 'expectedsize' argument."

		try:
			FileNode.newNode(
				self.h5file, where = '/', name = 'test', expectedsize = 100000)
		except TypeError:
			self.fail("\
FileNode.newNode() failed to accept 'expectedsize' argument.")


	def test03_NewFileWithExpectedRows(self):
		"Creation of a new file node with illegal 'expectedrows' argument."

		self.assertRaises(
			TypeError, FileNode.newNode,
			self.h5file, where = '/', name = 'test', expectedrows = 100000)



def copyFileToFile(srcfile, dstfile, blocksize = 4096):
	"""copyFileToFile(srcfile, dstfile[, blocksize]) -> None

	Copies a readable opened file 'srcfile' to a writable opened file 'destfile'
	in blocks of 'blocksize' bytes (4 KiB by default).
	"""

	data = srcfile.read(blocksize)
	while len(data) > 0:
		dstfile.write(data)
		data = srcfile.read(blocksize)



class WriteFileTestCase(unittest.TestCase):
	"Tests writing, seeking and truncating a new file node."

	datafname = 'test_filenode.dat'


	def setUp(self):
		"""setUp() -> None

		This method sets the following instance attributes:
		  * 'h5fname', the name of the temporary HDF5 file
		  * 'h5file', the writable, temporary HDF5 file with a '/test' node
		  * 'fnode', the writable file node in '/test'
		"""

		self.h5fname = tempfile.mktemp(suffix = '.h5', dir = '.')
		self.h5file = tables.openFile(
			self.h5fname, 'w', title = "Test for writing a file node")
		self.fnode = FileNode.newNode(self.h5file, where = '/', name = 'test')


	def tearDown(self):
		"""tearDown() -> None

		Closes 'fnode' and 'h5file'; removes 'h5fname'.
		"""

		self.fnode.close()
		self.fnode = None
		self.h5file.close()
		self.h5file = None
		os.remove(self.h5fname)


	def test00_WriteFile(self):
		"Writing a whole file node."

		datafile = file(self.datafname)
		try:
			copyFileToFile(datafile, self.fnode)
		finally:
			datafile.close()


	def test01_SeekFile(self):
		"Seeking and writing file node."

		self.fnode.write('0123')
		self.fnode.seek(8)
		self.fnode.write('4567')
		self.fnode.seek(3)
		data = self.fnode.read(6)
		self.assertEqual(
			data, '3\0\0\0\0''4',
			"Gap caused by forward seek was not properly filled.")

		self.fnode.seek(0)
		self.fnode.write('test')

		self.fnode.seek(0)
		data = self.fnode.read(4)
		self.assertNotEqual(
			data, 'test', "Data was overwritten instead of appended.")

		self.fnode.seek(-4, 2)
		data = self.fnode.read(4)
		self.assertEqual(data, 'test', "Written data was not appended.")

		self.fnode.seek(0, 2)
		oldendoff = self.fnode.tell()
		self.fnode.seek(-2, 2)
		self.fnode.write('test')
		newendoff = self.fnode.tell()
		self.assertEqual(
			newendoff, oldendoff + 4,
			"Pointer was not correctly moved on append.")


	def test02_TruncateFile(self):
		"Truncating a file node."

		self.fnode.write('test')

		self.fnode.seek(2)
		self.assertRaises(IOError, self.fnode.truncate)

		self.fnode.seek(6)
		self.fnode.truncate()
		self.fnode.seek(0)
		data = self.fnode.read()
		self.assertEqual(
			data, 'test\0\0', "File was not grown to the current offset.")

		self.fnode.truncate(8)
		self.fnode.seek(0)
		data = self.fnode.read()
		self.assertEqual(
			data, 'test\0\0\0\0', "File was not grown to an absolute size.")



class OpenFileTestCase(unittest.TestCase):
	"Tests opening an existing file node for reading and writing."

	def setUp(self):
		"""setUp() -> None

		This method sets the following instance attributes:
		  * 'h5fname', the name of the temporary HDF5 file
		  * 'h5file', the writable, temporary HDF5 file with a '/test' node
		"""

		self.h5fname = tempfile.mktemp(suffix = '.h5', dir = '.')
		self.h5file = tables.openFile(
			self.h5fname, 'w', title = "Test for opening a file node")

		fnode = FileNode.newNode(self.h5file, where = '/', name = 'test')
		fnode.close()


	def tearDown(self):
		"""tearDown() -> None

		Closes 'h5file'; removes 'h5fname'.
		"""

		self.h5file.close()
		self.h5file = None
		os.remove(self.h5fname)


	def test00_OpenFileRead(self):
		"Opening an existing file node for reading."

		node = self.h5file.getNode('/test')
		fnode = FileNode.openNode(node)
		self.assertEqual(
			fnode.node, node, "FileNode.openNode() opened the wrong node.")
		self.assertEqual(
			fnode.mode, 'r',
			"File was opened with an invalid mode %s." % repr(fnode.mode))
		self.assertEqual(
			fnode.tell(), 0L,
			"Pointer is not positioned at the beginning of the file.")
		fnode.close()


	def test01_OpenFileReadAppend(self):
		"Opening an existing file node for reading and appending."

		node = self.h5file.getNode('/test')
		fnode = FileNode.openNode(node, 'a+')
		self.assertEqual(
			fnode.node, node, "FileNode.openNode() opened the wrong node.")
		self.assertEqual(
			fnode.mode, 'a+',
			"File was opened with an invalid mode %s." % repr(fnode.mode))

		self.assertEqual(
			fnode.tell(), 0L,
			"Pointer is not positioned at the end of the file.")
		fnode.close()


	def test02_OpenFileInvalidMode(self):
		"Opening an existing file node with an invalid mode."

		self.assertRaises(
			IOError, FileNode.openNode, self.h5file.getNode('/test'), 'w')


	def test03_OpenFileNoAttrs(self):
		"Opening a node with no type attributes."

		node = self.h5file.getNode('/test')
		# 2004-10-02: This method doesn't exist! (BUG #1049297)
		self.h5file.delAttrNode('/test', '_type')
		#   So the value is changed to get the same result.
		#self.h5file.setAttrNode('/test', '_type', 'foobar')
		self.assertRaises(ValueError, FileNode.openNode, node)



class ReadFileTestCase(unittest.TestCase):
	"Tests reading from an existing file node."

	datafname = 'test_filenode.xbm'


	def setUp(self):
		"""setUp() -> None

		This method sets the following instance attributes:
		  * 'datafile', the opened data file
		  * 'h5fname', the name of the temporary HDF5 file
		  * 'h5file', the writable, temporary HDF5 file with a '/test' node
		  * 'fnode', the readable file node in '/test', with data in it
		"""

		self.datafile = file(self.datafname)

		self.h5fname = tempfile.mktemp(suffix = '.h5', dir = '.')
		self.h5file = tables.openFile(
			self.h5fname, 'w', title = "Test for reading a file node")

		fnode = FileNode.newNode(self.h5file, where = '/', name = 'test')
		copyFileToFile(self.datafile, fnode)
		fnode.close()

		self.datafile.seek(0)
		self.fnode = FileNode.openNode(self.h5file.getNode('/test'))


	def tearDown(self):
		"""tearDown() -> None

		Closes 'fnode', 'h5file' and 'datafile'; removes 'h5fname'.
		"""

		self.fnode.close()
		self.fnode = None

		self.h5file.close()
		self.h5file = None
		os.remove(self.h5fname)

		self.datafile.close()
		self.datafile = None


	def test00_CompareFile(self):
		"Reading and comparing a whole file node."

		import md5

		dfiledigest = md5.new(self.datafile.read()).digest()
		fnodedigest = md5.new(self.fnode.read()).digest()

		self.assertEqual(
			dfiledigest, fnodedigest,
			"Data read from file node differs from that in the file on disk.")


	def test01_Write(self):
		"Writing on a read-only file."

		self.assertRaises(IOError, self.fnode.write, 'no way')


	def test02_UseAsImageFile(self):
		"Using a file node with Python Imaging Library."

		try:
			import Image

			Image.open(self.fnode)
		except ImportError:
			# PIL not available, nothing to do.
			pass
		except IOError:
			self.fail("PIL was not able to create an image from the file node.")



class ReadlineTestCaseMixin:
	"""Mix-in class for text line-reading test cases.

	It provides a set of tests independent of the line separator string.
	Sub-classes must provide the 'lineSeparator' attribute.
	"""

	def setUp(self):
		"""setUp() -> None

		This method sets the following instance attributes:
		  * 'h5fname', the name of the temporary HDF5 file
		  * 'h5file', the writable, temporary HDF5 file with a '/test' node
		  * 'fnode', the readable file node in '/test', with text in it
		"""

		linesep = self.lineSeparator

		self.h5fname = tempfile.mktemp(suffix = '.h5', dir = '.')
		self.h5file = tables.openFile(
			self.h5fname, 'w',
			title = "Test for reading text lines from a file node")

		# Fill the node file with some text.
		fnode = FileNode.newNode(self.h5file, where = '/', name = 'test')
		fnode.lineSeparator = linesep
		fnode.write(linesep)
		fnode.write('short line%sshort line%s%s' % ((linesep,) * 3))
		fnode.write('long line ' * 20 + linesep)
		fnode.write('unterminated')
		fnode.close()

		# Re-open it for reading.
		self.fnode = FileNode.openNode(self.h5file.getNode('/test'))
		self.fnode.lineSeparator = linesep


	def tearDown(self):
		"""tearDown() -> None

		Closes 'fnode' and 'h5file'; removes 'h5fname'.
		"""

		self.fnode.close()
		self.fnode = None

		self.h5file.close()
		self.h5file = None
		os.remove(self.h5fname)


	def test00_Readline(self):
		"Reading individual lines."

		linesep = self.lineSeparator

		line = self.fnode.readline()
		self.assertEqual(line, linesep)

		line = self.fnode.readline()  # 'short line' + linesep
		line = self.fnode.readline()
		self.assertEqual(line, 'short line' + linesep)
		line = self.fnode.readline()
		self.assertEqual(line, linesep)

		line = self.fnode.readline()
		self.assertEqual(line, 'long line ' * 20 + linesep)

		line = self.fnode.readline()
		self.assertEqual(line, 'unterminated')

		line = self.fnode.readline()
		self.assertEqual(line, '')

		line = self.fnode.readline()
		self.assertEqual(line, '')


	def test01_ReadlineSeek(self):
		"Reading individual lines and seeking back and forth."

		linesep = self.lineSeparator
		lseplen = len(linesep)

		self.fnode.readline()  # linesep
		self.fnode.readline()  # 'short line' + linesep

		self.fnode.seek(-(lseplen + 4), 1)
		line = self.fnode.readline()
		self.assertEqual(
			line, 'line' + linesep, "Seeking back yielded different data.")

		self.fnode.seek(lseplen + 20, 1)  # Into the long line.
		line = self.fnode.readline()
		self.assertEqual(
			line[-(lseplen + 10):], 'long line ' + linesep,
			"Seeking forth yielded unexpected data.")


	def test02_Iterate(self):
		"Iterating over the lines."

		linesep = self.lineSeparator

		# Iterate to the end.
		for line in self.fnode:
			pass

		self.assertRaises(StopIteration, self.fnode.next)

		self.fnode.seek(0)

		line = self.fnode.next()
		self.assertEqual(line, linesep)

		line = self.fnode.next()
		self.assertEqual(line, 'short line' + linesep)


	def test03_Readlines(self):
		"Reading a list of lines."

		linesep = self.lineSeparator

		lines = self.fnode.readlines()
		self.assertEqual(
			lines, [
				linesep, 'short line' + linesep, 'short line' + linesep,
				linesep, 'long line ' * 20 + linesep, 'unterminated'])


	def test04_ReadlineSize(self):
		"Reading individual lines of limited size."

		linesep = self.lineSeparator
		lseplen = len(linesep)

		line = self.fnode.readline()  # linesep

		line = self.fnode.readline(lseplen + 20)
		self.assertEqual(line, 'short line' + linesep)

		line = self.fnode.readline(5)
		self.assertEqual(line, 'short')

		line = self.fnode.readline(lseplen + 20)
		self.assertEqual(line, ' line' + linesep)

		line = self.fnode.readline(lseplen)
		self.assertEqual(line, linesep)

		self.fnode.seek(-4, 2)
		line = self.fnode.readline(4)
		self.assertEqual(line, 'ated')

		self.fnode.seek(-4, 2)
		line = self.fnode.readline(20)
		self.assertEqual(line, 'ated')


	def test05_ReadlinesSize(self):
		"Reading a list of lines with a limited size."

		linesep = self.lineSeparator

		lines = self.fnode.readlines(
			len('%sshort line%sshort' % ((linesep,) * 2)))
		self.assertEqual(
			lines, [linesep, 'short line' + linesep, 'short'])

		line = self.fnode.readline()
		self.assertEqual(line, ' line' + linesep)



class MonoReadlineTestCase(ReadlineTestCaseMixin, unittest.TestCase):
	"Tests reading one-byte-separated text lines from an existing file node."

	lineSeparator = '\n'



class MultiReadlineTestCase(ReadlineTestCaseMixin, unittest.TestCase):
	"Tests reading multibyte-separated text lines from an existing file node."

	lineSeparator = '<br/>'



class LineSeparatorTestCase(unittest.TestCase):
	"Tests text line separator manipulation in a file node."

	def setUp(self):
		"""setUp() -> None

		This method sets the following instance attributes:
		  * 'h5fname', the name of the temporary HDF5 file
		  * 'h5file', the writable, temporary HDF5 file with a '/test' node
		  * 'fnode', the writable file node in '/test'
		"""

		self.h5fname = tempfile.mktemp(suffix = '.h5', dir = '.')
		self.h5file = tables.openFile(
			self.h5fname, 'w',
			title = "Test for line separator manipulation in a file node")
		self.fnode = FileNode.newNode(self.h5file, where = '/', name = 'test')


	def tearDown(self):
		"""tearDown() -> None

		Closes 'fnode' and 'h5file'; removes 'h5fname'.
		"""

		self.fnode.close()
		self.fnode = None
		self.h5file.close()
		self.h5file = None
		os.remove(self.h5fname)


	def test00_DefaultLineSeparator(self):
		"Default line separator."

		self.assertEqual(
			self.fnode.lineSeparator, os.linesep,
			"Default line separator does not match that in os.linesep.")


	def test01_SetLineSeparator(self):
		"Setting a valid line separator."

		try:
			self.fnode.lineSeparator = 'SEPARATOR'
		except ValueError:
			self.fail("Valid line separator was not accepted.")
		else:
			self.assertEqual(
				self.fnode.lineSeparator, 'SEPARATOR',
				"Line separator was not correctly set.")


	def test02_SetInvalidLineSeparator(self):
		"Setting an invalid line separator."

		self.assertRaises(
			ValueError, setattr, self.fnode, 'lineSeparator', '')
		self.assertRaises(
			ValueError, setattr, self.fnode, 'lineSeparator', 'x' * 1024)



class AttrsTestCase(unittest.TestCase):
	"Tests setting and getting file node attributes."

	def setUp(self):
		"""setUp() -> None

		This method sets the following instance attributes:
		  * 'h5fname', the name of the temporary HDF5 file
		  * 'h5file', the writable, temporary HDF5 file with a '/test' node
		  * 'fnode', the writable file node in '/test'
		"""

		self.h5fname = tempfile.mktemp(suffix = '.h5', dir = '.')
		self.h5file = tables.openFile(
			self.h5fname, 'w', title = "Test for file node attribute handling")
		self.fnode = FileNode.newNode(self.h5file, where = '/', name = 'test')


	def tearDown(self):
		"""tearDown() -> None

		Closes 'fnode' and 'h5file'; removes 'h5fname'.
		"""

		self.fnode.close()
		self.fnode = None
		self.h5file.close()
		self.h5file = None
		os.remove(self.h5fname)


	def test00_GetTypeAttr(self):
		"Getting the type attribute of a file node."

		self.assertEqual(
			getattr(self.fnode.attrs, '_type', None), FileNode.NodeType,
			"File node has no '_type' attribute.")


	def test01_SetSystemAttr(self):
		"Setting a system attribute on a file node."

		self.assertRaises(
			RuntimeError, setattr, self.fnode.attrs, 'CLASS', 'foobar')


	def test02_SetGetDelUserAttr(self):
		"Setting a user attribute on a file node."

		self.assertEqual(
			getattr(self.fnode.attrs, 'userAttr', None), None,
			"Inexistent attribute has a value that is not 'None'.")

		self.fnode.attrs.userAttr = 'foobar'
		self.assertEqual(
			getattr(self.fnode.attrs, 'userAttr', None), 'foobar',
			"User attribute was not correctly set.")

		self.fnode.attrs.userAttr = 'bazquux'
		self.assertEqual(
			getattr(self.fnode.attrs, 'userAttr', None), 'bazquux',
			"User attribute was not correctly changed.")

		del self.fnode.attrs.userAttr
		# 2004-10-14: This unearths BUG #1049285 in PyTables'
		#   attribute removal.
		##self.assertEqual(
		##	getattr(self.fnode.attrs, 'userAttr', None), None,
		##	"User attribute was not deleted.")
		#   So we look at the list of user attributes.
		if 'userAttr' in self.fnode.attrs._f_list():
			self.fail("User attribute was not deleted.")


	def test03_AttrsOnClosedFile(self):
		"Accessing attributes on a closed file node."

		self.fnode.close()
		self.assertRaises(AttributeError, getattr, self.fnode, 'attrs')



class ClosedH5FileTestCase(unittest.TestCase):
	"Tests accessing a file node in a closed PyTables file."

	def setUp(self):
		"""setUp() -> None

		This method sets the following instance attributes:
		  * 'h5fname', the name of the temporary HDF5 file
		  * 'h5file', the closed HDF5 file with a '/test' node
		  * 'fnode', the writable file node in '/test'
		"""

		self.h5fname = tempfile.mktemp(suffix = '.h5', dir = '.')
		self.h5file = tables.openFile(
			self.h5fname, 'w',
			title = "Test for accessing a file node on a closed h5file")
		self.fnode = FileNode.newNode(self.h5file, where = '/', name = 'test')

		self.h5file.close()


	def tearDown(self):
		"""tearDown() -> None

		Closes 'fnode'; removes 'h5fname'.
		"""

		self.fnode.close()
		self.fnode = None
		self.h5file = None
		os.remove(self.h5fname)


	def test00_Write(self):
		"Writing to a file node in a closed PyTables file."

		self.assertRaises(ValueError, self.fnode.write, 'data')
		# Uncomment this if there is a need to catch a UserWarning
		# in the future
# 		warnings.filterwarnings("error", category=UserWarning)
# 		try:
# 			self.fnode.write('data')
# 			# self.fnode.write('data')
# 		except UserWarning:
# 			if verbose:
# 				(type, value, traceback) = sys.exc_info()
# 				print "\nGreat!, the next UserWarning was catched!"
# 				print value
# 		else:
# 			self.fail("expected an UserWarning")
# 		# Reset the warning
# 		warnings.filterwarnings("default", category=UserWarning)


	def test01_Attrs(self):
		"Accessing the attributes of a file node in a closed PyTables file."

		self.assertRaises(AttributeError, getattr, self.fnode, 'attrs')



#----------------------------------------------------------------------

def suite():
	theSuite = unittest.TestSuite()

	theSuite.addTest(unittest.makeSuite(ClosedH5FileTestCase))

	return theSuite

 
if __name__ == '__main__':
	unittest.main()
	#unittest.main( defaultTest='suite' )




## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## End:
