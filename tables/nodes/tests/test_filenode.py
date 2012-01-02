########################################################################
#
#       License: BSD
#       Created: October 2, 2004
#       Author:  Ivan Vilata i Balaguer - reverse:net.selidor@ivan
#
#       $Id$
#
########################################################################

"Unit test for the filenode module."

import unittest, tempfile, os
import warnings

import tables
from tables.nodes import filenode
from tables.tests import common


__revision__ = '$Id$'



class NewFileTestCase(common.TempFileMixin, common.PyTablesTestCase):
    "Tests creating a new file node with the newNode() function."

    def test00_NewFile(self):
        "Creation of a brand new file node."

        try:
            fnode = filenode.newNode(self.h5file, where = '/', name = 'test')
            node = self.h5file.getNode('/test')
        except LookupError:
            self.fail("filenode.newNode() failed to create a new node.")
        else:
            self.assertEqual(
                    fnode.node, node,
                    "filenode.newNode() created a node in the wrong place.")


    def test01_NewFileTooFewArgs(self):
        "Creation of a new file node without arguments for node creation."

        self.assertRaises(TypeError, filenode.newNode, self.h5file)


    def test02_NewFileWithExpectedSize(self):
        "Creation of a new file node with 'expectedsize' argument."

        try:
            filenode.newNode(
                    self.h5file, where = '/', name = 'test', expectedsize = 100000)
        except TypeError:
            self.fail("\
filenode.newNode() failed to accept 'expectedsize' argument.")


    def test03_NewFileWithExpectedRows(self):
        "Creation of a new file node with illegal 'expectedrows' argument."

        self.assertRaises(
                TypeError, filenode.newNode,
                self.h5file, where = '/', name = 'test', expectedrows = 100000)



class ClosedFileTestCase(common.TempFileMixin, common.PyTablesTestCase):
    "Tests calling several methods on a closed file."

    def setUp(self):
        """setUp() -> None

        This method sets the following instance attributes:
          * 'h5fname', the name of the temporary HDF5 file
          * 'h5file', the writable, temporary HDF5 file with a '/test' node
          * 'fnode', the closed file node in '/test'
        """
        super(ClosedFileTestCase, self).setUp()
        self.fnode = filenode.newNode(self.h5file, where = '/', name = 'test')
        self.fnode.close()


    def tearDown(self):
        """tearDown() -> None

        Closes 'h5file'; removes 'h5fname'.
        """
        self.fnode = None
        super(ClosedFileTestCase, self).tearDown()


    # All these tests mey seem odd, but Python (2.3) files
    # do test whether the file is not closed regardless of their mode.

    def test00_Close(self):
        "Closing a closed file."

        try:
            self.fnode.close()
        except ValueError:
            self.fail("Could not close an already closed file.")


    def test01_Flush(self):
        "Flushing a closed file."

        self.assertRaises(ValueError, self.fnode.flush)


    def test02_Next(self):
        "Getting the next line of a closed file."

        self.assertRaises(ValueError, self.fnode.next)


    def test03_Read(self):
        "Reading a closed file."

        self.assertRaises(ValueError, self.fnode.read)


    def test04_Readline(self):
        "Reading a line from a closed file."

        self.assertRaises(ValueError, self.fnode.readline)


    def test05_Readlines(self):
        "Reading lines from a closed file."

        self.assertRaises(ValueError, self.fnode.readlines)


    def test06_Seek(self):
        "Seeking a closed file."

        self.assertRaises(ValueError, self.fnode.seek, 0)


    def test07_Tell(self):
        "Getting the pointer position in a closed file."

        self.assertRaises(ValueError, self.fnode.tell)


    def test08_Truncate(self):
        "Truncating a closed file."

        self.assertRaises(ValueError, self.fnode.truncate)


    def test09_Write(self):
        "Writing a closed file."

        self.assertRaises(ValueError, self.fnode.write, 'foo')


    def test10_Writelines(self):
        "Writing lines to a closed file."

        self.assertRaises(ValueError, self.fnode.writelines, ['foo\n'])



def copyFileToFile(srcfile, dstfile, blocksize = 4096):
    """copyFileToFile(srcfile, dstfile[, blocksize]) -> None

    Copies a readable opened file 'srcfile' to a writable opened file 'destfile'
    in blocks of 'blocksize' bytes (4 KiB by default).
    """

    data = srcfile.read(blocksize)
    while len(data) > 0:
        dstfile.write(data)
        data = srcfile.read(blocksize)



class WriteFileTestCase(common.TempFileMixin, common.PyTablesTestCase):
    "Tests writing, seeking and truncating a new file node."

    datafname = 'test_filenode.dat'


    def setUp(self):
        """setUp() -> None

        This method sets the following instance attributes:
          * 'h5fname', the name of the temporary HDF5 file
          * 'h5file', the writable, temporary HDF5 file with a '/test' node
          * 'fnode', the writable file node in '/test'
        """
        super(WriteFileTestCase, self).setUp()
        self.fnode = filenode.newNode(self.h5file, where = '/', name = 'test')
        self.datafname = self._testFilename(self.datafname)


    def tearDown(self):
        """tearDown() -> None

        Closes 'fnode' and 'h5file'; removes 'h5fname'.
        """
        self.fnode.close()
        self.fnode = None
        super(WriteFileTestCase, self).tearDown()


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



class OpenFileTestCase(common.TempFileMixin, common.PyTablesTestCase):
    "Tests opening an existing file node for reading and writing."

    def setUp(self):
        """setUp() -> None

        This method sets the following instance attributes:
          * 'h5fname', the name of the temporary HDF5 file
          * 'h5file', the writable, temporary HDF5 file with a '/test' node
        """
        super(OpenFileTestCase, self).setUp()
        fnode = filenode.newNode(self.h5file, where = '/', name = 'test')
        fnode.close()


    def test00_OpenFileRead(self):
        "Opening an existing file node for reading."

        node = self.h5file.getNode('/test')
        fnode = filenode.openNode(node)
        self.assertEqual(
                fnode.node, node, "filenode.openNode() opened the wrong node.")
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
        fnode = filenode.openNode(node, 'a+')
        self.assertEqual(
                fnode.node, node, "filenode.openNode() opened the wrong node.")
        self.assertEqual(
                fnode.mode, 'a+',
                "File was opened with an invalid mode %s." % repr(fnode.mode))

        self.assertEqual(
                fnode.tell(), 0L,
                "Pointer is not positioned at the beginning of the file.")
        fnode.close()


    def test02_OpenFileInvalidMode(self):
        "Opening an existing file node with an invalid mode."

        self.assertRaises(
                IOError, filenode.openNode, self.h5file.getNode('/test'), 'w')


    # This no longer works since type and type version attributes
    # are now system attributes.  ivb(2004-12-29)
    ##def test03_OpenFileNoAttrs(self):
    ##      "Opening a node with no type attributes."
    ##
    ##      node = self.h5file.getNode('/test')
    ##      self.h5file.delNodeAttr('/test', '_type')
    ##      # Another way to get the same result is changing the value.
    ##      ##self.h5file.setNodeAttr('/test', '_type', 'foobar')
    ##      self.assertRaises(ValueError, filenode.openNode, node)



class ReadFileTestCase(common.TempFileMixin, common.PyTablesTestCase):
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

        self.datafname = self._testFilename(self.datafname)
        self.datafile = file(self.datafname)

        super(ReadFileTestCase, self).setUp()

        fnode = filenode.newNode(self.h5file, where = '/', name = 'test')
        copyFileToFile(self.datafile, fnode)
        fnode.close()

        self.datafile.seek(0)
        self.fnode = filenode.openNode(self.h5file.getNode('/test'))


    def tearDown(self):
        """tearDown() -> None

        Closes 'fnode', 'h5file' and 'datafile'; removes 'h5fname'.
        """

        self.fnode.close()
        self.fnode = None

        super(ReadFileTestCase, self).tearDown()

        self.datafile.close()
        self.datafile = None


    def test00_CompareFile(self):
        "Reading and comparing a whole file node."

        # Try to use hashlib (included from Python 2.5 on)
        try:
            import hashlib
            dfiledigest = hashlib.md5(self.datafile.read()).digest()
            fnodedigest = hashlib.md5(self.fnode.read()).digest()
        except ImportError:
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



class ReadlineTestCase(common.TempFileMixin, common.PyTablesTestCase):
    """
    Base class for text line-reading test cases.

    It provides a set of tests independent of the line separator string.
    Sub-classes must provide the 'lineSeparator' attribute.
    """

    def setUp(self):
        """
        This method sets the following instance attributes:

        * ``h5fname``: the name of the temporary HDF5 file.
        * ``h5file``: the writable, temporary HDF5 file with a ``/test`` node.
        * ``fnode``: the readable file node in ``/test``, with text in it.
        """

        super(ReadlineTestCase, self).setUp()

        linesep = self.lineSeparator

        # Fill the node file with some text.
        fnode = filenode.newNode(self.h5file, where = '/', name = 'test')
        fnode.lineSeparator = linesep
        fnode.write(linesep)
        fnode.write('short line%sshort line%s%s' % ((linesep,) * 3))
        fnode.write('long line ' * 20 + linesep)
        fnode.write('unterminated')
        fnode.close()

        # Re-open it for reading.
        self.fnode = filenode.openNode(self.h5file.getNode('/test'))
        self.fnode.lineSeparator = linesep


    def tearDown(self):
        """tearDown() -> None

        Closes 'fnode' and 'h5file'; removes 'h5fname'.
        """

        self.fnode.close()
        self.fnode = None
        super(ReadlineTestCase, self).tearDown()


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



class MonoReadlineTestCase(ReadlineTestCase):
    "Tests reading one-byte-separated text lines from an existing file node."

    lineSeparator = '\n'



class MultiReadlineTestCase(ReadlineTestCase):
    "Tests reading multibyte-separated text lines from an existing file node."

    lineSeparator = '<br/>'



class LineSeparatorTestCase(common.TempFileMixin, common.PyTablesTestCase):
    "Tests text line separator manipulation in a file node."

    def setUp(self):
        """setUp() -> None

        This method sets the following instance attributes:
          * 'h5fname', the name of the temporary HDF5 file
          * 'h5file', the writable, temporary HDF5 file with a '/test' node
          * 'fnode', the writable file node in '/test'
        """
        super(LineSeparatorTestCase, self).setUp()
        self.fnode = filenode.newNode(self.h5file, where = '/', name = 'test')


    def tearDown(self):
        """tearDown() -> None

        Closes 'fnode' and 'h5file'; removes 'h5fname'.
        """
        self.fnode.close()
        self.fnode = None
        super(LineSeparatorTestCase, self).tearDown()


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



class AttrsTestCase(common.TempFileMixin, common.PyTablesTestCase):
    "Tests setting and getting file node attributes."

    def setUp(self):
        """setUp() -> None

        This method sets the following instance attributes:
          * 'h5fname', the name of the temporary HDF5 file
          * 'h5file', the writable, temporary HDF5 file with a '/test' node
          * 'fnode', the writable file node in '/test'
        """
        super(AttrsTestCase, self).setUp()
        self.fnode = filenode.newNode(self.h5file, where = '/', name = 'test')


    def tearDown(self):
        """tearDown() -> None

        Closes 'fnode' and 'h5file'; removes 'h5fname'.
        """
        self.fnode.close()
        self.fnode = None
        super(AttrsTestCase, self).tearDown()


    # This no longer works since type and type version attributes
    # are now system attributes.  ivb(2004-12-29)
    ##def test00_GetTypeAttr(self):
    ##      "Getting the type attribute of a file node."
    ##
    ##      self.assertEqual(
    ##              getattr(self.fnode.attrs, '_type', None), filenode.NodeType,
    ##              "File node has no '_type' attribute.")


    def test00_MangleTypeAttrs(self):
        "Mangling the type attributes on a file node."

        nodeType = getattr(self.fnode.attrs, 'NODE_TYPE', None)
        self.assertEqual(
                nodeType, filenode.NodeType,
                "File node does not have a valid 'NODE_TYPE' attribute.")

        nodeTypeVersion = getattr(self.fnode.attrs, 'NODE_TYPE_VERSION', None)
        self.assert_(
                nodeTypeVersion in filenode.NodeTypeVersions,
                "File node does not have a valid 'NODE_TYPE_VERSION' attribute.")

        # System attributes are now writable.  ivb(2004-12-30)
        ##self.assertRaises(
        ##      AttributeError,
        ##      setattr, self.fnode.attrs, 'NODE_TYPE', 'foobar')
        ##self.assertRaises(
        ##      AttributeError,
        ##      setattr, self.fnode.attrs, 'NODE_TYPE_VERSION', 'foobar')

        # System attributes are now removables.  F. Alted (2007-03-06)
#         self.assertRaises(
#                 AttributeError,
#                 delattr, self.fnode.attrs, 'NODE_TYPE')
#         self.assertRaises(
#                 AttributeError,
#                 delattr, self.fnode.attrs, 'NODE_TYPE_VERSION')


    # System attributes are now writable.  ivb(2004-12-30)
    ##def test01_SetSystemAttr(self):
    ##      "Setting a system attribute on a file node."
    ##
    ##      self.assertRaises(
    ##              AttributeError, setattr, self.fnode.attrs, 'CLASS', 'foobar')


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
        self.assertEqual(
                getattr(self.fnode.attrs, 'userAttr', None), None,
                "User attribute was not deleted.")
        # Another way is looking up the attribute in the attribute list.
        ##if 'userAttr' in self.fnode.attrs._f_list():
        ##      self.fail("User attribute was not deleted.")


    def test03_AttrsOnClosedFile(self):
        "Accessing attributes on a closed file node."

        self.fnode.close()
        self.assertRaises(AttributeError, getattr, self.fnode, 'attrs')



class ClosedH5FileTestCase(common.TempFileMixin, common.PyTablesTestCase):
    "Tests accessing a file node in a closed PyTables file."

    def setUp(self):
        """setUp() -> None

        This method sets the following instance attributes:
          * 'h5fname', the name of the temporary HDF5 file
          * 'h5file', the closed HDF5 file with a '/test' node
          * 'fnode', the writable file node in '/test'
        """
        super(ClosedH5FileTestCase, self).setUp()
        self.fnode = filenode.newNode(self.h5file, where = '/', name = 'test')
        self.h5file.close()

    def tearDown(self):
        """tearDown() -> None

        Closes 'fnode'; removes 'h5fname'.
        """

        # ivilata:  We know that a UserWarning will be raised
        #   because the PyTables file has already been closed.
        #   However, we don't want it to pollute the test output.
        warnings.filterwarnings('ignore', category = UserWarning)
        self.fnode.close()
        warnings.filterwarnings('default', category = UserWarning)

        self.fnode = None
        super(ClosedH5FileTestCase, self).tearDown()


    def test00_Write(self):
        "Writing to a file node in a closed PyTables file."

        self.assertRaises(ValueError, self.fnode.write, 'data')


    def test01_Attrs(self):
        "Accessing the attributes of a file node in a closed PyTables file."

        self.assertRaises(ValueError, getattr, self.fnode, 'attrs')



class OldVersionTestCase(common.PyTablesTestCase):
    """
    Base class for old version compatibility test cases.

    It provides some basic tests for file operations and attribute handling.
    Sub-classes must provide the 'oldversion' attribute
    and the 'oldh5fname' attribute.
    """

    def setUp(self):
        """
        This method sets the following instance attributes:

        * ``h5fname``: the name of the temporary HDF5 file.
        * ``h5file``: the writable, temporary HDF5 file with a ``/test`` node.
        * ``fnode``: the readable file node in ``/test``.
        """

        self.h5fname = tempfile.mktemp(suffix = '.h5')

        self.oldh5fname = self._testFilename(self.oldh5fname)
        oldh5f = tables.openFile(self.oldh5fname)
        oldh5f.copyFile(self.h5fname)
        oldh5f.close()

        self.h5file = tables.openFile(
                self.h5fname, 'r+',
                title = "Test for file node old version compatibility")
        self.fnode = filenode.openNode(self.h5file.root.test, 'a+')


    def tearDown(self):
        """Closes ``fnode`` and ``h5file``; removes ``h5fname``."""

        self.fnode.close()
        self.fnode = None
        self.h5file.close()
        self.h5file = None
        os.remove(self.h5fname)


    def test00_Read(self):
        "Reading an old version file node."

        self.fnode.lineSeparator = '\n'

        line = self.fnode.readline()
        self.assertEqual(line, 'This is only\n')

        line = self.fnode.readline()
        self.assertEqual(line, 'a test file\n')

        line = self.fnode.readline()
        self.assertEqual(line, 'for FileNode version %d\n' % self.oldversion)

        line = self.fnode.readline()
        self.assertEqual(line, '')

        self.fnode.seek(0)
        line = self.fnode.readline()
        self.assertEqual(line, 'This is only\n')


    def test01_Write(self):
        "Writing an old version file node."

        self.fnode.lineSeparator = '\n'

        self.fnode.write('foobar\n')
        self.fnode.seek(-7, 2)
        line = self.fnode.readline()
        self.assertEqual(line, 'foobar\n')


    def test02_Attributes(self):
        "Accessing attributes in an old version file node."

        self.fnode.attrs.userAttr = 'foobar'
        self.assertEqual(
                getattr(self.fnode.attrs, 'userAttr', None), 'foobar',
                "User attribute was not correctly set.")

        self.fnode.attrs.userAttr = 'bazquux'
        self.assertEqual(
                getattr(self.fnode.attrs, 'userAttr', None), 'bazquux',
                "User attribute was not correctly changed.")

        del self.fnode.attrs.userAttr
        self.assertEqual(
                getattr(self.fnode.attrs, 'userAttr', None), None,
                "User attribute was not deleted.")



class Version1TestCase(OldVersionTestCase):
    "Basic test for version 1 format compatibility."

    oldversion = 1
    oldh5fname = 'test_filenode_v1.h5'



#----------------------------------------------------------------------

def suite():
    """suite() -> test suite

    Returns a test suite consisting of all the test cases in the module.
    """

    theSuite = unittest.TestSuite()

    theSuite.addTest(unittest.makeSuite(NewFileTestCase))
    theSuite.addTest(unittest.makeSuite(ClosedFileTestCase))
    theSuite.addTest(unittest.makeSuite(WriteFileTestCase))
    theSuite.addTest(unittest.makeSuite(OpenFileTestCase))
    theSuite.addTest(unittest.makeSuite(ReadFileTestCase))
    theSuite.addTest(unittest.makeSuite(MonoReadlineTestCase))
    theSuite.addTest(unittest.makeSuite(MultiReadlineTestCase))
    theSuite.addTest(unittest.makeSuite(LineSeparatorTestCase))
    theSuite.addTest(unittest.makeSuite(AttrsTestCase))
    theSuite.addTest(unittest.makeSuite(ClosedH5FileTestCase))
    if common.numarray_imported:
        theSuite.addTest(unittest.makeSuite(Version1TestCase))

    return theSuite


if __name__ == '__main__':
    unittest.main(defaultTest = 'suite')



## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## End:
