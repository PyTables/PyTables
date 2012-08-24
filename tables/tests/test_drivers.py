import os
import unittest
import tempfile

from tables import *
from tables.tests import common
import tables.parameters


class FileDriverTestCase(common.PyTablesTestCase):
    file = None
    DRIVER = None

    def test00_newFile(self):
        self.file = tempfile.mktemp(".h5")
        fileh = openFile(self.file, mode="w", title="File title",
                         DRIVER=self.DRIVER)

        # Create an HDF5 file
        root = fileh.root

        # Create an array
        fileh.createArray(root, 'array', [1, 2], title="Array example")
        fileh.createTable(root, 'table', {'var1': IntCol()}, "Table example")
        root._v_attrs.testattr = 41
        fileh.close()

        try:
            # Checking opening of an existing file
            fileh = openFile(self.file, mode="r", title="File title",
                             DRIVER=self.DRIVER)

            # Get the CLASS attribute of the arr object
            title = fileh.root.array.getAttr("TITLE")

            self.assertEqual(title, "Array example")
            fileh.close()
        except:
            self.fail("Can't open file for reading with DRIVER=" + self.DRIVER)
        finally:
                # Remove the temporary file
            try:
                os.remove(self.file)
            except OSError:
                pass
            self.fileh = None


class DefaultDriverTestCase(FileDriverTestCase):
    DRIVER = None


class SEC2DriverTestCase(FileDriverTestCase):
    DRIVER = "H5FD_SEC2"


class STDIODriverTestCase(FileDriverTestCase):
    DRIVER = "H5FD_STDIO"


class COREDriverTestCase(FileDriverTestCase):
    DRIVER = "H5FD_CORE"


class CORE_INMEMORYDriverTestCase(common.PyTablesTestCase):
    DRIVER = "H5FD_CORE_INMEMORY"

    def _create_image(self, filename="in-memory", title="Title", mode='w'):
        fileh = openFile(filename, mode=mode, title=title,
                         DRIVER=self.DRIVER, H5FD_CORE_BACKING_STORE=0)

        fileh.createArray(fileh.root, 'array', [1, 2], title="Array")
        fileh.createTable(fileh.root, 'table', {'var1': IntCol()}, "Table")
        fileh.root._v_attrs.testattr = 41

        fileh.close()

        return fileh.getInMemoryFileContents()

    def test_newFileW(self):
        filename = tempfile.mktemp(".h5")
        image = self._create_image(filename, mode='w')
        self.assertTrue(len(image) > 0)
        self.assertEqual([ord(i) for i in image[:4]], [137, 72, 68, 70])
        self.assertFalse(os.path.exists(filename))

    def test_newFileA(self):
        filename = tempfile.mktemp(".h5")
        image = self._create_image(filename, mode='a')
        self.assertTrue(len(image) > 0)
        self.assertEqual([ord(i) for i in image[:4]], [137, 72, 68, 70])
        self.assertFalse(os.path.exists(filename))

    def test_openFileR(self):
        filename = tempfile.mktemp(".h5")
        image = self._create_image(filename)
        self.assertFalse(os.path.exists(filename))

        # Open an existing file
        fileh = openFile(filename, mode="r",
                         DRIVER=self.DRIVER,
                         H5FD_CORE_INMEMORY_IMAGE=image,
                         H5FD_CORE_BACKING_STORE=0)

        # Get the CLASS attribute of the arr object
        self.assertTrue(hasattr(fileh.root._v_attrs, "TITLE"))
        self.assertEqual(fileh.getNodeAttr("/", "TITLE"), "Title")
        self.assertTrue(hasattr(fileh.root._v_attrs, "testattr"))
        self.assertEqual(fileh.getNodeAttr("/", "testattr"), 41)
        self.assertTrue(hasattr(fileh.root, "array"))
        self.assertEqual(fileh.getNodeAttr("/array", "TITLE"), "Array")
        self.assertTrue(hasattr(fileh.root, "table"))
        self.assertEqual(fileh.getNodeAttr("/table", "TITLE"), "Table")
        self.assertEqual(fileh.root.array.read(), [1, 2])

        fileh.close()

    def test_openFileRW(self):
        filename = tempfile.mktemp(".h5")
        image = self._create_image(filename)
        self.assertFalse(os.path.exists(filename))

        # Open an existing file
        fileh = openFile(filename, mode="r+",
                         DRIVER=self.DRIVER,
                         H5FD_CORE_INMEMORY_IMAGE=image,
                         H5FD_CORE_BACKING_STORE=0)

        # Get the CLASS attribute of the arr object
        self.assertTrue(hasattr(fileh.root._v_attrs, "TITLE"))
        self.assertEqual(fileh.getNodeAttr("/", "TITLE"), "Title")
        self.assertTrue(hasattr(fileh.root._v_attrs, "testattr"))
        self.assertEqual(fileh.getNodeAttr("/", "testattr"), 41)
        self.assertTrue(hasattr(fileh.root, "array"))
        self.assertEqual(fileh.getNodeAttr("/array", "TITLE"), "Array")
        self.assertTrue(hasattr(fileh.root, "table"))
        self.assertEqual(fileh.getNodeAttr("/table", "TITLE"), "Table")
        self.assertEqual(fileh.root.array.read(), [1, 2])

        fileh.createArray(fileh.root, 'array2', range(10000), title="Array2")
        fileh.root._v_attrs.testattr2 = 42

        fileh.close()

        self.assertFalse(os.path.exists(filename))

    def test_openFileRW_update(self):
        filename = tempfile.mktemp(".h5")
        image1 = self._create_image(filename)
        self.assertFalse(os.path.exists(filename))

        # Open an existing file
        fileh = openFile(filename, mode="r+",
                         DRIVER=self.DRIVER,
                         H5FD_CORE_INMEMORY_IMAGE=image1,
                         H5FD_CORE_BACKING_STORE=0)

        # Get the CLASS attribute of the arr object
        self.assertTrue(hasattr(fileh.root._v_attrs, "TITLE"))
        self.assertEqual(fileh.getNodeAttr("/", "TITLE"), "Title")
        self.assertTrue(hasattr(fileh.root._v_attrs, "testattr"))
        self.assertEqual(fileh.getNodeAttr("/", "testattr"), 41)
        self.assertTrue(hasattr(fileh.root, "array"))
        self.assertEqual(fileh.getNodeAttr("/array", "TITLE"), "Array")
        self.assertTrue(hasattr(fileh.root, "table"))
        self.assertEqual(fileh.getNodeAttr("/table", "TITLE"), "Table")
        self.assertEqual(fileh.root.array.read(), [1, 2])

        data = range(2 * tables.parameters.H5FD_CORE_INCREMENT)
        fileh.createArray(fileh.root, 'array2', data, title="Array2")
        fileh.root._v_attrs.testattr2 = 42

        fileh.close()

        self.assertFalse(os.path.exists(filename))

        image2 = fileh.getInMemoryFileContents()

        self.assertNotEqual(len(image1), len(image2))
        self.assertNotEqual(image1, image2)

        # Open an existing file
        fileh = openFile(filename, mode="r",
                         DRIVER=self.DRIVER,
                         H5FD_CORE_INMEMORY_IMAGE=image2,
                         H5FD_CORE_BACKING_STORE=0)

        # Get the CLASS attribute of the arr object
        self.assertTrue(hasattr(fileh.root._v_attrs, "TITLE"))
        self.assertEqual(fileh.getNodeAttr("/", "TITLE"), "Title")
        self.assertTrue(hasattr(fileh.root._v_attrs, "testattr"))
        self.assertEqual(fileh.getNodeAttr("/", "testattr"), 41)
        self.assertTrue(hasattr(fileh.root, "array"))
        self.assertEqual(fileh.getNodeAttr("/array", "TITLE"), "Array")
        self.assertTrue(hasattr(fileh.root, "table"))
        self.assertEqual(fileh.getNodeAttr("/table", "TITLE"), "Table")
        self.assertEqual(fileh.root.array.read(), [1, 2])

        self.assertTrue(hasattr(fileh.root._v_attrs, "testattr2"))
        self.assertEqual(fileh.getNodeAttr("/", "testattr2"), 42)
        self.assertTrue(hasattr(fileh.root, "array2"))
        self.assertEqual(fileh.getNodeAttr("/array2", "TITLE"), "Array2")
        self.assertEqual(fileh.root.array2.read(), data)

        fileh.close()

        self.assertFalse(os.path.exists(filename))

    def test_openFileA(self):
        filename = tempfile.mktemp(".h5")
        image = self._create_image(filename=filename)
        self.assertFalse(os.path.exists(filename))

        # Open an existing file
        fileh = openFile(filename, mode="a",
                         DRIVER=self.DRIVER,
                         H5FD_CORE_INMEMORY_IMAGE=image,
                         H5FD_CORE_BACKING_STORE=0)

        # Get the CLASS attribute of the arr object
        self.assertTrue(hasattr(fileh.root._v_attrs, "TITLE"))
        self.assertEqual(fileh.getNodeAttr("/", "TITLE"), "Title")
        self.assertTrue(hasattr(fileh.root._v_attrs, "testattr"))
        self.assertEqual(fileh.getNodeAttr("/", "testattr"), 41)
        self.assertTrue(hasattr(fileh.root, "array"))
        self.assertEqual(fileh.getNodeAttr("/array", "TITLE"), "Array")
        self.assertTrue(hasattr(fileh.root, "table"))
        self.assertEqual(fileh.getNodeAttr("/table", "TITLE"), "Table")
        self.assertEqual(fileh.root.array.read(), [1, 2])

        fileh.close()

        self.assertFalse(os.path.exists(filename))

    def test_openFileA_update(self):
        filename = tempfile.mktemp(".h5")
        image1 = self._create_image(filename)
        self.assertFalse(os.path.exists(filename))

        # Open an existing file
        fileh = openFile(filename, mode="a",
                         DRIVER=self.DRIVER,
                         H5FD_CORE_INMEMORY_IMAGE=image1,
                         H5FD_CORE_BACKING_STORE=0)

        # Get the CLASS attribute of the arr object
        self.assertTrue(hasattr(fileh.root._v_attrs, "TITLE"))
        self.assertEqual(fileh.getNodeAttr("/", "TITLE"), "Title")
        self.assertTrue(hasattr(fileh.root._v_attrs, "testattr"))
        self.assertEqual(fileh.getNodeAttr("/", "testattr"), 41)
        self.assertTrue(hasattr(fileh.root, "array"))
        self.assertEqual(fileh.getNodeAttr("/array", "TITLE"), "Array")
        self.assertTrue(hasattr(fileh.root, "table"))
        self.assertEqual(fileh.getNodeAttr("/table", "TITLE"), "Table")
        self.assertEqual(fileh.root.array.read(), [1, 2])

        data = range(2 * tables.parameters.H5FD_CORE_INCREMENT)
        fileh.createArray(fileh.root, 'array2', data, title="Array2")
        fileh.root._v_attrs.testattr2 = 42

        fileh.close()

        self.assertFalse(os.path.exists(filename))

        image2 = fileh.getInMemoryFileContents()

        self.assertNotEqual(len(image1), len(image2))
        self.assertNotEqual(image1, image2)

        # Open an existing file
        fileh = openFile(filename, mode="r",
                         DRIVER=self.DRIVER,
                         H5FD_CORE_INMEMORY_IMAGE=image2,
                         H5FD_CORE_BACKING_STORE=0)

        # Get the CLASS attribute of the arr object
        self.assertTrue(hasattr(fileh.root._v_attrs, "TITLE"))
        self.assertEqual(fileh.getNodeAttr("/", "TITLE"), "Title")
        self.assertTrue(hasattr(fileh.root._v_attrs, "testattr"))
        self.assertEqual(fileh.getNodeAttr("/", "testattr"), 41)
        self.assertTrue(hasattr(fileh.root, "array"))
        self.assertEqual(fileh.getNodeAttr("/array", "TITLE"), "Array")
        self.assertTrue(hasattr(fileh.root, "table"))
        self.assertEqual(fileh.getNodeAttr("/table", "TITLE"), "Table")
        self.assertEqual(fileh.root.array.read(), [1, 2])

        self.assertTrue(hasattr(fileh.root._v_attrs, "testattr2"))
        self.assertEqual(fileh.getNodeAttr("/", "testattr2"), 42)
        self.assertTrue(hasattr(fileh.root, "array2"))
        self.assertEqual(fileh.getNodeAttr("/array2", "TITLE"), "Array2")
        self.assertEqual(fileh.root.array2.read(), data)

        fileh.close()

        self.assertFalse(os.path.exists(filename))

    def test_flush(self):
        filename = tempfile.mktemp(".h5")
        fileh = openFile(filename, mode="w", title="Title",
                 DRIVER=self.DRIVER, H5FD_CORE_BACKING_STORE=0)

        fileh.createArray(fileh.root, 'array', [1, 2], title="Array")
        fileh.createTable(fileh.root, 'table', {'var1': IntCol()}, "Table")
        fileh.root._v_attrs.testattr = 41

        fileh.flush()

        image1 = fileh.getInMemoryFileContents()

        self.assertTrue(len(image1) > 0)
        self.assertEqual([ord(i) for i in image1[:4]], [137, 72, 68, 70])

        fileh.close()

        image2 = fileh.getInMemoryFileContents()

        self.asserrtEqual(len(image1), len(image2))
        self.asserrtEqual(image1, image2)
        self.assertFalse(os.path.exists(filename))

    def test_str(self):
        filename = tempfile.mktemp(".h5")
        fileh = openFile(filename, mode="w", title="Title",
                 DRIVER=self.DRIVER, H5FD_CORE_BACKING_STORE=0)

        fileh.createArray(fileh.root, 'array', [1, 2], title="Array")
        fileh.createTable(fileh.root, 'table', {'var1': IntCol()}, "Table")
        fileh.root._v_attrs.testattr = 41

        # ensure that the __str__ method works even if there is no phisical
        # file on disk (in which case the os.stat operation for date retrieval
        # fails)
        self.assertTrue(str(fileh) is not None)

        fileh.close()
        self.assertFalse(os.path.exists(filename))


#----------------------------------------------------------------------
def suite():
    theSuite = unittest.TestSuite()
    theSuite.addTest(unittest.makeSuite(FileDriverTestCase))
    theSuite.addTest(unittest.makeSuite(SEC2DriverTestCase))
    theSuite.addTest(unittest.makeSuite(STDIODriverTestCase))
    theSuite.addTest(unittest.makeSuite(COREDriverTestCase))
    theSuite.addTest(unittest.makeSuite(CORE_INMEMORYDriverTestCase))
    return theSuite


if __name__ == '__main__':
    unittest.main(defaultTest='suite')

## Local Variables:
## mode: python
## End:
