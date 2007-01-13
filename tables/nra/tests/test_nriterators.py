import unittest

from tables.tests import common
from tables.nra import nriterators

class NRIteratorsTests(unittest.TestCase):
    """Define a set of unit tests for the nriterators module.

    The tests will be run  for a hypothetical nested table whith the
    following structure:

    #############################################################
    #           #                    INFO                       #
    #           #################################################
    # POSITION  #           NAME        #         COORD         #
    #           #################################################
    #           #   FIRST   #   SECOND  #   X   #   Y   #   Z   #
    #############################################################

    """

    def setUp(self):
        """Set up the unit tests execution environment.
        """

        # A row of the sample nested/flat table
        self.row = [1, (('Paco', 'Perez'), (10, 20, 30))]

        self.flat_row = [1, 'Paco', 'Perez', 10, 20, 30]

        # A buffer
        self.row1 = [2, (('Maria', 'Luisa'), (0, 2.0, 10))]
        self.row2 = [3, (('C3Peanut', 'Tofu'), (10, 30, 20))]
        self.buffer = [self.row, self.row1, self.row2]

        # An arrays list equivalent to buffer
        self.arrays_list = [[1, 2, 3],
            [(('Paco', 'Perez'), (10, 20, 30)),
            (('Maria', 'Luisa'), (0, 2.0, 10)),
            (('C3Peanut', 'Tofu'), (10, 30, 20))]]

        # Names description of the nested/flat table structure
        self.names = ['position', ('info', [('name', ['first', 'second']),
            ('coord', ['x', 'y', 'z'])])]

        self.flat_names = ['position', 'first', 'second', 'x', 'y', 'z']
        self.subnames = ['position', 'info', 'name', 'first', 'second',
            'coord', 'x', 'y', 'z']

        # Formats description of the nested/flat table structure
        self.formats = ['Int64', [['a5', 'a5'],['Float32', 'f4', 'f4']]]

        self.flat_formats = ['Int64', 'a5', 'a5', 'Float32', 'f4', 'f4']

        # descr description of the nested/flat table structure
        self.descr = [('position', 'Int64'), ('info', [
            ('name', [('first','a5'), ('second','a5')]),
           ('coord', [('x','Float32'), ('y', 'f4'), ('z', 'f4')])])]

        self.flat_descr = [('position', 'Int64'), ('first','a5'),
            ('second','a5'), ('x','Float32'), ('y', 'f4'), ('z', 'f4')]

        # descr description of the table structure using automatic names
        self.autoNamedDescr = [('c1', 'Int64'), ('c2', [
            ('c1', [('c1','a5'), ('c2','a5')]),
           ('c2', [('c1','Float32'), ('c2', 'f4'), ('c3', 'f4')])])]

        self.infoDescr = \
            [('info', [('name', [('first','a5'), ('second','a5')]),
           ('coord', [('x','Float32'), ('y', 'f4'), ('z', 'f4')])])]


    #
    # Tests for the nriterators module
    #
    def testDescrStructure(self):
        """Check structure of the descr list.
        """

        common.verbosePrint('\nTesting descr structure')
        for item in nriterators.flattenDescr(self.descr, check=True):
            self.failIfEqual(item, None)


    def testFormatsStructure(self):
        """Check the structure of the formats list.
        """

        common.verbosePrint('\nTesting formats structure')
        for f in nriterators.flattenFormats(self.formats, check=True):
            self.failIfEqual(f, None)


    def testNamesStructure(self):
        """Check the structure of the names list.
        """

        common.verbosePrint('\nTesting names structure')
        for item in nriterators.flattenNames(self.names):
            self.failIfEqual(item, None)


    def testMakeDescr(self):
        """Check the generation of a descr from formats and names.
        """

        common.verbosePrint('\nTesting getDescr function')
        mix = [f for f in nriterators.getDescr(None, self.formats)]
        self.assertEqual(mix, self.autoNamedDescr)
        mix = \
            [f for f in nriterators.getDescr(self.names, self.formats)]
        self.assertEqual(mix, self.descr)


    def testNamesFromDescr(self):
        """Retrieves the names list from the descr list.
        """

        # Check getNamesFromDescr function
        common.verbosePrint('\nTesting getNamesFromDescr function')
        new_names = \
            [item for item in nriterators.getNamesFromDescr(self.descr)]
        self.assertEqual(self.names, new_names)


    def testFormatsFromDescr(self):
        """Retrieves the formats list from the descr list.
        """

        # Check getFormatsFromDescr function
        new_formats = \
            [f for f in nriterators.getFormatsFromDescr(self.descr)]
        self.assertEqual(self.formats, new_formats)


    def testGetFieldDescr(self):
        """Check the getFieldDescr function.
        """

        fieldName = 'info'
        infoDescr = [fd for fd in nriterators.getFieldDescr(fieldName,
            self.descr)]
        self.assertEqual(infoDescr, self.infoDescr)


    def testSubFieldNames(self):
        """Check the syntax of the names list components.
        """

        common.verbosePrint('\nTesting names list decomposition')
        subnames = [sn for sn in nriterators.getSubNames(self.names)]
        self.assertEqual(subnames, self.subnames)


    def testNamesUniqueness(self):
        """Check that names are unique at every level of the names list.
        """

        common.verbosePrint('\nTesting checkNamesUniqueness')
        foo = nriterators.checkNamesUniqueness
        badNames = ['info',
        ('info', [('name', ['first', 'second']),
                                ('coord', ['x', 'y', 'z'])])]
        self.assertRaises(ValueError, foo, badNames)
        badNames = ['position',
        ('info', [('coord', ['first', 'second']),
                                ('coord', ['x', 'y', 'z'])])]
        self.assertRaises(ValueError, foo, badNames)
        badNames = ['position',
        ('info', [('name', ['first', 'second']),
                                ('coord', ['x', 'x', 'z'])])]
        self.assertRaises(ValueError, foo, badNames)
        goodNames = ['position',
        ('info', [('info', ['first', 'second']),
                                ('coord', ['x', 'y', 'z'])])]
        self.assertEqual(foo(goodNames), None)


    def testBufferStructureWDescr(self):
        """Check the structure of a buffer row using the descr list.
        """

        common.verbosePrint('\nTesting buffer row structure with zipBufferDescr')
        mix = [item for item in nriterators.zipBufferDescr(self.row,
            self.descr)]
        self.assertEqual(mix, zip(self.flat_row, self.flat_formats))


def suite():
    """Return a test suite consisting of all the test cases in the module."""

    theSuite = unittest.TestSuite()
    theSuite.addTest(unittest.makeSuite(NRIteratorsTests))
    return theSuite


if __name__ == '__main__':
    unittest.main(defaultTest='NRIteratorsTests' )
