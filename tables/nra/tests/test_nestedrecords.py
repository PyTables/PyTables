# This test suite is still necessary in case the user still wants a
# nested RecArray which is based in numarray instead of NumPy.

import unittest

import numarray
import numarray.records

from tables.tests import common
from tables import nra

class NestedRecordTests(common.PyTablesTestCase):
    """Define a set of unit tests for the nestedrecords module.

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
        self.row = [1,
            (('Paco', 'Perez'), (10, 20, 30))]

        self.flat_row = [1, 'Paco', 'Perez', 10, 20, 30]

        # A buffer
        self.row1 = [2,
            (('Maria', 'Luisa'), (0, 2.0, 10))]
        self.row2 = [3,
            (('C3Peanut', 'Tofu'), (10, 30, 20))]
        self.buffer = [self.row, self.row1, self.row2]
        self.flat_buffer = [[1, 'Paco', 'Perez', 10, 20, 30],
            [2, 'Maria', 'Luisa', 0, 2.0, 10],
            [3, 'C3Peanut', 'Tofu', 10, 30, 20]]

        # An array list equivalent to buffer
        self.array_list = [[1, 2, 3],
            [[('Paco', 'Maria', 'C3Peanut'), ('Perez', 'Luisa', 'Tofu')],
            [[10, 0, 10], [20, 2.0, 30], [30, 10, 20]]]]

        # Names description of the nested/flat table structure
        self.names = ['position',
            ('info', [('name', ['first', 'second']),
                                    ('coord', ['x', 'y', 'z'])])]

        self.flat_names = ['position', 'first', 'second', 'x', 'y', 'z']

        # Formats description of the nested/flat table structure
        self.formats = ['Int64',
            [['S9', '()a9'],['()Float32', 'f4', 'f4']]]

        self.flat_formats = ['Int64', 'a9', 'a9', 'Float32', 'f4', 'f4']

        # descr description of the nested/flat table structure
        self.descr = [('position', 'Int64'),
            ('info', [('name', [('first','S9'), ('second','()a9')]),
            ('coord', [('x','()Float32'), ('y', 'f4'), ('z', 'f4')])])]

        self.flat_descr = [('position', 'Int64'), ('first','S9'),
            ('second','()a9'), ('x','()Float32'), ('y', 'f4'), ('z', 'f4')]


    def testArrayStructure(self):
        """Check the isThereStructure function.
        """

        common.verbosePrint( '\nTesting array structure check function')
        common.verbosePrint( 'With descr description...')
        cse = \
            nra.nestedrecords._isThereStructure( None, self.descr,
            self.buffer)
        self.assertEqual(cse, None)

        common.verbosePrint( 'With formats description...')
        cse = \
            nra.nestedrecords._isThereStructure(self.formats, None,
            self.buffer)
        self.assertEqual(cse, None)

        common.verbosePrint( 'With no description...')
        self.assertRaises(NotImplementedError,
            nra.nestedrecords._isThereStructure, None, None, self.buffer)
        self.assertRaises(ValueError, nra.nestedrecords._isThereStructure,
            None, None, None)


    def testArrayUniqueSyntax(self):
        """Check the onlyOneSyntax function.
        """

        common.verbosePrint( '\nTesting the uniqueness of the array syntax')
        self.assertEqual(nra.nestedrecords._onlyOneSyntax(self.descr, None,
            None), None)
        self.assertEqual(nra.nestedrecords._onlyOneSyntax(None,
            self.formats, None), None)
        self.assertRaises(ValueError, nra.nestedrecords._onlyOneSyntax,
            self.descr, self.formats, None)
        self.assertRaises(ValueError, nra.nestedrecords._onlyOneSyntax,
            self.descr, None, self.names)


    def testArrayFormats(self):
        """Check the checkFormats function.
        """

        common.verbosePrint( '\nTesting samples of formats description')
        formats = 'formats should be a list'
        self.assertRaises(TypeError, nra.nestedrecords._checkFormats,
            formats)
        # Formats must be a list of strings or sequences
        formats = [25,
            [['a5', 'a5'],['Float32', 'f4', 'f4']]]
        self.assertRaises(TypeError, nra.nestedrecords._checkFormats,
            formats)
        # If formats is OK checkFormats returns None
        self.assertEqual(nra.nestedrecords._checkFormats(self.formats),
            None)


    def testArrayNames(self):
        """Check the checkNames function.
        """

        common.verbosePrint( '\nTesting samples of names description')
        names = 'names should be a list'
        self.assertRaises(TypeError, nra.nestedrecords._checkNames,
            names)
        # Names must be a list of strings or 2-tuples
        names = [25,
            ('info', [('name', ['first', 'second']),
                                ('coord', ['x', 'y', 'z'])])]
        self.assertRaises(TypeError, nra.nestedrecords._checkNames, names)

        # Names must be unique at any given level
        names = ['position',
            ('info', [('name', ['first', 'second']),
                                ('coord', ['x', 'y', 'y'])])]
        self.assertRaises(ValueError, nra.nestedrecords._checkNames, names)

        # If names is OK checkNames returns None
        self.assertEqual(nra.nestedrecords._checkNames(self.names), None)


    def testArrayDescr(self):
        """Check the checkDescr function.
        """

        common.verbosePrint( '\nTesting samples of descr description')
        # Descr must be a list of 2-tuples
        descr = 'some descr specification'
        self.assertRaises(TypeError, nra.nestedrecords._checkDescr, descr)

        # names in descr must be strings
        # formats must be strings or list of 2-tuples
        descr = [(25, 'Int64'),
            ('info', [('name', [('first','a5'), ('second','a5')]),
                   ('coord', [('x','Float32'), ('y', 'f4'), ('z', 'f4')])])]
        self.assertRaises(TypeError, nra.nestedrecords._checkDescr, descr)

        descr = [('25', 'position', 'Int64'),
            ('info', [('name', [('first','a5'), ('second','a5')]),
                   ('coord', [('x','Float32'), ('y', 'f4'), ('z', 'f4')])])]
        self.assertRaises(TypeError, nra.nestedrecords._checkDescr, descr)

        # If descr is OK checkDescr returns None
        self.assertEqual(nra.nestedrecords._checkDescr(self.descr), None)


    def testFieldsDescr(self):
        """Check the checkFieldsInDescr function.
        """

        common.verbosePrint( """\nTesting the field names syntax in a """
            """sample descr list""")
        descr = [('position', 'Int64'),
            ('info', [('name', [('first','a5'), ('second','a5')]),
               ('coord', [('x/equis','Float32'), ('y', 'f4'), ('z', 'f4')])])]
        self.assertRaises(ValueError, nra.nestedrecords._checkFieldsInDescr,
            descr)


    def testBufferStructure(self):
        """Check the checkBufferStructure function.
        """

        row = [(('Paco', 'Perez'), (10, 20, 30))]
        # A buffer
        buffer = [row, self.row1, self.row2]
        self.assertRaises((ValueError, TypeError),
            nra.nestedrecords._checkBufferStructure, self.descr, buffer,
            nra.nriterators.zipBufferDescr)


    def testCreateNestedRecArray(self):
        """Check the array function.
        """

        flatarray = numarray.records.array(self.flat_buffer,
            self.flat_formats)
        common.verbosePrint( """\nTesting the creation of a nested """
            """recarray: buffer + formats""")
        nra1 = nra.array(formats=self.formats, buffer=self.buffer)
        common.verbosePrint(
            """\nTesting the creation of a nested recarray: buffer + """
            """formats + names""")
        nra2 = nra.array(names=self.names,
            formats=self.formats, buffer=self.buffer)
        common.verbosePrint(
            """\nTesting the creation of a nested recarray: buffer + descr""")
        nra3 = nra.array(descr=self.descr, buffer=self.buffer)

        self.assertEqual(common.areArraysEqual(nra1, nra2), False)

        self.assert_(common.areArraysEqual(nra2, nra3))


    def testNRAFromRA(self):
        """Check the array function with a RecArray instance.
        """

        buffer_ = [('Cuke', 123, (45, 67)), ('Tader', 321, (76, 54))]
        names = ['name', 'value', 'pair']
        formats = ['a6', 'Int8', '(2,)Int16']
        ra = numarray.records.array(
            buffer_, names=names, formats=formats)
##            buffer_, names=names, formats=formats, aligned=True)

        names1 = ['newName', 'newValue', 'newPair']
        nra0 = nra.array(buffer=ra, descr=zip(names1, formats))
        nra1 = nra.array(buffer=buffer_, descr=zip(names1, formats))
        self.assert_(common.areArraysEqual(nra0, nra1))

        # Bad number of fields
        badFormats = ['Int8', '(2,)Int16']
        self.assertRaises(ValueError, nra.array, buffer=ra,
            formats=badFormats)

        # Bad format in the first field
        badFormats = ['a9', 'Int8', '(2,)Int16']
        self.assertRaises(ValueError, nra.array, buffer=ra,
            formats=badFormats)


    def testNRAFromNRA(self):
        """Check the array function with a NestedRecArray instance.
        """

        nra0 = nra.array(buffer=self.buffer, descr=self.descr)
        my_Descr = [('ID', 'Int64'),
            ('data', [('name', [('first','a9'), ('second','a9')]),
            ('coord', [('x','Float32'), ('y', 'f4'), ('z', 'f4')])])]
        nra1 = nra.array(buffer=self.buffer, descr=my_Descr)
        nra2 = nra.array(buffer=nra0, descr=my_Descr)
        self.assert_(common.areArraysEqual(nra2, nra1))

        # Bad number of fields
        badDescr = [
            ('data', [('name', [('first','a9'), ('second','a9')]),
            ('coord', [('x','Float32'), ('y', 'f4'), ('z', 'f4')])])]
        self.assertRaises(ValueError, nra.array, buffer=nra0,
            descr=badDescr)

        # Bad format in the first field
        badDescr = [('ID', 'b1'),
            ('data', [('name', [('first','a9'), ('second','a9')]),
            ('coord', [('x','Float32'), ('y', 'f4'), ('z', 'f4')])])]
        self.assertRaises(ValueError, nra.array, buffer=nra0,
            descr=badDescr)


    #
    # Tests for the nra.fromarrays function
    #
    def testNRAFromArrayList(self):
        """Check the fromarrays function.
        """

        # arrayList argument is a list of lists
        nra0 = nra.array(buffer=self.buffer, descr=self.descr)
        nra1 = nra.fromarrays(self.array_list, formats=self.formats)
        nra2 = nra.fromarrays(self.array_list,
            formats=self.formats, names=self.names)
        nra3 = nra.fromarrays(self.array_list, descr=self.descr)

        self.assertEqual(common.areArraysEqual(nra1, nra2), False)
        self.assert_(common.areArraysEqual(nra2, nra3))
        self.assert_(common.areArraysEqual(nra0, nra2))

        # arrayList argument is a list of NestedRecArrays
        nra0 = nra.array(buffer=[[1,4],[2,4]], formats=['f8','f4'])
        self.assertRaises(TypeError, nra.fromarrays,
            [nra0, nra0.field('c2')], formats=[['f8','f4'],'f4'])


    def testGetSlice(self):
        """Get a nested array slice.
        """

        my_buffer = [[1, (('Cuke', 'Skywalker'), (10, 20, 30))],
            [2, (('Princess', 'Lettuce'), (0, 2.0, 10))],
            [3, (('Ham', 'Solo'), (0, 2.0, 10))],
            [4, (('Obi', 'Cannoli'), (0, 2.0, 10))],
            [5, (('Chew', 'Brocoli'), (0, 2.0, 10))],
            [6, (('Master', 'Yoda'), (0, 2.0, 10))],
            [7, (('Tofu', 'Robot'), (0, 2.0, 10))],
            [8, (('C3Peanut', 'Robot'), (10, 30, 20))]]
        nra0 = nra.array(descr=self.descr, buffer=my_buffer)

        slice_ = nra0[1:2]
        model = nra.array(
            [[2, (('Princess', 'Lettuce'), (0, 2.0, 10))]], descr=self.descr)
        self.assert_(common.areArraysEqual(slice_, model))

        slice_ = nra0[1:4]
        model = nra.array(
            [[2, (('Princess', 'Lettuce'), (0, 2.0, 10))],
            [3, (('Ham', 'Solo'), (0, 2.0, 10))],
            [4, (('Obi', 'Cannoli'), (0, 2.0, 10))]], descr=self.descr)
        self.assert_(common.areArraysEqual(slice_, model))

        slice_ = nra0[1:4:2]
        model = nra.array(
            [[2, (('Princess', 'Lettuce'), (0, 2.0, 10))],
            [4, (('Obi', 'Cannoli'), (0, 2.0, 10))]], descr=self.descr)
        self.assert_(common.areArraysEqual(slice_, model))

        slice_ = nra0[:4]
        model = nra.array(
            [[1, (('Cuke', 'Skywalker'), (10, 20, 30))],
            [2, (('Princess', 'Lettuce'), (0, 2.0, 10))],
            [3, (('Ham', 'Solo'), (0, 2.0, 10))],
            [4, (('Obi', 'Cannoli'), (0, 2.0, 10))]], descr=self.descr)
        self.assert_(common.areArraysEqual(slice_, model))

        slice_ = nra0[:7:3]
        model = nra.array(
            [[1, (('Cuke', 'Skywalker'), (10, 20, 30))],
            [4, (('Obi', 'Cannoli'), (0, 2.0, 10))],
            [7, (('Tofu', 'Robot'), (0, 2.0, 10))]], descr=self.descr)
        self.assert_(common.areArraysEqual(slice_, model))

        slice_ = nra0[:]
        self.assert_(common.areArraysEqual(slice_, nra0))

        slice_ = nra0[::2]
        model = nra.array(
            [[1, (('Cuke', 'Skywalker'), (10, 20, 30))],
            [3, (('Ham', 'Solo'), (0, 2.0, 10))],
            [5, (('Chew', 'Brocoli'), (0, 2.0, 10))],
            [7, (('Tofu', 'Robot'), (0, 2.0, 10))]], descr=self.descr)
        self.assert_(common.areArraysEqual(slice_, model))

        slice_ = nra0[4:-2]
        model = nra.array(
            [[5, (('Chew', 'Brocoli'), (0, 2.0, 10))],
            [6, (('Master', 'Yoda'), (0, 2.0, 10))]], descr=self.descr)
        self.assert_(common.areArraysEqual(slice_, model))

        slice_ = nra0[-1:-3]
        model = nra.array(None, descr=self.descr)
        self.assert_(common.areArraysEqual(slice_, model))

        slice_ = nra0[-5::2]
        model = nra.array(
            [[4, (('Obi', 'Cannoli'), (0, 2.0, 10))],
            [6, (('Master', 'Yoda'), (0, 2.0, 10))],
            [8, (('C3Peanut', 'Robot'), (10, 30, 20))]], descr=self.descr)
        self.assert_(common.areArraysEqual(slice_, model))


    def testSetSlice(self):
        """Set a nested array slice.
        """

        nra0 = nra.array(descr=self.descr, buffer=self.buffer)

        buffer = [
            [10, (('Paco', 'Perez'), (10, 20, 30))],
            [20, (('Maria', 'Luisa'), (0, 2.0, 10))],
            [30, (('C3Peanut', 'Tofu'), (10, 30, 20))]
        ]
        model = nra.array(buffer, descr=self.descr)

        nra0[0:3] = model[0:3]

        self.assert_(common.areArraysEqual(nra0, model))


    def testGetTopLevelFlatField(self):
        """Check the NestedRecArray.field method.
        """

        nra0 = nra.array(descr=self.descr, buffer=self.buffer)

        # Test top level flat fields
        nra1 = nra0.field('position')
        ra1 = numarray.array([1, 2, 3], type='Int64')
        self.assert_(common.areArraysEqual(nra1, ra1))


    def testGetBottomLevelField(self):
        """Check the NestedRecArray.field method.
        """

        nra0 = nra.array(descr=self.descr, buffer=self.buffer)

        # Test bottom level fields
        nra1 = nra0.field('info/coord/x')
        ra1 = numarray.array([10, 0, 10], type='Float32')
        self.assert_(common.areArraysEqual(nra1, ra1))


    def testGetNestedField(self):
        """Check the NestedRecArray.field method.
        """

        nra0 = nra.array(descr=self.descr, buffer=self.buffer)

        # Test top level nested fields
        # The info field
        buffer = [
            [('Paco', 'Perez'), (10, 20, 30)],
            [('Maria', 'Luisa'), (0, 2.0, 10)],
            [('C3Peanut', 'Tofu'), (10, 30, 20)]
        ]
        my_descr = [('name', [('first','a9'), ('second','a9')]),
            ('coord', [('x','Float32'), ('y', 'f4'), ('z', 'f4')])]
        model = nra.array(buffer, descr=my_descr)
        modelFirst = model.field('name/first')

        nra1 = nra0.field('info')
        nra1First = nra1.field('name/first')
        nra2 = nra1.field('name')
        nra3=nra2.field('first')

        self.assert_(common.areArraysEqual(model, nra1))
        self.assert_(common.areArraysEqual(modelFirst, nra1First))
        self.assert_(common.areArraysEqual(modelFirst, nra3))


    def testSetRow2NestedRecord(self):
        """Check the NestedRecArray.__setitem__ with NestedRecord instances.
        """

        nra0 = nra.array(descr=self.descr, buffer=self.buffer)

        buffer = [
            [10, (('Paco', 'Perez'), (10, 20, 30))],
            [20, (('Maria', 'Luisa'), (0, 2.0, 10))],
            [30, (('C3Peanut', 'Tofu'), (10, 30, 20))]
        ]
        model = nra.array(buffer, descr=self.descr)

        nra0[0] = model[0]
        nra0[1] = model[1]
        nra0[2] = model[2]

        self.assert_(common.areArraysEqual(nra0, model))


    def testNRAadd(self):
        """Check the addition of nested arrays.
        """

        ra1 = numarray.records.array([[1, 2], [3, 4]],
            formats=['Int32', 'Int32'])
        ra2 = numarray.records.array([[5, 6], [7, 8]],
            formats=['Int32', 'Int32'])
        ra3 = ra1 + ra2
        nra1 = nra.array(buffer=ra1,
            descr=[('c1', 'Int32'), ('c2', 'Int32')])
        nra2 = nra.array(buffer=ra2,
            descr=[('a', 'Int32'), ('b', 'Int32')])
        nra3 = nra1 + nra2
        nra4 = nra1 + ra1
        self.assert_(common.areArraysEqual(nra3._flatArray, ra3))
        self.assertEqual(nra3.descr, nra1.descr)
        self.assert_(common.areArraysEqual(nra4._flatArray,
            nra1._flatArray + ra1))
        self.assertRaises(TypeError, nra.NestedRecArray.__add__, nra1, 3)


    # NestedRecord tests

    def testNestedRecordCreation(self):
        """Check the creation of NestedRecord instances from NestedRecArrays.
        """

        nra0 = nra.array(descr=self.descr, buffer=self.buffer)

        nrecord = nra0[0]
        self.assert_(isinstance(nrecord, nra.NestedRecord))
        self.assert_(common.areArraysEqual(nra0, nrecord.array))
        self.assertEqual(nrecord.row, 0)


    def testFlattenNestedRecord(self):
        """Check the flattening of NestedRecord instances.
        """

        nra0 = nra.array(descr=self.descr, buffer=self.buffer)
        nrecord = nra0[0]
        frecord = nrecord.asRecord()

        self.assert_(isinstance(frecord, numarray.records.Record))
        self.assert_(common.areArraysEqual(nra0.asRecArray(), frecord.array))
        self.assertEqual(nrecord.row, frecord.row)


    def testNestedRecordFlatField(self):
        """Retrieving flat fields from nested records.
        """

        nra0 = nra.array(descr=self.descr, buffer=self.buffer)

        position = nra0.field('position')[0]
        firstName = nra0.field('info/name/first')[0]

        self.assertEqual(position, 1)
        self.assertEqual(firstName, 'Paco')


    def testNestedRecordSetFlatField(self):
        """Set flat fields of nested records.
        """

        nra0 = nra.array(descr=self.descr, buffer=self.buffer)
        nrecord = nra0[0]
        nrecord.setfield('position', 24)
        nrecord.setfield('info/name/first', 'Joan')

        position = nrecord.field('position')
        firstName = nrecord.field('info/name/first')

        self.assertEqual(position, 24)
        self.assertEqual(firstName, 'Joan')


    def testNestedRecordNestedField(self):
        """Get nested fields from nested records.
        """

        nra0 = nra.array(descr=self.descr, buffer=self.buffer)

        name = nra0.field('info/name')[0]

        self.assertEqual(name.array._names, ['first', 'second'])
        self.assertEqual(name.field('first'), 'Paco')
        self.assertEqual(name.field('second'), 'Perez')


    def testNestedRecordSetNestedField(self):
        """Set nested fields of nested records.
        """

        nra0 = nra.array(descr=self.descr, buffer=self.buffer)
        nrecord = nra0[0]

##        nra2 = nra.array(
##            buffer = [['Joan', 'Clos']],
##            descr = [('first', 'a9'), ('second', 'a9')])

        nra2 = nra.array(
            buffer = [[1, (('Joan', 'Clos'), (10, 20, 30))]],
            descr = self.descr)

##        nrecord.setfield('info/name', nra2[0])
        nrecord.setfield('info', nra2.field('info')[0])

        my_buffer = [[1, (('Joan', 'Clos'), (10, 20, 30))],
            [2, (('Maria', 'Luisa'), (0, 2.0, 10))],
            [3, (('C3Peanut', 'Tofu'), (10.0, 30.0, 20.0))]]
        nra3 = nra.array(buffer=my_buffer, descr=self.descr)

        self.assert_(common.areArraysEqual(nra0, nra3))


def suite():
    """Return a test suite consisting of all the test cases in the module."""

    theSuite = unittest.TestSuite()
    theSuite.addTest(unittest.makeSuite(NestedRecordTests))
    return theSuite


if __name__ == '__main__':
    unittest.main(defaultTest='NestedRecordTests' )
