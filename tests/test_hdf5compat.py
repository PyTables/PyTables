"""Test module for compatibility with plain HDF files."""

import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np

import tables as tb
from tests import common
from tables import utilsextension
from tables.utils import print_versions


class PaddedArrayTestCase(common.TestFileMixin, common.PyTablesTestCase):
    """Test for H5T_COMPOUND (Table) datatype with padding.

    Regression test for issue gh-734

    itemsize.h5 was created with h5py with the array `expectedData` (see below)
    in the table `/Test`:
    'A' and 'B' are 4 + 4 bytes, with 8 bytes padding.

    $ h5ls -v itemsize.h5
    Test                     Dataset {3/3}
    Location:  1:800
    Links:     1
    Storage:   48 logical bytes, 48 allocated bytes, 100.00% utilization
    Type:      struct {
                   "A"                +0    native unsigned int
                   "B"                +4    native unsigned int
               } 16 bytes

    """

    h5fname = common.test_filename("itemsize.h5")

    def test(self):
        arr = self.h5file.get_node("/Test")
        data = arr.read()
        expectedData = np.array(
            [(1, 11), (2, 12), (3, 13)],
            dtype={
                "names": ["A", "B"],
                "formats": ["<u4", "<u4"],
                "offsets": [0, 4],
                "itemsize": 16,
            },
        )
        self.assertTrue(common.are_arrays_equal(data, expectedData))


class PaddedCompoundAttrTestCase(
    common.TestFileMixin, common.PyTablesTestCase
):
    """Test for a padded H5T_COMPOUND stored as an attribute.

    Companion to PaddedArrayTestCase for the attribute reader (see #765):
    the same trailing padding that tables already handle also has to be
    honored when the compound comes from an attribute, otherwise the NumPy
    container is allocated too small and the attribute read runs past it.

    padded_compound_attr.h5 was created with h5py; the dataset `/Test`
    carries an attribute `compound_attr` holding the array `expectedData`
    (see below).  'A' and 'B' are 4 + 4 bytes with 8 bytes of trailing
    padding, so the on-disk compound is 16 bytes wide.

    $ h5dump -A -H padded_compound_attr.h5
    ATTRIBUTE "compound_attr" {
       DATATYPE  H5T_COMPOUND {
          H5T_STD_U32LE "A";
          H5T_STD_U32LE "B";
       } 16 bytes
       DATASPACE  SIMPLE { ( 3 ) / ( 3 ) }
    }

    """

    h5fname = common.test_filename("padded_compound_attr.h5")

    def test(self):
        expectedData = np.array(
            [(1, 11), (2, 12), (3, 13)],
            dtype={
                "names": ["A", "B"],
                "formats": ["<u4", "<u4"],
                "offsets": [0, 4],
                "itemsize": 16,
            },
        )
        attr = self.h5file.get_node("/Test")._v_attrs.compound_attr
        # The container must keep the on-disk itemsize, otherwise H5Aread
        # writes the padded 16-byte records into a packed 8-byte buffer.
        self.assertEqual(attr.dtype.itemsize, expectedData.dtype.itemsize)
        self.assertTrue(common.are_arrays_equal(attr, expectedData))


class WideBitfieldAttrTestCase(common.TestFileMixin, common.PyTablesTestCase):
    """Test for a wider-than-1-byte H5T_BITFIELD stored as an attribute.

    PyTables maps every bitfield to the 1-byte NumPy ``"b1"`` (bool) type
    regardless of its on-disk size.  For a bitfield wider than 1 byte the
    receive buffer was therefore sized below H5Tget_size and H5Aread wrote
    the full-width records past its end, a heap overflow driven purely by
    the attribute datatype.  PyTables only ever writes 1-byte bitfields
    (booleans), so a wider one comes from a foreign file and must be
    reported as an unsupported type instead of being read into an
    undersized container.

    wide_bitfield_attr.h5 was created with h5py; the group `/Test` carries
    a scalar and a 100-element H5T_STD_B64LE (8-byte) attribute plus a
    1-byte H5T_STD_B8LE attribute standing in for a regular boolean.

    $ h5dump -A -H wide_bitfield_attr.h5
    GROUP "Test" {
       ATTRIBUTE "array_bitfield"  { DATATYPE H5T_STD_B64LE ... ( 100 ) }
       ATTRIBUTE "ok_bool"         { DATATYPE H5T_STD_B8LE  ... SCALAR }
       ATTRIBUTE "scalar_bitfield" { DATATYPE H5T_STD_B64LE ... SCALAR }
    }

    """

    h5fname = common.test_filename("wide_bitfield_attr.h5")

    def test(self):
        # Materializing the attribute set reads every attribute eagerly and
        # flags the 8-byte bitfields as unsupported with a DataTypeWarning.
        with self.assertWarns(tb.exceptions.DataTypeWarning):
            attrs = self.h5file.get_node("/Test")._v_attrs

        # A 1-byte bitfield is a normal boolean and keeps reading fine.
        self.assertEqual(bool(attrs.ok_bool), True)

        # Both the scalar and the array path must refuse the 8-byte bitfield
        # as unsupported (returning None) rather than reading its full-width
        # records into a 1-byte-per-element buffer.
        self.assertIsNone(attrs.scalar_bitfield)
        self.assertIsNone(attrs.array_bitfield)


class OversizeStringAttrTestCase(common.PyTablesTestCase):
    """Test for a non-scalar string stored in a system attribute.

    H5ATTRget_attribute_string() sized its receive buffer for a single
    element (H5Tget_size), but H5Aread fills the whole attribute, so a
    fixed-length string attribute holding more than one element wrote past
    the buffer.  read_f_attr() reads PYTABLES_FORMAT_VERSION through this
    function, so merely identifying a crafted file with is_pytables_file()
    triggered the overflow (ASAN: heap write of 256 bytes into a 5-byte
    allocation).  A real PyTables file keeps that attribute scalar, so a
    multi-element one comes from a foreign file and must be read without
    running past the buffer.

    oversize_str_attr.h5 was created with h5py; the root group carries a
    64-element 8-byte PYTABLES_FORMAT_VERSION attribute whose first element
    is ``b"2.1"``.

    $ h5dump -A -H oversize_str_attr.h5
    ATTRIBUTE "PYTABLES_FORMAT_VERSION" {
       DATATYPE  H5T_STRING { STRSIZE 8; ... }
       DATASPACE  SIMPLE { ( 64 ) / ( 64 ) }
    }

    """

    h5fname = common.test_filename("oversize_str_attr.h5")

    def test(self):
        # Reading the oversized attribute must stay within the buffer and
        # return the leading string rather than overflowing it.
        version = tb.is_pytables_file(self.h5fname)
        self.assertEqual(version, "2.1")


class EnumTestCase(common.TestFileMixin, common.PyTablesTestCase):
    """Test for enumerated datatype.

    See ftp://ftp.hdfgroup.org/HDF5/current/src/unpacked/test/enum.c.

    """

    h5fname = common.test_filename("smpl_enum.h5")

    def test(self):
        self.assertIn("/EnumTest", self.h5file)

        arr = self.h5file.get_node("/EnumTest")
        self.assertIsInstance(arr, tb.Array)

        enum = arr.get_enum()
        expectedEnum = tb.Enum(["RED", "GREEN", "BLUE", "WHITE", "BLACK"])
        self.assertEqual(enum, expectedEnum)

        data = list(arr.read())
        expectedData = [
            enum[name]
            for name in [
                "RED",
                "GREEN",
                "BLUE",
                "WHITE",
                "BLACK",
                "RED",
                "GREEN",
                "BLUE",
                "WHITE",
                "BLACK",
            ]
        ]
        self.assertEqual(data, expectedData)


class NumericTestCase(common.TestFileMixin, common.PyTablesTestCase):
    """Test for several numeric datatypes.

    See
    ftp://ftp.ncsa.uiuc.edu/HDF/files/hdf5/samples/[fiu]l?{8,16,32,64}{be,le}.c
    (they seem to be no longer available).

    """

    def test(self):
        self.assertIn("/TestArray", self.h5file)

        arr = self.h5file.get_node("/TestArray")
        self.assertIsInstance(arr, tb.Array)

        self.assertEqual(arr.atom.type, self.type)
        self.assertEqual(arr.byteorder, self.byteorder)
        self.assertEqual(arr.shape, (6, 5))

        data = arr.read()
        expectedData = np.array(
            [
                [0, 1, 2, 3, 4],
                [1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7],
                [4, 5, 6, 7, 8],
                [5, 6, 7, 8, 9],
            ],
            dtype=self.type,
        )
        self.assertTrue(common.are_arrays_equal(data, expectedData))


class EnumAttributeTestCase(common.PyTablesTestCase):
    """Test reading scalar H5T_ENUM attributes from foreign HDF5 files.

    Regression test for issue gh-1035.  ``enum_attr.h5`` was created with
    the HDF5 C API and contains two 8-bit enum attributes on the root group.
    The enum members are represented by their underlying integer values when
    exposed through the PyTables attribute API.
    """

    h5fname = common.test_filename("enum_attr.h5")

    def test(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with tb.open_file(self.h5fname) as h5file:
                attrs = h5file.root._v_attrs
                false_value = attrs.false_value
                true_value = attrs.true_value

        datatype_warnings = [
            warning
            for warning in caught
            if issubclass(warning.category, tb.exceptions.DataTypeWarning)
        ]
        self.assertEqual(datatype_warnings, [])
        self.assertEqual(false_value.dtype, np.dtype("int8"))
        self.assertEqual(true_value.dtype, np.dtype("int8"))
        self.assertEqual(false_value, np.int8(0))
        self.assertEqual(true_value, np.int8(1))


class EnumAttributeTypesTestCase(common.PyTablesTestCase):
    """Check enum scalar widths, signs, byte order, and unsupported shapes.

    ``enum_attr_types.h5`` was generated with the portable
    ``tests/_testtools/create_enum_attr_types.py`` HDF5 C-API helper.  It
    contains native-endian and foreign-endian signed/unsigned enum scalars,
    an enum array attribute, and a legal 16-byte enum scalar.
    """

    h5fname = common.test_filename("enum_attr_types.h5")

    def test(self):
        expected = {
            "signed_be16": (np.int16(0x1234), np.dtype("int16")),
            "signed_le32": (np.int32(0x12345678), np.dtype("int32")),
            "unsigned_be16": (np.uint16(0xFEDC), np.dtype("uint16")),
            "unsigned_le32": (np.uint32(0xFEDCBA98), np.dtype("uint32")),
        }

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with tb.open_file(self.h5fname) as h5file:
                attrs = h5file.root._v_attrs
                actual = {name: getattr(attrs, name) for name in expected}
                enum_array = attrs.unsigned_be16_array
                wide_enum = attrs.signed_i128
                with warnings.catch_warnings():
                    warnings.simplefilter(
                        "error", tb.exceptions.DataTypeWarning
                    )
                    with self.assertRaises(tb.exceptions.DataTypeWarning):
                        attrs._g_getattr(h5file.root, "signed_i128")

        datatype_warnings = [
            warning
            for warning in caught
            if issubclass(warning.category, tb.exceptions.DataTypeWarning)
        ]
        self.assertEqual(len(datatype_warnings), 2)
        self.assertIsNone(enum_array)
        self.assertIsNone(wide_enum)
        for name, (value, dtype) in expected.items():
            self.assertEqual(actual[name].dtype, dtype)
            self.assertEqual(actual[name], value)


class NumericAttributeTypesTestCase(common.PyTablesTestCase):
    """Check scalar and array numeric attributes in both byte orders."""

    h5fname = common.test_filename("numeric_attr_types.h5")

    def test(self):
        integer_values = {
            "signed_i16_le": (-0x1234, np.dtype("int16")),
            "signed_i16_be": (-0x1234, np.dtype("int16")),
            "signed_i32_le": (-0x12345678, np.dtype("int32")),
            "signed_i32_be": (-0x12345678, np.dtype("int32")),
            "signed_i64_le": (-0x123456789ABCDEF, np.dtype("int64")),
            "signed_i64_be": (-0x123456789ABCDEF, np.dtype("int64")),
            "unsigned_u16_le": (0xCDEF, np.dtype("uint16")),
            "unsigned_u16_be": (0xCDEF, np.dtype("uint16")),
            "unsigned_u32_le": (0x89ABCDEF, np.dtype("uint32")),
            "unsigned_u32_be": (0x89ABCDEF, np.dtype("uint32")),
            "unsigned_u64_le": (0xF23456789ABCDEF0, np.dtype("uint64")),
            "unsigned_u64_be": (0xF23456789ABCDEF0, np.dtype("uint64")),
        }
        float_values = {
            "float32_le": (1234.5, np.dtype("float32")),
            "float32_be": (1234.5, np.dtype("float32")),
            "float64_le": (-98765.25, np.dtype("float64")),
            "float64_be": (-98765.25, np.dtype("float64")),
        }

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with tb.open_file(self.h5fname) as h5file:
                attrs = h5file.root._v_attrs
                for name, (expected, dtype) in integer_values.items():
                    value = getattr(attrs, name)
                    self.assertEqual(value.dtype, dtype)
                    self.assertEqual(value, expected)

                for name, (expected, dtype) in float_values.items():
                    value = getattr(attrs, name)
                    self.assertEqual(value.dtype, dtype)
                    self.assertEqual(value, expected)

                for stem, dtype in (
                    ("float32_negzero", np.dtype("float32")),
                    ("float64_negzero", np.dtype("float64")),
                ):
                    for order in ("le", "be"):
                        value = getattr(attrs, f"{stem}_{order}")
                        self.assertEqual(value.dtype, dtype)
                        self.assertEqual(value, 0)
                        self.assertTrue(np.signbit(value))

                for stem, dtype in (
                    ("float32_inf", np.dtype("float32")),
                    ("float64_inf", np.dtype("float64")),
                ):
                    for order in ("le", "be"):
                        value = getattr(attrs, f"{stem}_{order}")
                        self.assertEqual(value.dtype, dtype)
                        self.assertTrue(np.isposinf(value))

                for stem, dtype in (
                    ("float32_nan", np.dtype("float32")),
                    ("float64_nan", np.dtype("float64")),
                ):
                    for order in ("le", "be"):
                        value = getattr(attrs, f"{stem}_{order}")
                        self.assertEqual(value.dtype, dtype)
                        self.assertTrue(np.isnan(value))

                signed_array = attrs.signed_i32_be_array
                self.assertEqual(signed_array.dtype, np.dtype("int32"))
                self.assertTrue(
                    np.array_equal(
                        signed_array,
                        np.array([-0x12345678, 0, 0x12345678], dtype="int32"),
                    )
                )
                unsigned_array = attrs.unsigned_u64_be_array
                self.assertEqual(unsigned_array.dtype, np.dtype("uint64"))
                self.assertTrue(
                    np.array_equal(
                        unsigned_array,
                        np.array(
                            [0xF23456789ABCDEF0, 0x0123456789ABCDEF],
                            dtype="uint64",
                        ),
                    )
                )
                float32_array = attrs.float32_be_special_array
                self.assertEqual(float32_array.dtype, np.dtype("float32"))
                self.assertEqual(float32_array[0], 0)
                self.assertTrue(np.signbit(float32_array[0]))
                self.assertTrue(np.isposinf(float32_array[1]))
                self.assertTrue(np.isnan(float32_array[2]))
                float64_array = attrs.float64_le_special_array
                self.assertEqual(float64_array.dtype, np.dtype("float64"))
                self.assertEqual(float64_array[0], 0)
                self.assertTrue(np.signbit(float64_array[0]))
                self.assertTrue(np.isposinf(float64_array[1]))
                self.assertTrue(np.isnan(float64_array[2]))

                narrow_scalars = {
                    "narrow_i64_scalar": (-0x345, np.dtype("int16")),
                    "narrow_u64_scalar": (0xABC, np.dtype("uint16")),
                }
                for name, (expected, dtype) in narrow_scalars.items():
                    value = getattr(attrs, name)
                    self.assertEqual(value.dtype, dtype)
                    self.assertEqual(value, expected)

                narrow_arrays = {
                    "narrow_i64": (
                        np.array((-0x345, 0, 0x456), dtype="int16"),
                        np.dtype("int16"),
                    ),
                    "narrow_u64": (
                        np.array((0xABC, 0x123, 0x456), dtype="uint16"),
                        np.dtype("uint16"),
                    ),
                }
                for stem, (expected, dtype) in narrow_arrays.items():
                    for suffix in ("rank_array", "h5t_array"):
                        value = getattr(attrs, f"{stem}_{suffix}")
                        self.assertEqual(value.dtype, dtype)
                        self.assertTrue(np.array_equal(value, expected))

        self.assertEqual(caught, [])


class F64BETestCase(NumericTestCase):
    h5fname = common.test_filename("smpl_f64be.h5")
    type = "float64"
    byteorder = "big"


class F64LETestCase(NumericTestCase):
    h5fname = common.test_filename("smpl_f64le.h5")
    type = "float64"
    byteorder = "little"


class I64BETestCase(NumericTestCase):
    h5fname = common.test_filename("smpl_i64be.h5")
    type = "int64"
    byteorder = "big"


class I64LETestCase(NumericTestCase):
    h5fname = common.test_filename("smpl_i64le.h5")
    type = "int64"
    byteorder = "little"


class I32BETestCase(NumericTestCase):
    h5fname = common.test_filename("smpl_i32be.h5")
    type = "int32"
    byteorder = "big"


class I32LETestCase(NumericTestCase):
    h5fname = common.test_filename("smpl_i32le.h5")
    type = "int32"
    byteorder = "little"


class F64VAXTestCase(common.TestFileMixin, common.PyTablesTestCase):
    """Test for a dataset with a VAX ordered floating point datatype.

    smpl_f64vax.h5 holds a chunked 6x5 ``/TestArray`` created with the
    predefined ``H5T_VAX_F64`` datatype, for which ``H5Tget_order()``
    reports ``H5T_ORDER_VAX``.  PyTables cannot interpret that byte order,
    so no data is checked here; opening the dataset only has to report the
    byte order as unsupported.

    """

    h5fname = common.test_filename("smpl_f64vax.h5")

    def test(self):
        arr = self.h5file.get_node("/TestArray")
        self.assertIsInstance(arr, tb.Array)

        self.assertEqual(arr.atom.type, "float64")
        self.assertEqual(arr.shape, (6, 5))
        self.assertEqual(arr.byteorder, "unsupported")


class ChunkedCompoundTestCase(common.TestFileMixin, common.PyTablesTestCase):
    """Test for a more complex and chunked compound structure.

    This is generated by a chunked version of the example in
    ftp://ftp.ncsa.uiuc.edu/HDF/files/hdf5/samples/compound2.c.

    """

    h5fname = common.test_filename("smpl_compound_chunked.h5")

    def test(self):
        self.assertIn("/CompoundChunked", self.h5file)

        tbl = self.h5file.get_node("/CompoundChunked")
        self.assertIsInstance(tbl, tb.Table)

        self.assertEqual(
            tbl.colnames,
            ["a_name", "c_name", "d_name", "e_name", "f_name", "g_name"],
        )

        self.assertEqual(tbl.coltypes["a_name"], "int32")
        self.assertEqual(tbl.coldtypes["a_name"].shape, ())

        self.assertEqual(tbl.coltypes["c_name"], "string")
        self.assertEqual(tbl.coldtypes["c_name"].shape, ())

        self.assertEqual(tbl.coltypes["d_name"], "int16")
        self.assertEqual(tbl.coldtypes["d_name"].shape, (5, 10))

        self.assertEqual(tbl.coltypes["e_name"], "float32")
        self.assertEqual(tbl.coldtypes["e_name"].shape, ())

        self.assertEqual(tbl.coltypes["f_name"], "float64")
        self.assertEqual(tbl.coldtypes["f_name"].shape, (10,))

        self.assertEqual(tbl.coltypes["g_name"], "uint8")
        self.assertEqual(tbl.coldtypes["g_name"].shape, ())

        for m in range(len(tbl)):
            row = tbl[m]
            # This version of the loop seems to fail because of ``iterrows()``.
            # for (m, row) in enumerate(tbl):
            self.assertEqual(row["a_name"], m)
            self.assertEqual(row["c_name"], b"Hello!")
            dRow = row["d_name"]
            for n in range(5):
                for o in range(10):
                    self.assertEqual(dRow[n][o], m + n + o)
            self.assertAlmostEqual(row["e_name"], m * 0.96, places=6)
            fRow = row["f_name"]
            for n in range(10):
                self.assertAlmostEqual(fRow[n], m * 1024.9637)
            self.assertEqual(row["g_name"], ord("m"))


class ContiguousCompoundTestCase(
    common.TestFileMixin, common.PyTablesTestCase
):
    """Test for support of native contiguous compound datasets.

    This example has been provided by Dav Clark.

    """

    h5fname = common.test_filename("non-chunked-table.h5")

    def test(self):
        self.assertIn("/test_var/structure variable", self.h5file)

        tbl = self.h5file.get_node("/test_var/structure variable")
        self.assertIsInstance(tbl, tb.Table)

        self.assertEqual(tbl.colnames, ["a", "b", "c", "d"])

        self.assertEqual(tbl.coltypes["a"], "float64")
        self.assertEqual(tbl.coldtypes["a"].shape, ())

        self.assertEqual(tbl.coltypes["b"], "float64")
        self.assertEqual(tbl.coldtypes["b"].shape, ())

        self.assertEqual(tbl.coltypes["c"], "float64")
        self.assertEqual(tbl.coldtypes["c"].shape, (2,))

        self.assertEqual(tbl.coltypes["d"], "string")
        self.assertEqual(tbl.coldtypes["d"].shape, ())

        for row in tbl.iterrows():
            self.assertEqual(row["a"], 3.0)
            self.assertEqual(row["b"], 4.0)
            self.assertTrue(
                common.allequal(
                    row["c"], np.array([2.0, 3.0], dtype="float64")
                )
            )
            self.assertEqual(row["d"], b"d")

        self.h5file.close()


class ContiguousCompoundAppendTestCase(
    common.TestFileMixin, common.PyTablesTestCase
):
    """Test for appending data to native contiguous compound datasets."""

    h5fname = common.test_filename("non-chunked-table.h5")

    def test(self):
        self.assertIn("/test_var/structure variable", self.h5file)
        self.h5file.close()
        # Do a copy to a temporary to avoid modifying the original file
        h5fname_copy = tempfile.mktemp(".h5")
        shutil.copy(self.h5fname, h5fname_copy)
        # Reopen in 'a'ppend mode
        try:
            self.h5file = tb.open_file(h5fname_copy, "a")
        except OSError:
            # Problems for opening (probably not permissions to write the file)
            return
        tbl = self.h5file.get_node("/test_var/structure variable")
        # Try to add rows to a non-chunked table (this should raise an error)
        self.assertRaises(
            tb.HDF5ExtError, tbl.append, [(4.0, 5.0, [2.0, 3.0], "d")]
        )
        # Appending using the Row interface
        self.assertRaises(tb.HDF5ExtError, tbl.row.append)
        # Remove the file copy
        self.h5file.close()  # Close the handler first
        Path(h5fname_copy).unlink()


class ExtendibleTestCase(common.TestFileMixin, common.PyTablesTestCase):
    """Test for extendible datasets.

    See the example programs in the Introduction to HDF5.

    """

    h5fname = common.test_filename("smpl_SDSextendible.h5")

    def test(self):
        self.assertIn("/ExtendibleArray", self.h5file)

        arr = self.h5file.get_node("/ExtendibleArray")
        self.assertIsInstance(arr, tb.EArray)

        self.assertEqual(arr.byteorder, "big")
        self.assertEqual(arr.atom.type, "int32")
        self.assertEqual(arr.shape, (10, 5))
        self.assertEqual(arr.extdim, 0)
        self.assertEqual(len(arr), 10)

        data = arr.read()
        expectedData = np.array(
            [
                [1, 1, 1, 3, 3],
                [1, 1, 1, 3, 3],
                [1, 1, 1, 0, 0],
                [2, 0, 0, 0, 0],
                [2, 0, 0, 0, 0],
                [2, 0, 0, 0, 0],
                [2, 0, 0, 0, 0],
                [2, 0, 0, 0, 0],
                [2, 0, 0, 0, 0],
                [2, 0, 0, 0, 0],
            ],
            dtype=arr.atom.type,
        )

        self.assertTrue(common.are_arrays_equal(data, expectedData))


class SzipTestCase(common.TestFileMixin, common.PyTablesTestCase):
    """Test for native HDF5 files with datasets compressed with szip."""

    h5fname = common.test_filename("test_szip.h5")

    def test(self):
        self.assertIn("/dset_szip", self.h5file)

        arr = self.h5file.get_node("/dset_szip")
        filters = (
            "Filters(complib='szip', shuffle=False, bitshuffle=False, "
            "fletcher32=False, least_significant_digit=None)"
        )
        self.assertEqual(repr(arr.filters), filters)


class ManyFilterClientValuesTestCase(
    common.TestFileMixin, common.PyTablesTestCase
):
    """Test for a filter pipeline holding more values than can be read.

    get_filter_names() reads the client data values of each filter into a
    fixed buffer, but H5Pget_filter() reports how many of them the file
    holds rather than how many it stored, so a dataset written with more
    values than the buffer takes used to be reported with the tail of the
    tuple coming from past the end of that buffer.

    many_filter_cdvalues.h5 was created with h5py; the chunked dataset
    `/array` carries filter 32768 (an id no registered filter uses, so it
    is reported with an empty name) with 40 client data values, the i-th
    one being 0xAA000000 + i.

        dcpl = h5p.create(h5p.DATASET_CREATE)
        dcpl.set_chunk((4,))
        dcpl.set_filter(
            32768, h5z.FLAG_OPTIONAL,
            tuple(0xAA000000 + i for i in range(40)),
        )

    """

    h5fname = common.test_filename("many_filter_cdvalues.h5")

    def test(self):
        arr = self.h5file.get_node("/array")
        self.assertTrue(
            common.are_arrays_equal(arr.read(), np.arange(4, dtype="f8"))
        )

        (values,) = utilsextension.get_filters(
            arr._v_parent._v_objectid, arr._v_name
        ).values()

        # Only the values that were actually read may be reported, so the
        # tuple is shorter than the 40 the file holds and every value in it
        # is one of those written above.
        self.assertLess(len(values), 40)
        self.assertEqual(
            list(values), [0xAA000000 + i for i in range(len(values))]
        )


class WideEnumLeafTestCase(common.TestFileMixin, common.PyTablesTestCase):
    """Test for an enum whose base integer is wider than any native one.

    get_native_type() and get_type_enum() ask HDF5 for the native type of
    the enum with H5Tget_native_type(), which overflows a heap buffer when
    the base integer is wider than the widest native integer (more than 8
    bytes), so merely opening such a leaf used to corrupt memory.

    wide_enum_leaf.h5 was created with h5py; the chunked dataset `/array`
    has an enum datatype whose base is a 16-byte integer.

        base = h5t.STD_I64LE.copy()
        base.set_size(16)
        base.set_precision(128)
        et = h5t.enum_create(base)
        et.enum_insert(b"A", 0)
        et.enum_insert(b"B", 1)
        dcpl = h5p.create(h5p.DATASET_CREATE)
        dcpl.set_chunk((2,))
        h5d.create(f.id, b"array", et, h5s.create_simple((4,)), dcpl=dcpl)

    """

    h5fname = common.test_filename("wide_enum_leaf.h5")

    def test(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            node = self.h5file.get_node("/array")
        # The base has no NumPy counterpart, so the leaf is unsupported
        # instead of overflowing while reading its native type.
        self.assertIsInstance(node, tb.UnImplemented)


class Blosc2ForgedTypesizeTestCase(
    common.TestFileMixin, common.PyTablesTestCase
):
    """Test for a Blosc2 filter whose typesize disagrees with the datatype.

    The optimized Blosc2 slice read took the item size from the filter's
    client data values (cd_values[2]) rather than from the dataset's
    datatype.  That value drives both the temporary chunk buffer and the
    copy into the receive buffer, which PyTables sizes from the datatype,
    so a file whose filter typesize is larger than its element size made
    b2nd_copy_buffer2 write past the receive buffer (ASAN: heap write past
    the per-element NumPy buffer in get_blosc2_slice).  A real PyTables
    file keeps the two in sync, so a mismatch comes from a foreign file and
    must drop to the plain HDF5 read.

    Whether that plain read succeeds depends on the HDF5 version: the Blosc2
    filter returns a chunk sized from the forged typesize, which HDF5 1.14
    truncates to the real chunk size but HDF5 2.x rejects.  Either a correct
    read or a clean HDF5ExtError is fine; what matters is no overflow.

    blosc2_forged_typesize.h5 was created with h5py; the 64x64 int8 dataset
    `/data` stores a single Blosc2 chunk (no shuffle) whose filter
    cd_values[2] is forged to 8 while the datatype stays 1 byte.
    """

    h5fname = common.test_filename("blosc2_forged_typesize.h5")

    def test(self):
        expected = (np.arange(64 * 64) & 0x7F).astype("i1").reshape(64, 64)
        arr = self.h5file.get_node("/data")
        self.assertEqual(arr.shape, (64, 64))
        # Both a partial slice and a full read must stay within the receive
        # buffer; the mismatched typesize forces the plain HDF5 read.
        for key in ((slice(8, 56), slice(4, 60)), slice(None)):
            with self.subTest(key=key):
                try:
                    data = arr[key]
                except tb.HDF5ExtError:
                    continue
                self.assertTrue(common.are_arrays_equal(data, expected[key]))


class VlenTableColumnTestCase(common.TestFileMixin, common.PyTablesTestCase):
    """Test for a table whose compound type has a variable-length column.

    get_nested_type() mapped a variable-length member to its base atom and
    dropped the on-disk ``hvl_t`` size, so the NumPy record was smaller than
    H5Tget_size(disk_type) while H5Dread filled the whole ``hvl_t``, writing
    past the read buffer (ASAN: heap-buffer-overflow in H5TBOread_records
    reached from Table._read_records).  Variable-length data belongs in a
    VLArray, not a table column, so such a leaf must open as UnImplemented
    instead of overflowing on read.

    vlen_table_column.h5 was created with h5py; the dataset `/table` is a
    compound {a: int32, b: vlen int32} with the usual PyTables TABLE
    attributes.
    """

    h5fname = common.test_filename("vlen_table_column.h5")

    def test(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            node = self.h5file.get_node("/table")
        # Rejected on open, so the overflowing table read is never reached.
        self.assertIsInstance(node, tb.UnImplemented)


class OddSizeIntTableColumnTestCase(
    common.TestFileMixin, common.PyTablesTestCase
):
    """Test for a table column whose integer has no matching native size.

    get_nested_type() builds the native in-memory compound with
    H5Tcreate(H5T_COMPOUND, on_disk_size) and inserts each member at its
    on-disk offset, which only stays valid while every member keeps its
    on-disk size.  An odd-precision integer column (3, 5, 6 or 7 bytes) has
    no NumPy counterpart, so get_native_type() widens it (e.g. 5 -> 8 bytes)
    and the inserted member overruns the compound.  H5Dread then walks past
    the malformed datatype on read (crash in H5T__sort_value, reached from
    Table._read_records).  A real PyTables table only stores standard integer
    widths, so such a column comes from a foreign file and must open as
    UnImplemented instead.

    oddsize_int_table.h5 was created with h5py; the dataset `/table` is a
    compound {c: 5-byte little-endian integer} with the usual PyTables TABLE
    attributes.
    """

    h5fname = common.test_filename("oddsize_int_table.h5")

    def test(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            node = self.h5file.get_node("/table")
        # Rejected on open, so the crashing table read is never reached.
        self.assertIsInstance(node, tb.UnImplemented)


class MultidimTableLeafTestCase(common.TestFileMixin, common.PyTablesTestCase):
    """Test for a Table/VLArray leaf whose dataset has more than one dimension.

    Table._get_info() and H5VLARRAYget_info() read the dataset extent into
    a single ``hsize_t`` (tables and VLArrays are 1-D), so a chunked compound
    or vlen dataset of rank greater than one made H5Sget_simple_extent_dims()
    write past that buffer merely on opening the leaf.  A forged ``CLASS``
    attribute reaches those readers directly, bypassing the rank-aware
    which_class() guess.  Such a leaf comes from a foreign file and must open
    as UnImplemented instead of overflowing.

    multidim_table_leaf.h5 was created with h5py; `/table` is a chunked
    compound {a: int32, b: float64} and `/vlarray` a chunked vlen int32, both
    2-D and both carrying the usual PyTables TABLE/VLARRAY attributes.
    """

    h5fname = common.test_filename("multidim_table_leaf.h5")

    def test(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            table = self.h5file.get_node("/table")
            vlarray = self.h5file.get_node("/vlarray")
        # Rejected on open, so the out-of-bounds extent read never happens.
        self.assertIsInstance(table, tb.UnImplemented)
        self.assertIsInstance(vlarray, tb.UnImplemented)


class ForgedLeafClassTestCase(common.TestFileMixin, common.PyTablesTestCase):
    """Test for a leaf whose ``CLASS`` attribute disagrees with its datatype.

    A ``CLASS`` attribute makes PyTables load a node with the matching reader
    without re-checking the datatype, so a foreign file can point the VLArray
    or Array readers at a datatype they cannot handle.  H5VLARRAYget_info()
    did not require an ``H5T_VLEN`` datatype, so a fixed ``H5T_ARRAY`` dataset
    labelled ``VLARRAY`` was read as an array of ``hvl_t`` handles: the row
    buffer overflowed and the reader dereferenced file-controlled bytes as
    data pointers.  Symmetrically, Array/CArray/EArray built the atom from a
    variable-length datatype's base and then read with the vlen memory type,
    overflowing the base-sized buffer with ``hvl_t`` handles.  Both reads
    segfault on the unpatched tree.  Such leaves come from a foreign file and
    must open as UnImplemented instead.

    forged_leaf_class.h5 was created with h5py.  `/vlarray_arraytype` is a 1-D
    chunked ``H5T_ARRAY`` (10 x int64) labelled ``VLARRAY``; `/array_vlentype`,
    `/carray_vlentype` and `/earray_vlentype` are 1-D chunked vlen int64
    datasets labelled ``ARRAY``, ``CARRAY`` and ``EARRAY``.  All carry the
    usual PyTables leaf attributes.
    """

    h5fname = common.test_filename("forged_leaf_class.h5")

    def test(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            nodes = [
                self.h5file.get_node("/vlarray_arraytype"),
                self.h5file.get_node("/array_vlentype"),
                self.h5file.get_node("/carray_vlentype"),
                self.h5file.get_node("/earray_vlentype"),
            ]
        # Rejected on open, so the out-of-bounds read never happens.
        for node in nodes:
            self.assertIsInstance(node, tb.UnImplemented)


class NestedArrayTypeTestCase(common.TestFileMixin, common.PyTablesTestCase):
    """Test for a nested H5T_ARRAY datatype (an array of arrays).

    hdf5_to_np_ext_type() mapped an H5T_ARRAY to the NumPy shape of its own
    dimensions but discarded the shape of its base type.  When the base is
    itself an H5T_ARRAY (a foreign nested array datatype), only the outer
    dimension survived, so the NumPy container was built smaller than
    H5Tget_size and H5Dread/H5Aread wrote the full element past it: a leaf,
    a table column and an attribute each overflowed their read buffer.  The
    full shape has a NumPy counterpart, so the value must read back intact.

    nested_array_leaf.h5 was created with h5py; the datatype
    ``[2] H5T_ARRAY { [3] H5T_STD_I32LE }`` appears as the group attribute
    ``nested``, the Array leaf ``/array`` and the ``b`` column of the compound
    ``/table`` ({a: int32, b: nested}).
    """

    h5fname = common.test_filename("nested_array_leaf.h5")

    def test(self):
        attr = self.h5file.root._v_attrs.nested
        self.assertTrue(
            common.are_arrays_equal(
                attr, np.arange(6, dtype="int32").reshape(2, 3)
            )
        )

        array = self.h5file.get_node("/array")
        self.assertEqual(array.atom.shape, (2, 3))
        self.assertTrue(
            common.are_arrays_equal(
                array.read(), np.arange(24, dtype="int32").reshape(4, 2, 3)
            )
        )

        table = self.h5file.get_node("/table")
        self.assertEqual(table.coldtypes["b"].shape, (2, 3))
        data = table.read()
        self.assertTrue(
            common.are_arrays_equal(data["a"], np.arange(5, dtype="int32"))
        )
        self.assertTrue(
            common.are_arrays_equal(
                data["b"], np.arange(30, dtype="int32").reshape(5, 2, 3)
            )
        )


# this demonstrates github #203
class MatlabFileTestCase(common.TestFileMixin, common.PyTablesTestCase):
    h5fname = common.test_filename("matlab_file.mat")

    def test_unicode(self):
        array = self.h5file.get_node("/", "a")
        self.assertEqual(array.shape, (3, 1))

    # in Python 3 this will be the same as the test above
    def test_string(self):
        array = self.h5file.get_node("/", "a")
        self.assertEqual(array.shape, (3, 1))

    def test_numpy_str(self):
        array = self.h5file.get_node(np.str_("/"), np.str_("a"))
        self.assertEqual(array.shape, (3, 1))


class ObjectReferenceTestCase(common.TestFileMixin, common.PyTablesTestCase):
    h5fname = common.test_filename("test_ref_array1.mat")

    def test_node_var(self):
        array = self.h5file.get_node("/ANN/my_arr")
        self.assertEqual(array.shape, (1, 3))

    def test_ref_utf_str(self):
        array = self.h5file.get_node("/ANN/my_arr")

        self.assertTrue(
            common.are_arrays_equal(
                array[0][0][0], np.array([0, 0], dtype=np.uint64)
            )
        )


class ObjectReferenceRecursiveTestCase(
    common.TestFileMixin, common.PyTablesTestCase
):
    h5fname = common.test_filename("test_ref_array2.mat")

    def test_var(self):
        array = self.h5file.get_node("/var")
        self.assertEqual(array.shape, (3, 1))

    def test_ref_str(self):
        array = self.h5file.get_node("/var")

        self.assertTrue(
            common.are_arrays_equal(
                array[1][0][0],
                np.array([[116], [101], [115], [116]], dtype=np.uint16),
            )
        )

    def test_double_ref(self):
        array = self.h5file.get_node("/var")
        self.assertTrue(
            common.are_arrays_equal(
                array[2][0][0][1][0],
                np.array(
                    [[105], [110], [115], [105], [100], [101]], dtype=np.uint16
                ),
            )
        )


def suite():
    """Return a test suite consisting of all the test cases in the module."""

    theSuite = common.unittest.TestSuite()
    niter = 1

    for i in range(niter):
        theSuite.addTest(common.make_suite(PaddedArrayTestCase))
        theSuite.addTest(common.make_suite(PaddedCompoundAttrTestCase))
        theSuite.addTest(common.make_suite(WideBitfieldAttrTestCase))
        theSuite.addTest(common.make_suite(OversizeStringAttrTestCase))
        theSuite.addTest(common.make_suite(EnumTestCase))
        theSuite.addTest(common.make_suite(EnumAttributeTestCase))
        theSuite.addTest(common.make_suite(EnumAttributeTypesTestCase))
        theSuite.addTest(common.make_suite(NumericAttributeTypesTestCase))
        theSuite.addTest(common.make_suite(F64BETestCase))
        theSuite.addTest(common.make_suite(F64LETestCase))
        theSuite.addTest(common.make_suite(I64BETestCase))
        theSuite.addTest(common.make_suite(I64LETestCase))
        theSuite.addTest(common.make_suite(I32BETestCase))
        theSuite.addTest(common.make_suite(I32LETestCase))
        theSuite.addTest(common.make_suite(F64VAXTestCase))
        theSuite.addTest(common.make_suite(ChunkedCompoundTestCase))
        theSuite.addTest(common.make_suite(ContiguousCompoundTestCase))
        theSuite.addTest(common.make_suite(ContiguousCompoundAppendTestCase))
        theSuite.addTest(common.make_suite(ExtendibleTestCase))
        theSuite.addTest(common.make_suite(SzipTestCase))
        theSuite.addTest(common.make_suite(ManyFilterClientValuesTestCase))
        theSuite.addTest(common.make_suite(WideEnumLeafTestCase))
        theSuite.addTest(common.make_suite(Blosc2ForgedTypesizeTestCase))
        theSuite.addTest(common.make_suite(VlenTableColumnTestCase))
        theSuite.addTest(common.make_suite(MultidimTableLeafTestCase))
        theSuite.addTest(common.make_suite(ForgedLeafClassTestCase))
        theSuite.addTest(common.make_suite(NestedArrayTypeTestCase))
        theSuite.addTest(common.make_suite(MatlabFileTestCase))
        theSuite.addTest(common.make_suite(ObjectReferenceTestCase))
        theSuite.addTest(common.make_suite(ObjectReferenceRecursiveTestCase))

    return theSuite


if __name__ == "__main__":
    import sys

    print_versions()
    common.parse_argv(sys.argv)
    common.unittest.main(defaultTest="suite")
