"""Create the numeric scalar/array compatibility fixture for test_hdf5compat.

Usage: python create_numeric_attr_types.py OUTPUT.h5 [--hdf5 PATH]

The output path is explicit. H5Fcreate replaces only that requested file.
The attributes use HDF5's predefined little- and big-endian integer and IEEE
floating-point types, while H5Tget_native_type is used only for the host
memory buffer passed to H5Awrite.
"""

from __future__ import annotations

import os
import json
import ctypes as C
import hashlib
import argparse
import ctypes.util
from pathlib import Path


def checked(value, label):
    if isinstance(value, int) and value < 0:
        raise RuntimeError(f"{label} failed: {value}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--hdf5", type=Path)
    args = parser.parse_args()

    dll_dir = None
    if args.hdf5 is not None:
        library = str(args.hdf5.resolve())
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            dll_dir = os.add_dll_directory(str(args.hdf5.resolve().parent))
    else:
        library = ctypes.util.find_library("hdf5")
        if library is None:
            library = "hdf5.dll" if os.name == "nt" else "libhdf5.so"

    try:
        h5 = C.CDLL(library)
        hid = C.c_int64
        herr = C.c_int
        hsize = C.c_uint64

        def fn(name, restype, *argtypes):
            function = getattr(h5, name)
            function.restype = restype
            function.argtypes = list(argtypes)
            return function

        H5open = fn("H5open", herr)
        H5Fcreate = fn("H5Fcreate", hid, C.c_char_p, C.c_uint, hid, hid)
        H5Fclose = fn("H5Fclose", herr, hid)
        H5Oopen = fn("H5Oopen", hid, hid, C.c_char_p, hid)
        H5Oclose = fn("H5Oclose", herr, hid)
        H5Screate = fn("H5Screate", hid, C.c_int)
        H5Screate_simple = fn(
            "H5Screate_simple",
            hid,
            C.c_int,
            C.POINTER(hsize),
            C.POINTER(hsize),
        )
        H5Sclose = fn("H5Sclose", herr, hid)
        H5Tcopy = fn("H5Tcopy", hid, hid)
        H5Tclose = fn("H5Tclose", herr, hid)
        H5Tset_precision = fn("H5Tset_precision", herr, hid, C.c_size_t)
        H5Tset_offset = fn("H5Tset_offset", herr, hid, C.c_size_t)
        H5Tget_native_type = fn("H5Tget_native_type", hid, hid, C.c_int)
        H5Tarray_create = fn(
            "H5Tarray_create2", hid, hid, C.c_uint, C.POINTER(hsize)
        )
        H5Acreate2 = fn("H5Acreate2", hid, hid, C.c_char_p, hid, hid, hid, hid)
        H5Awrite = fn("H5Awrite", herr, hid, hid, C.c_void_p)
        H5Aclose = fn("H5Aclose", herr, hid)

        H5open()
        H5P_DEFAULT = hid(0)
        H5F_ACC_TRUNC = 2
        H5S_SCALAR = 0
        type_names = (
            "H5T_STD_I16LE_g",
            "H5T_STD_I16BE_g",
            "H5T_STD_I32LE_g",
            "H5T_STD_I32BE_g",
            "H5T_STD_I64LE_g",
            "H5T_STD_I64BE_g",
            "H5T_STD_U16LE_g",
            "H5T_STD_U16BE_g",
            "H5T_STD_U32LE_g",
            "H5T_STD_U32BE_g",
            "H5T_STD_U64LE_g",
            "H5T_STD_U64BE_g",
            "H5T_IEEE_F32LE_g",
            "H5T_IEEE_F32BE_g",
            "H5T_IEEE_F64LE_g",
            "H5T_IEEE_F64BE_g",
        )
        types = {name: hid.in_dll(h5, name).value for name in type_names}

        output = args.output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        fid = checked(
            H5Fcreate(
                os.fsencode(output), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT
            ),
            "file",
        )
        root = checked(H5Oopen(fid, b"/", H5P_DEFAULT), "root")
        scalar_space = checked(H5Screate(H5S_SCALAR), "scalar dataspace")
        attribute_count = 0

        def add_attribute(name, type_id, space_id, ctype, values):
            nonlocal attribute_count
            native_id = -1
            attr_id = -1
            try:
                native_id = checked(
                    H5Tget_native_type(type_id, 0), f"native type {name}"
                )
                attr_id = checked(
                    H5Acreate2(
                        root,
                        name.encode(),
                        type_id,
                        space_id,
                        H5P_DEFAULT,
                        H5P_DEFAULT,
                    ),
                    f"attribute {name}",
                )
                value_buffer = (ctype * len(values))(*values)
                checked(
                    H5Awrite(attr_id, native_id, C.byref(value_buffer)),
                    f"write {name}",
                )
                attribute_count += 1
            finally:
                if attr_id >= 0:
                    H5Aclose(attr_id)
                if native_id >= 0:
                    H5Tclose(native_id)

        def add_scalar(name, type_name, ctype, value):
            add_attribute(
                name, types[type_name], scalar_space, ctype, (value,)
            )

        def add_rank_array(name, type_id, ctype, values):
            dimensions = (hsize * 1)(len(values))
            space_id = checked(
                H5Screate_simple(1, dimensions, None),
                f"rank-1 dataspace {name}",
            )
            try:
                add_attribute(name, type_id, space_id, ctype, values)
            finally:
                H5Sclose(space_id)

        def add_h5t_array(name, type_id, ctype, values):
            dimensions = (hsize * 1)(len(values))
            array_id = checked(
                H5Tarray_create(type_id, 1, dimensions), f"array type {name}"
            )
            try:
                add_attribute(name, array_id, scalar_space, ctype, values)
            finally:
                H5Tclose(array_id)

        integer_cases = (
            (
                "signed_i16",
                "H5T_STD_I16LE_g",
                "H5T_STD_I16BE_g",
                C.c_int16,
                -0x1234,
            ),
            (
                "signed_i32",
                "H5T_STD_I32LE_g",
                "H5T_STD_I32BE_g",
                C.c_int32,
                -0x12345678,
            ),
            (
                "signed_i64",
                "H5T_STD_I64LE_g",
                "H5T_STD_I64BE_g",
                C.c_int64,
                -0x123456789ABCDEF,
            ),
            (
                "unsigned_u16",
                "H5T_STD_U16LE_g",
                "H5T_STD_U16BE_g",
                C.c_uint16,
                0xCDEF,
            ),
            (
                "unsigned_u32",
                "H5T_STD_U32LE_g",
                "H5T_STD_U32BE_g",
                C.c_uint32,
                0x89ABCDEF,
            ),
            (
                "unsigned_u64",
                "H5T_STD_U64LE_g",
                "H5T_STD_U64BE_g",
                C.c_uint64,
                0xF23456789ABCDEF0,
            ),
        )
        for stem, little, big, ctype, value in integer_cases:
            add_scalar(f"{stem}_le", little, ctype, value)
            add_scalar(f"{stem}_be", big, ctype, value)

        float_cases = (
            (
                "float32",
                "H5T_IEEE_F32LE_g",
                "H5T_IEEE_F32BE_g",
                C.c_float,
                1234.5,
            ),
            (
                "float64",
                "H5T_IEEE_F64LE_g",
                "H5T_IEEE_F64BE_g",
                C.c_double,
                -98765.25,
            ),
            (
                "float32_negzero",
                "H5T_IEEE_F32LE_g",
                "H5T_IEEE_F32BE_g",
                C.c_float,
                -0.0,
            ),
            (
                "float64_negzero",
                "H5T_IEEE_F64LE_g",
                "H5T_IEEE_F64BE_g",
                C.c_double,
                -0.0,
            ),
            (
                "float32_inf",
                "H5T_IEEE_F32LE_g",
                "H5T_IEEE_F32BE_g",
                C.c_float,
                float("inf"),
            ),
            (
                "float64_inf",
                "H5T_IEEE_F64LE_g",
                "H5T_IEEE_F64BE_g",
                C.c_double,
                float("inf"),
            ),
            (
                "float32_nan",
                "H5T_IEEE_F32LE_g",
                "H5T_IEEE_F32BE_g",
                C.c_float,
                float("nan"),
            ),
            (
                "float64_nan",
                "H5T_IEEE_F64LE_g",
                "H5T_IEEE_F64BE_g",
                C.c_double,
                float("nan"),
            ),
        )
        for stem, little, big, ctype, value in float_cases:
            add_scalar(f"{stem}_le", little, ctype, value)
            add_scalar(f"{stem}_be", big, ctype, value)

        array_cases = (
            (
                "signed_i32_be_array",
                "H5T_STD_I32BE_g",
                C.c_int32,
                (-0x12345678, 0, 0x12345678),
            ),
            (
                "unsigned_u64_be_array",
                "H5T_STD_U64BE_g",
                C.c_uint64,
                (0xF23456789ABCDEF0, 0x0123456789ABCDEF),
            ),
            (
                "float32_be_special_array",
                "H5T_IEEE_F32BE_g",
                C.c_float,
                (-0.0, float("inf"), float("nan")),
            ),
            (
                "float64_le_special_array",
                "H5T_IEEE_F64LE_g",
                C.c_double,
                (-0.0, float("inf"), float("nan")),
            ),
        )
        for name, type_name, ctype, values in array_cases:
            add_h5t_array(name, types[type_name], ctype, values)

        def add_narrow_integer(stem, type_name, ctype, scalar, values):
            custom_id = checked(H5Tcopy(types[type_name]), f"copy type {stem}")
            try:
                checked(H5Tset_precision(custom_id, 12), f"precision {stem}")
                checked(H5Tset_offset(custom_id, 4), f"offset {stem}")
                add_attribute(
                    f"{stem}_scalar", custom_id, scalar_space, ctype, (scalar,)
                )
                add_rank_array(f"{stem}_rank_array", custom_id, ctype, values)
                add_h5t_array(f"{stem}_h5t_array", custom_id, ctype, values)
            finally:
                H5Tclose(custom_id)

        add_narrow_integer(
            "narrow_i64",
            "H5T_STD_I64LE_g",
            C.c_int16,
            -0x345,
            (-0x345, 0, 0x456),
        )
        add_narrow_integer(
            "narrow_u64",
            "H5T_STD_U64LE_g",
            C.c_uint16,
            0xABC,
            (0xABC, 0x123, 0x456),
        )

        H5Sclose(scalar_space)
        H5Oclose(root)
        H5Fclose(fid)

        print(
            json.dumps(
                {
                    "output": str(output),
                    "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                    "size": output.stat().st_size,
                    "hdf5": library,
                    "attribute_count": attribute_count,
                    "host_byteorder": __import__("sys").byteorder,
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        if dll_dir is not None:
            dll_dir.close()


if __name__ == "__main__":
    main()
