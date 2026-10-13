"""Create the scalar/array enum compatibility fixture used by test_hdf5compat.

Usage: python create_enum_attr_types.py OUTPUT.h5 [--hdf5 PATH_TO_HDF5_LIBRARY]
The output path is explicit and H5Fcreate replaces only that requested file.
"""

from __future__ import annotations

import os
import sys
import ctypes as C
import struct
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
        library = str(args.hdf5)
        if os.name == "nt" and hasattr(os, "add_dll_directory"):
            dll_dir = os.add_dll_directory(str(args.hdf5.parent))
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
        H5Sclose = fn("H5Sclose", herr, hid)
        H5Tclose = fn("H5Tclose", herr, hid)
        H5Tcreate = fn("H5Tcreate", hid, C.c_int, C.c_size_t)
        H5Tset_order = fn("H5Tset_order", herr, hid, C.c_int)
        H5Tset_precision = fn("H5Tset_precision", herr, hid, C.c_size_t)
        H5Tset_sign = fn("H5Tset_sign", herr, hid, C.c_int)
        H5Tenum_create = fn("H5Tenum_create", hid, hid)
        H5Tenum_insert = fn(
            "H5Tenum_insert", herr, hid, C.c_char_p, C.c_void_p
        )
        H5Tget_native_type = fn("H5Tget_native_type", hid, hid, C.c_int)
        try:
            H5Tarray_create = fn(
                "H5Tarray_create2", hid, hid, C.c_uint, C.POINTER(hsize)
            )
        except AttributeError:
            H5Tarray_create = fn(
                "H5Tarray_create", hid, hid, C.c_int, C.POINTER(hsize)
            )
        H5Acreate2 = fn("H5Acreate2", hid, hid, C.c_char_p, hid, hid, hid, hid)
        H5Awrite = fn("H5Awrite", herr, hid, hid, C.c_void_p)
        H5Aclose = fn("H5Aclose", herr, hid)

        H5open()
        H5P_DEFAULT = 0
        H5F_ACC_TRUNC = 2
        H5S_SCALAR = 0
        H5T_INTEGER = 0
        H5T_ORDER_LE = 0
        H5T_SGN_2 = 1
        native = {
            name: C.c_int64.in_dll(h5, name).value
            for name in (
                "H5T_STD_I16BE_g",
                "H5T_STD_I32LE_g",
                "H5T_STD_U16BE_g",
                "H5T_STD_U32LE_g",
            )
        }

        def add_scalar(root, space, name, base_id, fmt, members, value):
            enum_id = checked(H5Tenum_create(base_id), f"enum {name}")
            native_id = -1
            attr_id = -1
            try:
                for member_name, member_value in members:
                    raw = struct.pack(fmt, member_value)
                    raw_buffer = C.create_string_buffer(raw)
                    checked(
                        H5Tenum_insert(
                            enum_id, member_name.encode(), raw_buffer
                        ),
                        f"member {name}/{member_name}",
                    )
                attr_id = checked(
                    H5Acreate2(root, name.encode(), enum_id, space, 0, 0, 0),
                    f"attribute {name}",
                )
                native_id = checked(
                    H5Tget_native_type(enum_id, 0), f"native type {name}"
                )
                raw_value = struct.pack("=" + fmt[-1], value)
                value_buffer = C.create_string_buffer(raw_value)
                checked(
                    H5Awrite(attr_id, native_id, value_buffer), f"write {name}"
                )
            finally:
                if attr_id >= 0:
                    H5Aclose(attr_id)
                if native_id >= 0:
                    H5Tclose(native_id)
                H5Tclose(enum_id)

        output = args.output
        if not output.parent.exists():
            raise FileNotFoundError(output.parent)
        fid = checked(
            H5Fcreate(
                os.fsencode(output), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT
            ),
            "file",
        )
        root = -1
        space = -1
        try:
            root = checked(H5Oopen(fid, b"/", H5P_DEFAULT), "root")
            space = checked(H5Screate(H5S_SCALAR), "scalar dataspace")
            add_scalar(
                root,
                space,
                "signed_be16",
                native["H5T_STD_I16BE_g"],
                ">h",
                (("ZERO", 0), ("VALUE", 0x1234)),
                0x1234,
            )
            add_scalar(
                root,
                space,
                "signed_le32",
                native["H5T_STD_I32LE_g"],
                "<i",
                (("ZERO", 0), ("VALUE", 0x12345678)),
                0x12345678,
            )
            add_scalar(
                root,
                space,
                "unsigned_be16",
                native["H5T_STD_U16BE_g"],
                ">H",
                (("ZERO", 0), ("VALUE", 0xFEDC)),
                0xFEDC,
            )
            add_scalar(
                root,
                space,
                "unsigned_le32",
                native["H5T_STD_U32LE_g"],
                "<I",
                (("ZERO", 0), ("VALUE", 0xFEDCBA98)),
                0xFEDCBA98,
            )

            array_base = native["H5T_STD_U16BE_g"]
            array_enum = checked(H5Tenum_create(array_base), "array enum")
            array_type = -1
            native_array_type = -1
            array_attr = -1
            try:
                for member_name, member_value in (
                    (b"ZERO", 0),
                    (b"VALUE", 0xFEDC),
                ):
                    raw_buffer = C.create_string_buffer(
                        struct.pack(">H", member_value)
                    )
                    checked(
                        H5Tenum_insert(array_enum, member_name, raw_buffer),
                        "array member",
                    )
                dimensions = (hsize * 2)(2)
                array_type = checked(
                    H5Tarray_create(array_enum, 1, dimensions),
                    "array datatype",
                )
                array_attr = checked(
                    H5Acreate2(
                        root,
                        b"unsigned_be16_array",
                        array_type,
                        space,
                        0,
                        0,
                        0,
                    ),
                    "array attribute",
                )
                native_array_type = checked(
                    H5Tget_native_type(array_type, 0), "native array datatype"
                )
                values = C.create_string_buffer(
                    struct.pack("=HH", 0x1234, 0xFEDC)
                )
                checked(
                    H5Awrite(array_attr, native_array_type, values),
                    "array write",
                )
            finally:
                if array_attr >= 0:
                    H5Aclose(array_attr)
                if native_array_type >= 0:
                    H5Tclose(native_array_type)
                if array_type >= 0:
                    H5Tclose(array_type)
                H5Tclose(array_enum)

            wide_base = checked(
                H5Tcreate(H5T_INTEGER, 16), "wide integer type"
            )
            wide_enum = -1
            wide_native = -1
            wide_attr = -1
            try:
                checked(
                    H5Tset_order(wide_base, H5T_ORDER_LE), "wide byte order"
                )
                checked(H5Tset_precision(wide_base, 128), "wide precision")
                checked(H5Tset_sign(wide_base, H5T_SGN_2), "wide sign")
                wide_enum = checked(H5Tenum_create(wide_base), "wide enum")
                wide_value = (1).to_bytes(16, sys.byteorder, signed=True)
                raw_value = C.create_string_buffer(wide_value)
                checked(
                    H5Tenum_insert(wide_enum, b"VALUE", raw_value),
                    "wide member",
                )
                wide_attr = checked(
                    H5Acreate2(
                        root, b"signed_i128", wide_enum, space, 0, 0, 0
                    ),
                    "wide attribute",
                )
                wide_native = checked(
                    H5Tget_native_type(wide_enum, 0), "wide native"
                )
                checked(
                    H5Awrite(wide_attr, wide_native, raw_value), "wide write"
                )
            finally:
                if wide_attr >= 0:
                    H5Aclose(wide_attr)
                if wide_native >= 0:
                    H5Tclose(wide_native)
                if wide_enum >= 0:
                    H5Tclose(wide_enum)
                H5Tclose(wide_base)
        finally:
            if space >= 0:
                H5Sclose(space)
            if root >= 0:
                H5Oclose(root)
            H5Fclose(fid)
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        print(f"{output} bytes={output.stat().st_size} sha256={digest}")
    finally:
        if dll_dir is not None:
            dll_dir.close()


if __name__ == "__main__":
    main()
