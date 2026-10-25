"""Regression tests for runtime-library discovery in setup.py."""

import ctypes
import ctypes.util
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch


PosixPackage = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "setup.py"),
    run_name="pytables_setup_tests",
)["PosixPackage"]


class PosixRuntimeDiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.package = PosixPackage(
            "bzip2", "BZ2", "bzlib", "bz2", "bz2"
        )

    def test_uses_soname_when_unversioned_so_is_linker_script(self):
        def load_library(name):
            if name == "libbz2.so.1":
                return object()
            raise OSError("invalid ELF header")

        with (
            patch("ctypes.util.find_library", return_value="libbz2.so.1"),
            patch.object(ctypes, "CDLL", side_effect=load_library),
        ):
            self.assertTrue(self.package.find_runtime_path(["/usr/lib64"]))

    def test_missing_soname_keeps_explicit_directory_fallback(self):
        def load_library(name):
            if name == "/opt/local/lib/libbz2.so":
                return object()
            raise OSError("library unavailable")

        with (
            patch("ctypes.util.find_library", return_value=None),
            patch.object(ctypes, "CDLL", side_effect=load_library),
        ):
            self.assertTrue(
                self.package.find_runtime_path(["/opt/local/lib"])
            )

    def test_unloadable_soname_keeps_explicit_directory_fallback(self):
        def load_library(name):
            if name == "/opt/local/lib/libbz2.so":
                return object()
            raise OSError("library unavailable")

        with (
            patch("ctypes.util.find_library", return_value="libbz2.so.1"),
            patch.object(ctypes, "CDLL", side_effect=load_library),
        ):
            self.assertTrue(
                self.package.find_runtime_path(["/opt/local/lib"])
            )

    def test_soname_lookup_error_keeps_explicit_directory_fallback(self):
        def load_library(name):
            if name == "/opt/local/lib/libbz2.so":
                return object()
            raise OSError("library unavailable")

        with (
            patch("ctypes.util.find_library", side_effect=OSError("no tmp")),
            patch.object(ctypes, "CDLL", side_effect=load_library),
        ):
            self.assertTrue(
                self.package.find_runtime_path(["/opt/local/lib"])
            )


if __name__ == "__main__":
    unittest.main()
