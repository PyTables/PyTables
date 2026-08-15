"""Security and compatibility tests for automatic pickle loading."""

import os
import sys
import shlex
import pickle
from pathlib import Path
from unittest import mock

import numpy as np

import tables as tb
import tables.ptpickle as ptpickle
from tables.tests import common


class _TouchFile:
    """Pickle payload used to detect unintended command execution."""

    def __init__(self, path):
        self.path = os.fspath(path)

    def __reduce__(self):
        if os.name == "nt":
            command = f'cmd /c type nul > "{self.path}"'
        else:
            command = f"touch {shlex.quote(self.path)}"
        return os.system, (command,)


class _LegacyObjectAtom(tb.ObjectAtom):
    """Object atom using the historical ``fromarray(array)`` signature."""

    def fromarray(self, array):
        return super().fromarray(array)


class _DirectLegacyObjectAtom(tb.ObjectAtom):
    """Historical override that performs loading itself."""

    def __init__(self):
        self.called = False

    def fromarray(self, array):
        self.called = True
        return pickle.loads(array.tobytes())


class PickleSecurityTestCase(common.TempFileMixin, common.PyTablesTestCase):
    def setUp(self):
        super().setUp()
        self.sentinel = Path(f"{self.h5fname}.sentinel")

    def tearDown(self):
        self.sentinel.unlink(missing_ok=True)
        super().tearDown()

    def _malicious_payload(self):
        payload = pickle.dumps(_TouchFile(self.sentinel), protocol=0)
        module = os.system.__module__.encode("ascii")
        self.assertIn(b"c" + module + b"\nsystem\n", payload)
        self.assertTrue(payload.endswith(b"."))
        return payload

    def _store_attribute_payload(self, payload):
        self.h5file.root._v_attrs.payload = np.bytes_(payload)

    def _store_object_atom_payload(self, payload):
        vlarray = self.h5file.create_vlarray(
            "/", "payload", atom=tb.UInt8Atom()
        )
        vlarray.append(np.frombuffer(payload, dtype=np.uint8))
        vlarray.attrs.PSEUDOATOM = "object"

    def _reopen_trusted(self):
        with self.assertWarnsRegex(
            tb.PickleSecurityWarning, "only open files you trust"
        ):
            self._reopen("r", allow_pickle=True)

    def test_attribute_payload_is_not_loaded_by_default(self):
        payload = self._malicious_payload()
        self._store_attribute_payload(payload)

        self._reopen("r")

        actual = self.h5file.root._v_attrs.payload
        self.assertFalse(self.sentinel.exists())
        self.assertEqual(bytes(actual), payload)

    def test_plain_bytes_ending_in_dot_remain_bytes(self):
        payload = b"this is not a pickle."
        self._store_attribute_payload(payload)

        self._reopen("r")

        self.assertEqual(bytes(self.h5file.root._v_attrs.payload), payload)

        self._reopen_trusted()
        self.assertEqual(bytes(self.h5file.root._v_attrs.payload), payload)

    def test_serialization_remains_enabled_when_loading_is_disabled(self):
        expected = {"answer": 42}
        with self.assertWarnsRegex(
            tb.PickleSecurityWarning, "requires unpickling"
        ):
            payload = ptpickle.dumps(expected)
        self._store_attribute_payload(payload)

        self._reopen("r")
        actual = self.h5file.root._v_attrs.payload
        self.assertIsInstance(actual, np.bytes_)
        self.assertEqual(bytes(actual), payload)

        self._reopen_trusted()
        self.assertEqual(self.h5file.root._v_attrs.payload, expected)

    def test_trusted_attribute_round_trip(self):
        expected = {"answer": 42}
        self._store_attribute_payload(pickle.dumps(expected, protocol=0))

        self._reopen_trusted()

        self.assertEqual(self.h5file.root._v_attrs.payload, expected)

    def test_object_atom_payload_is_not_loaded_by_default(self):
        self._store_object_atom_payload(self._malicious_payload())

        self._reopen("r")

        node = self.h5file.root.payload
        accessors = {
            "read": node.read,
            "index": lambda: node[0],
            "iteration": lambda: next(iter(node)),
        }
        for name, accessor in accessors.items():
            with self.subTest(accessor=name):
                try:
                    with self.assertRaisesRegex(
                        tb.PickleNotAllowedError, "allow_pickle=True"
                    ):
                        accessor()
                finally:
                    self.assertFalse(self.sentinel.exists())

    def test_trusted_object_atom_round_trip(self):
        expected = {"answer": 42}
        vlarray = self.h5file.create_vlarray(
            "/", "objects", atom=tb.ObjectAtom()
        )
        with self.assertWarnsRegex(
            tb.PickleSecurityWarning, "requires unpickling"
        ):
            vlarray.append(expected)

        self._reopen("r")
        with self.assertRaises(tb.PickleNotAllowedError):
            self.h5file.root.objects.read()

        self._reopen_trusted()
        self.assertEqual(self.h5file.root.objects.read(), [expected])

    def test_policy_is_scoped_to_each_file_handle(self):
        expected = {"answer": 42}
        payload = pickle.dumps(expected, protocol=0)
        self._store_attribute_payload(payload)
        self.h5file.close()

        safe_file = tb.open_file(self.h5fname, "r")
        with self.assertWarns(tb.PickleSecurityWarning):
            trusted_file = tb.open_file(self.h5fname, "r", allow_pickle=True)
        self.h5file = safe_file
        try:
            self.assertEqual(bytes(safe_file.root._v_attrs.payload), payload)
            self.assertEqual(trusted_file.root._v_attrs.payload, expected)
        finally:
            trusted_file.close()

    def test_explicit_false_overrides_global_true(self):
        expected = {"answer": 42}
        payload = pickle.dumps(expected, protocol=0)
        self._store_attribute_payload(payload)
        self.h5file.close()

        with mock.patch.object(tb.parameters, "ALLOW_PICKLE", True):
            self.h5file = tb.open_file(self.h5fname, "r", allow_pickle=False)

        self.assertEqual(bytes(self.h5file.root._v_attrs.payload), payload)

    def test_global_true_applies_to_new_file_handles(self):
        expected = {"answer": 42}
        self._store_attribute_payload(pickle.dumps(expected, protocol=0))
        self.h5file.close()

        with mock.patch.object(tb.parameters, "ALLOW_PICKLE", True):
            with self.assertWarns(tb.PickleSecurityWarning):
                self.h5file = tb.open_file(self.h5fname, "r")

        self.assertEqual(self.h5file.root._v_attrs.payload, expected)

    def test_global_change_does_not_affect_open_file_handle(self):
        vlarray = self.h5file.create_vlarray(
            "/", "objects", atom=tb.ObjectAtom()
        )
        with self.assertWarns(tb.PickleSecurityWarning):
            vlarray.append({"answer": 42})

        with mock.patch.object(tb.parameters, "ALLOW_PICKLE", True):
            with self.assertRaises(tb.PickleNotAllowedError):
                vlarray.read()

    def test_unused_buffers_keyword_is_not_forwarded(self):
        expected = {"answer": 42}
        payload = pickle.dumps(expected)
        stdlib_loads = pickle.loads

        def loads_without_buffers(
            data, *, fix_imports=True, encoding="ASCII", errors="strict"
        ):
            return stdlib_loads(
                data,
                fix_imports=fix_imports,
                encoding=encoding,
                errors=errors,
            )

        with mock.patch.object(
            ptpickle._pickle, "loads", side_effect=loads_without_buffers
        ):
            actual = ptpickle.loads(payload, allow_pickle=True)

        self.assertEqual(actual, expected)

    def test_low_level_loader_is_disabled_by_default(self):
        payload = self._malicious_payload()

        try:
            with mock.patch.object(tb.parameters, "ALLOW_PICKLE", True):
                with self.assertRaises(tb.PickleNotAllowedError):
                    ptpickle.loads(payload)
        finally:
            self.assertFalse(self.sentinel.exists())

    def test_legacy_object_atom_override(self):
        expected = {"answer": 42}
        atom = _DirectLegacyObjectAtom()
        vlarray = self.h5file.create_vlarray("/", "safe_objects", atom=atom)
        with self.assertWarns(tb.PickleSecurityWarning):
            vlarray.append(expected)

        with self.assertRaises(tb.PickleNotAllowedError):
            vlarray.read()
        self.assertFalse(atom.called)

        self.h5file.close()
        with self.assertWarns(tb.PickleSecurityWarning):
            self.h5file = tb.open_file(self.h5fname, "w", allow_pickle=True)
        vlarray = self.h5file.create_vlarray(
            "/", "trusted_objects", atom=_LegacyObjectAtom()
        )
        with self.assertWarns(tb.PickleSecurityWarning):
            vlarray.append(expected)

        self.assertEqual(vlarray.read(), [expected])
        with self.assertWarns(tb.PickleSecurityWarning):
            raw = tb.ObjectAtom().toarray(expected)
        with self.assertRaises(tb.PickleNotAllowedError):
            tb.ObjectAtom().fromarray(raw)

    def test_allow_pickle_must_be_bool(self):
        self.h5file.close()

        with self.assertRaisesRegex(TypeError, "allow_pickle must be a bool"):
            tb.open_file(self.h5fname, "r", allow_pickle="yes")


def suite():
    test_suite = common.unittest.TestSuite()
    test_suite.addTest(common.make_suite(PickleSecurityTestCase))
    return test_suite


if __name__ == "__main__":
    common.parse_argv(sys.argv)
    common.print_versions()
    common.unittest.main(defaultTest="suite")
