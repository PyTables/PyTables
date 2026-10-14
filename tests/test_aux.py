import unittest

import numpy as np

import tables as tb
from tests import common


class TestAuxiliaryFunctions(unittest.TestCase):
    def test_keysort(self):
        N = 1000
        rnd = np.random.randint(N, size=N)
        for dtype1 in ("S6", "b1", "i1", "i8", "u4", "u8", "f4", "f8"):
            for dtype2 in ("u4", "i8"):
                a = np.array(rnd, dtype1)
                b = np.array(rnd, dtype2)

                c = a.copy()
                d = c.argsort()
                e = c[d]
                f = b[d]

                tb.indexesextension.keysort(a, b)
                self.assertTrue((a == e).all())
                self.assertTrue((b == f).all())

    def test_keysort_short_array2(self):
        # keysort reorders the first len(array1) elements of array2, so a
        # shorter array2 used to be read and written out of bounds.  It must
        # be rejected instead.
        for dtype1 in ("i8", "f8", "S6"):
            a = np.arange(100).astype(dtype1)
            b = np.zeros(1, dtype="i8")
            with self.assertRaises(ValueError):
                tb.indexesextension.keysort(a, b)


def suite():
    theSuite = unittest.TestSuite()
    theSuite.addTest(common.make_suite(TestAuxiliaryFunctions))
    return theSuite


if __name__ == "__main__":
    unittest.main(defaultTest="suite")
