import unittest
import sys

from tables import utils


class SizeOfRecursiveTestCase(unittest.TestCase):

    def comparison(self, obj, expected_size):
        self.assertEqual(utils.sizeof_recursive(obj), expected_size)

    def test_int(self):
        obj = 1
        expected_size = sys.getsizeof(obj)
        self.comparison(obj, expected_size)

    def test_string(self):
        obj = 'a'
        expected_size = sys.getsizeof(obj)
        self.comparison(obj, expected_size)

    def test_list(self):
        obj = [1, 2]
        expected_size = sum([sys.getsizeof(item) for item in [obj, 1, 2]])
        self.comparison(obj, expected_size)

    def test_dict(self):
        obj = {'a':1, 'b':'12345'}
        expected_size = sum([sys.getsizeof(item) for item in [obj, 'a', 'b',
                                                              1, '12345']])
        self.comparison(obj, expected_size)

    def test_double_nested(self):
        obj = [1, [2, 3]]
        expected_size = sum([sys.getsizeof(item) for item in [obj, 1, [2, 3],
                                                              2, 3]])
        self.comparison(obj, expected_size)
        

def suite():
    theSuite = unittest.TestSuite()
    niter = 1

    for i in range(niter):
        theSuite.addTest(unittest.makeSuite(SizeOfRecursiveTestCase))

    return theSuite


if __name__ == '__main__':
    unittest.main( defaultTest='suite' )
