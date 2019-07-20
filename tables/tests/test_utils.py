import sys
from io import StringIO

try:
    from unittest.mock import patch
except ImportError:
    from mock import patch

from tables.tests import common
from tables.tests.common import unittest
from tables.tests.common import PyTablesTestCase as TestCase

import tables.scripts.ptrepack as ptrepack
import tables.scripts.ptdump as ptdump
import tables.scripts.pttree as pttree


class ptrepackTestCase(TestCase):
    """Test ptrepack"""

    @patch.object(ptrepack, 'copy_leaf')
    @patch.object(ptrepack, 'open_file')
    def test_paths_windows(self, mock_open_file, mock_copy_leaf):
        """Checking handling of windows filenames: test gh-616"""

        # this filename has a semi-colon to check for
        # regression of gh-616
        src_fn = 'D:\\window~1\\path\\000\\infile'
        src_path = '/'
        dst_fn = 'another\\path\\'
        dst_path = '/path/in/outfile'

        argv = ['ptrepack', src_fn + ':' + src_path, dst_fn + ':' + dst_path]
        with patch.object(sys, 'argv', argv):
            ptrepack.main()

        args, kwargs = mock_open_file.call_args_list[0]
        self.assertEqual(args, (src_fn, 'r'))

        args, kwargs = mock_copy_leaf.call_args_list[0]
        self.assertEqual(args, (src_fn, dst_fn, src_path, dst_path))


class ptdumpTestCase(TestCase):
    """Test ptdump"""

    @patch.object(ptdump, 'open_file')
    @patch('sys.stdout', new_callable=StringIO)
    def test_paths_windows(self, _, mock_open_file):
        """Checking handling of windows filenames: test gh-616"""

        # this filename has a semi-colon to check for
        # regression of gh-616 (in ptdump)
        src_fn = 'D:\\window~1\\path\\000\\ptdump'
        src_path = '/'

        argv = ['ptdump', src_fn + ':' + src_path]
        with patch.object(sys, 'argv', argv):
            ptdump.main()

        args, kwargs = mock_open_file.call_args_list[0]
        self.assertEqual(args, (src_fn, 'r'))


class pttreeTestCase(TestCase):
    """Test ptdump"""

    @patch.object(pttree.tables, 'open_file')
    @patch.object(pttree, 'get_tree_str')
    @patch('sys.stdout', new_callable=StringIO)
    def test_paths_windows(self, _, mock_get_tree_str, mock_open_file):
        """Checking handling of windows filenames: test gh-616"""

        # this filename has a semi-colon to check for
        # regression of gh-616 (in pttree)
        src_fn = 'D:\\window~1\\path\\000\\pttree'
        src_path = '/'

        argv = ['pttree', src_fn + ':' + src_path]
        with patch.object(sys, 'argv', argv):
            pttree.main()

        args, kwargs = mock_open_file.call_args_list[0]
        self.assertEqual(args, (src_fn, 'r'))


def suite():
    theSuite = unittest.TestSuite()

    theSuite.addTest(unittest.makeSuite(ptrepackTestCase))
    theSuite.addTest(unittest.makeSuite(ptdumpTestCase))
    theSuite.addTest(unittest.makeSuite(pttreeTestCase))

    return theSuite

if __name__ == '__main__':
    common.parse_argv(sys.argv)
    common.print_versions()
    unittest.main(defaultTest='suite')
