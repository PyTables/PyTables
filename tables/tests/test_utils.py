import sys
from io import StringIO
from unittest.mock import patch

from tables.tests import common
from tables.utils import print_versions
from tables.scripts import ptdump, pttree, ptrepack


class PTRepackTestCase(common.PyTablesTestCase):
    """Test ptrepack"""

    @patch.object(ptrepack, "copy_leaf")
    @patch.object(ptrepack.tb, "open_file")
    def test_paths_windows(self, mock_open_file, mock_copy_leaf):
        """Checking handling of windows filenames: test gh-616"""

        # this filename has a semi-colon to check for
        # regression of gh-616
        src_fn = "D:\\window~1\\path\\000\\infile"
        src_path = "/"
        dst_fn = "another\\path\\"
        dst_path = "/path/in/outfile"

        argv = ["ptrepack", src_fn + ":" + src_path, dst_fn + ":" + dst_path]
        with patch.object(sys, "argv", argv):
            ptrepack.main()

        args, _ = mock_open_file.call_args_list[0]
        self.assertEqual(args, (src_fn, "r"))

        args, _ = mock_copy_leaf.call_args_list[0]
        self.assertEqual(args, (src_fn, dst_fn, src_path, dst_path))


class PTDumpTestCase(common.PyTablesTestCase):
    """Test ptdump"""

    def setUp(self):
        super().setUp()
        ptdump.options.allow_pickle = False

    @patch.object(ptdump.tb, "open_file")
    @patch("sys.stdout", new_callable=StringIO)
    def test_paths_windows(self, _, mock_open_file):
        """Checking handling of windows filenames: test gh-616"""

        # this filename has a semi-colon to check for
        # regression of gh-616 (in ptdump)
        src_fn = "D:\\window~1\\path\\000\\ptdump"
        src_path = "/"

        argv = ["ptdump", src_fn + ":" + src_path]
        with patch.object(sys, "argv", argv):
            ptdump.main()

        args, kwargs = mock_open_file.call_args_list[0]
        self.assertEqual(args, (src_fn, "r"))
        self.assertEqual(kwargs, {"allow_pickle": False})

    @patch.object(ptdump.tb, "open_file")
    @patch("sys.stdout", new_callable=StringIO)
    def test_allow_pickle(self, _, mock_open_file):
        """Checking the trusted-file command-line opt-in."""

        src_fn = "trusted.h5"
        argv = ["ptdump", "--allow-pickle", src_fn]
        with patch.object(sys, "argv", argv):
            ptdump.main()

        mock_open_file.assert_called_once_with(src_fn, "r", allow_pickle=True)


class PTTreeTestCase(common.PyTablesTestCase):
    """Test ptdump"""

    @patch.object(pttree.tb, "open_file")
    @patch.object(pttree, "get_tree_str")
    @patch("sys.stdout", new_callable=StringIO)
    def test_paths_windows(self, _, mock_get_tree_str, mock_open_file):
        """Checking handling of windows filenames: test gh-616"""

        # this filename has a semi-colon to check for
        # regression of gh-616 (in pttree)
        src_fn = "D:\\window~1\\path\\000\\pttree"
        src_path = "/"

        argv = ["pttree", src_fn + ":" + src_path]
        with patch.object(sys, "argv", argv):
            pttree.main()

        args, _ = mock_open_file.call_args_list[0]
        self.assertEqual(args, (src_fn, "r"))


def suite():
    theSuite = common.unittest.TestSuite()

    theSuite.addTest(common.make_suite(PTRepackTestCase))
    theSuite.addTest(common.make_suite(PTDumpTestCase))
    theSuite.addTest(common.make_suite(PTTreeTestCase))

    return theSuite


if __name__ == "__main__":
    print_versions()
    common.parse_argv(sys.argv)
    common.unittest.main(defaultTest="suite")
