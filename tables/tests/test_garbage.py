# -*- coding: utf-8 -*-

########################################################################
#
# License: BSD
# Created: 2005-09-20
# Author: Ivan Vilata i Balaguer - ivan@selidor.net
#
# $Id$
#
########################################################################

"""Test module for detecting uncollectable garbage in PyTables.

This test module *must* be loaded in the last place.  It just checks for
the existence of uncollectable garbage in ``gc.garbage`` after running
all the tests.

"""

import gc

from tables.tests import common
from tables.tests.common import unittest
from tables.tests.common import PyTablesTestCase as TestCase


class GarbageTestCase(TestCase):
    """Test for uncollectable garbage."""

    def test00(self):
        """Checking for uncollectable garbage."""

        garbageLen = len(gc.garbage)
        if garbageLen == 0:
            return  # success

        if common.verbose:
            classCount = {}
            # Count uncollected objects for each class.
            for obj in gc.garbage:
                objClass = obj.__class__.__name__
                if objClass in classCount:
                    classCount[objClass] += 1
                else:
                    classCount[objClass] = 1
            incidence = ['``%s``: %d' % (cls, cnt)
                         for (cls, cnt) in classCount.items()]
            print("Class incidence:", ', '.join(incidence))
        self.fail("Possible leak: %d uncollected objects." % garbageLen)


def suite():
    """Return a test suite consisting of all the test cases in the module."""

    theSuite = unittest.TestSuite()
    theSuite.addTest(unittest.makeSuite(GarbageTestCase))
    return theSuite


if __name__ == '__main__':
    import sys
    common.parse_argv(sys.argv)
    common.print_versions()
    unittest.main(defaultTest='suite')


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
