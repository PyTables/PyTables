# -*- coding: utf-8 -*-

########################################################################
#
# License: BSD
# Created: 2005-12-02
# Author: Ivan Vilata i Balaguer - ivan@selidor.net
#
# $Id$
#
########################################################################

"""Unit tests for PyTables.

This package contains some modules which provide a ``suite()`` function
(with no arguments) which returns a test suite for some PyTables
functionality.

"""
from __future__ import absolute_import

from tables.tests.common import print_versions
from tables.tests.test_all import test, suite
