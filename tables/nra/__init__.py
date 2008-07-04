"""
Nested record array implementation for numarray
===============================================

:Author:   Ivan Vilata i Balaguer
:Contact:  ivan@selidor.net
:Created:  2007-01-12
:License:  BSD
:Revision: $Id$

This package provides the `NestedRecArray` and `NestedRecord` classes,
which can be used to handle arrays of nested records in a way which is
compatible with ``numarray.records``.  Several utility functions are
provided for creating nested record arrays.
"""

from tables.nra.nestedrecords import (
    NestedRecArray, NestedRecord, array, fromarrays, fromnumpy )

__all__ = [
    'NestedRecArray', 'NestedRecord',
    'array', 'fromarrays', 'fromnumpy' ]
