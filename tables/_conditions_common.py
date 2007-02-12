"""
Utility functions and classes for supporting query conditions (common).

:Author:   Ivan Vilata i Balaguer
:Contact:  ivilata@carabos.com
:Created:  2007-02-12
:License:  BSD
:Revision: $Id$
"""

import re
from tables.numexpr.compiler import typecode_to_kind

_no_matching_opcode = re.compile(r"[^a-z]([a-z]+)_([a-z]+)[^a-z]")
# E.g. "gt" and "bfc" from "couldn't find matching opcode for 'gt_bfc'".

def _unsupported_operation_error(exception):
    """
    Make the \"no matching opcode\" Numexpr `exception` more clear.

    A new exception of the same kind is returned.
    """
    message = exception.args[0]
    op, types = _no_matching_opcode.search(message).groups()
    newmessage = "unsupported operand types for *%s*: " % op
    newmessage += ', '.join([typecode_to_kind[t] for t in types[1:]])
    return exception.__class__(newmessage)
