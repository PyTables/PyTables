########################################################################
#
#       License: BSD
#       Created: December 17, 2004
#       Author:  Francesc Altet - faltet@carabos.com
#
#       $Source: /home/ivan/_/programari/pytables/svn/cvs/pytables/pytables/tables/exceptions.py,v $
#       $Id: exceptions.py,v 1.2 2004/12/24 18:16:02 falted Exp $
#
########################################################################

"""Declare exceptions and warnings that are specific to PyTables"""

class NodeError(AttributeError, LookupError):
    """Accessing a nonexistant node or overwriting an existing ones.

    Nodes in a pytables file cannot simply be overwritten by reassignment. Instead,
    they have to be deleted explicitely before they can created newly. This is done
    to protect interactive users from inadvertedly deleting whole trees of data by
    a single erroneous command."""
    pass

class NaturalNameWarning(Warning):
    """Issued when a non-pythonic name is given for a node.
    This is not an error and may even be very useful in certain contexts,
    but one should be aware that such nodes cannot be accessed using
    natural naming. (Instead, getattr has to be used explicitely.)"""
    pass

