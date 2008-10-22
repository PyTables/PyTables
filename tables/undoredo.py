########################################################################
#
#       License: BSD
#       Created: February 15, 2005
#       Author:  Ivan Vilata - reverse:net.selidor@ivan
#
#       $Source$
#       $Id$
#
########################################################################

"""
Support for undoing and redoing actions.

Functions:

* undo(file, operation, *args)
* redo(file, operation, *args)
* moveToShadow(file, path)
* moveFromShadow(file, path)
* attrToShadow(file, path, name)
* attrFromShadow(file, path, name)

Misc variables:

`__docformat__`
    The format of documentation strings in this module.
`__version__`
    Repository version of this file.
"""

from tables.path import splitPath



__docformat__ = 'reStructuredText'
"""The format of documentation strings in this module."""

__version__ = '$Revision$'
"""Repository version of this file."""



def undo(file_, operation, *args):
    if operation == 'CREATE':
        undoCreate(file_, args[0])
    elif operation == 'REMOVE':
        undoRemove(file_, args[0])
    elif operation == 'MOVE':
        undoMove(file_, args[0], args[1])
    elif operation == 'ADDATTR':
        undoAddAttr(file_, args[0], args[1])
    elif operation == 'DELATTR':
        undoDelAttr(file_, args[0], args[1])
    else:
        raise NotImplementedError("""\
the requested unknown operation %r can not be undone; \
please report this to the authors""" % operation)


def redo(file_, operation, *args):
    if operation == 'CREATE':
        redoCreate(file_, args[0])
    elif operation == 'REMOVE':
        redoRemove(file_, args[0])
    elif operation == 'MOVE':
        redoMove(file_, args[0], args[1])
    elif operation == 'ADDATTR':
        redoAddAttr(file_, args[0], args[1])
    elif operation == 'DELATTR':
        redoDelAttr(file_, args[0], args[1])
    else:
        raise NotImplementedError("""\
the requested unknown operation %r can not be redone; \
please report this to the authors""" % operation)


def moveToShadow(file_, path):
    node = file_._getNode(path)

    (shparent, shname) = file_._shadowName()
    node._g_move(shparent, shname)


def moveFromShadow(file_, path):
    (shparent, shname) = file_._shadowName()
    node = shparent._f_getChild(shname)

    (pname, name) = splitPath(path)
    parent = file_._getNode(pname)
    node._g_move(parent, name)


def undoCreate(file_, path):
    moveToShadow(file_, path)

def redoCreate(file_, path):
    moveFromShadow(file_, path)

def undoRemove(file_, path):
    moveFromShadow(file_, path)

def redoRemove(file_, path):
    moveToShadow(file_, path)

def undoMove(file_, origpath, destpath):
    (origpname, origname) = splitPath(origpath)

    node = file_._getNode(destpath)
    origparent = file_._getNode(origpname)
    node._g_move(origparent, origname)

def redoMove(file_, origpath, destpath):
    (destpname, destname) = splitPath(destpath)

    node = file_._getNode(origpath)
    destparent = file_._getNode(destpname)
    node._g_move(destparent, destname)


def attrToShadow(file_, path, name):
    node = file_._getNode(path)
    attrs = node._v_attrs
    value = getattr(attrs, name)

    (shparent, shname) = file_._shadowName()
    shattrs = shparent._v_attrs

    # Set the attribute only if it has not been kept in the shadow.
    # This avoids re-pickling complex attributes on REDO.
    if not shname in shattrs:
        shattrs._g__setattr(shname, value)

    attrs._g__delattr(name)


def attrFromShadow(file_, path, name):
    (shparent, shname) = file_._shadowName()
    shattrs = shparent._v_attrs
    value = getattr(shattrs, shname)

    node = file_._getNode(path)
    node._v_attrs._g__setattr(name, value)

    # Keeping the attribute in the shadow allows reusing it on Undo/Redo.
    ##shattrs._g__delattr(shname)


def undoAddAttr(file_, path, name):
    attrToShadow(file_, path, name)

def redoAddAttr(file_, path, name):
    attrFromShadow(file_, path, name)

def undoDelAttr(file_, path, name):
    attrFromShadow(file_, path, name)

def redoDelAttr(file_, path, name):
    attrToShadow(file_, path, name)



## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## End:
