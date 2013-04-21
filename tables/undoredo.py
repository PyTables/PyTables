# -*- coding: utf-8 -*-

########################################################################
#
# License: BSD
# Created: February 15, 2005
# Author:  Ivan Vilata - reverse:net.selidor@ivan
#
# $Source$
# $Id$
#
########################################################################

"""Support for undoing and redoing actions.

Functions:

* undo(file, operation, *args)
* redo(file, operation, *args)
* move_to_shadow(file, path)
* move_from_shadow(file, path)
* attr_to_shadow(file, path, name)
* attr_from_shadow(file, path, name)

Misc variables:

`__docformat__`
    The format of documentation strings in this module.

"""

from tables.path import split_path
from tables._past import previous_api


__docformat__ = 'reStructuredText'
"""The format of documentation strings in this module."""


def undo(file_, operation, *args):
    if operation == 'CREATE':
        undo_create(file_, args[0])
    elif operation == 'REMOVE':
        undo_remove(file_, args[0])
    elif operation == 'MOVE':
        undo_move(file_, args[0], args[1])
    elif operation == 'ADDATTR':
        undo_add_attr(file_, args[0], args[1])
    elif operation == 'DELATTR':
        undo_del_attr(file_, args[0], args[1])
    else:
        raise NotImplementedError("the requested unknown operation %r can "
                                  "not be undone; please report this to the "
                                  "authors" % operation)


def redo(file_, operation, *args):
    if operation == 'CREATE':
        redo_create(file_, args[0])
    elif operation == 'REMOVE':
        redo_remove(file_, args[0])
    elif operation == 'MOVE':
        redo_move(file_, args[0], args[1])
    elif operation == 'ADDATTR':
        redo_add_attr(file_, args[0], args[1])
    elif operation == 'DELATTR':
        redo_del_attr(file_, args[0], args[1])
    else:
        raise NotImplementedError("the requested unknown operation %r can "
                                  "not be redone; please report this to the "
                                  "authors" % operation)


def move_to_shadow(file_, path):
    node = file_._get_node(path)

    (shparent, shname) = file_._shadow_name()
    node._g_move(shparent, shname)

moveToShadow = previous_api(move_to_shadow)


def move_from_shadow(file_, path):
    (shparent, shname) = file_._shadow_name()
    node = shparent._f_get_child(shname)

    (pname, name) = split_path(path)
    parent = file_._get_node(pname)
    node._g_move(parent, name)

moveFromShadow = previous_api(move_from_shadow)


def undo_create(file_, path):
    move_to_shadow(file_, path)

undoCreate = previous_api(undo_create)


def redo_create(file_, path):
    move_from_shadow(file_, path)

redoCreate = previous_api(redo_create)


def undo_remove(file_, path):
    move_from_shadow(file_, path)

undoRemove = previous_api(undo_remove)


def redo_remove(file_, path):
    move_to_shadow(file_, path)

redoRemove = previous_api(redo_remove)


def undo_move(file_, origpath, destpath):
    (origpname, origname) = split_path(origpath)

    node = file_._get_node(destpath)
    origparent = file_._get_node(origpname)
    node._g_move(origparent, origname)

undoMove = previous_api(undo_move)


def redo_move(file_, origpath, destpath):
    (destpname, destname) = split_path(destpath)

    node = file_._get_node(origpath)
    destparent = file_._get_node(destpname)
    node._g_move(destparent, destname)

redoMove = previous_api(redo_move)


def attr_to_shadow(file_, path, name):
    node = file_._get_node(path)
    attrs = node._v_attrs
    value = getattr(attrs, name)

    (shparent, shname) = file_._shadow_name()
    shattrs = shparent._v_attrs

    # Set the attribute only if it has not been kept in the shadow.
    # This avoids re-pickling complex attributes on REDO.
    if not shname in shattrs:
        shattrs._g__setattr(shname, value)

    attrs._g__delattr(name)

attrToShadow = previous_api(attr_to_shadow)


def attr_from_shadow(file_, path, name):
    (shparent, shname) = file_._shadow_name()
    shattrs = shparent._v_attrs
    value = getattr(shattrs, shname)

    node = file_._get_node(path)
    node._v_attrs._g__setattr(name, value)

    # Keeping the attribute in the shadow allows reusing it on Undo/Redo.
    # shattrs._g__delattr(shname)

attrFromShadow = previous_api(attr_from_shadow)


def undo_add_attr(file_, path, name):
    attr_to_shadow(file_, path, name)

undoAddAttr = previous_api(undo_add_attr)


def redo_add_attr(file_, path, name):
    attr_from_shadow(file_, path, name)

redoAddAttr = previous_api(redo_add_attr)


def undo_del_attr(file_, path, name):
    attr_from_shadow(file_, path, name)

undoDelAttr = previous_api(undo_del_attr)


def redo_del_attr(file_, path, name):
    attr_to_shadow(file_, path, name)

redoDelAttr = previous_api(redo_del_attr)


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## End:
