# -*- coding: utf-8 -*-

########################################################################
#
# License: BSD
# Created: April 9, 2013
# Author:  Anthony Scopatz - scopatz@gmail.com
#
# $Id$
#
########################################################################

"""A module with no PyTables dependencies that helps with deprecation warnings.
"""
from inspect import getmembers, isclass
from warnings import warn

def previous_api(obj):
    """A decorator-like function for dealing with deprecations."""
    if isclass(obj):
        # punt if not a function or method
        return obj
    for key, value in getmembers(obj):
        if key == '__name__':
            newname = value
            break
    oldname = new2oldnames[newname]
    warnmsg = ("{0}() is pending deprecation, use {1}() instead.  "
               "You may use the pt2to3 tool to update your source code.")
    warnmsg = warnmsg.format(oldname, newname)
    def oldfunc(*args, **kwargs):
        warn(warnmsg, PendingDeprecationWarning)
        return obj(*args, **kwargs)
    oldfunc.__doc__ = (obj.__doc__ or '') + "\n\n.. warning::\n\n    " + warnmsg + "\n"
    return oldfunc

# old name, new name
old2newnames = dict([
    # from array.py
    ('getEnum', 'get_enum'),
    ('_initLoop', '_init_loop'), 
    ('_fancySelection', '_fancy_selection'),
    ('_checkShape', '_check_shape'),
    ('_readSlice', '_read_slice'),
    ('_readCoords', '_read_coords'),
    ('_readSelection', '_read_selection'),
    ('_writeSlice', '_write_slice'),
    ('_writeCoords', '_write_coords'),
    ('_writeSelection', '_write_selection'),
    ('_g_copyWithStats', '_g_copy_with_stats'),
    ('_c_classId', '_c_classid'),
    # from atom.py
    ('_checkBase', '_checkbase'),
    # from attributeset.py
    ('_g_updateNodeLocation', '_g_update_node_location'),
    ('_g_logAdd', '_g_log_add'),
    ('_g_delAndLog', '_g_del_and_log'),
    # from description.py
    ('_g_setNestedNamesDescr', '_g_set_nested_names_descr'),
    ('_g_setPathNames', '_g_set_path_names'),
    ('getColsInOrder', 'get_cols_in_order'),
    ('joinPaths', 'join_paths'),
    ('metaIsDescription', 'MetaIsDescription'),
    # from earray.py
    ('_checkShapeAppend', '_check_shape_append'),
    # from expression.py
    ('_exprvarsCache', '_exprvars_cache'),
    ('_requiredExprVars', '_required_expr_vars'),
    ('setInputsRange', 'set_inputs_range'),
    ('setOutput', 'set_output'),
    ('setOutputRange', 'set_output_range'),
    # from file.py
    ('_opToCode', '_op_to_code'),
    ('_codeToOp', '_code_to_op'),
    ('_transVersion', '_trans_version'),
    ('_transGroupParent', '_trans_group_parent'),
    ('_transGroupName', '_trans_group_name'),
    ('_transGroupPath', '_trans_group_path'),
    ('_actionLogParent', '_action_log_parent'),
    ('_actionLogName', '_action_log_name'),
    ('_actionLogPath', '_action_log_path'),
    ('_transParent', '_trans_parent'),
    ('_transName', '_trans_name'),
    ('_transPath', '_trans_path'),
    ('_shadowParent', '_shadow_parent'),
    ('_shadowName', '_shadow_name'),
    ('_shadowPath', '_shadow_path'),
    ('copyFile', 'copy_file'),
    ('openFile', 'open_file'),
    ('_getValueFromContainer', '_get_value_from_container'),
    ('__getRootGroup', '__get_root_group'),
    ('rootUEP', 'root_uep'),
    ('_getOrCreatePath', '_get_or_create_path'),
    ('_createPath', '_create_path'),
    ('createGroup', 'create_group'),
    ('createTable', 'create_table'),
    ('createArray', 'create_array'),
    ('createCArray', 'create_carray'),
    ('createEArray', 'create_earray'),
    ('createVLArray', 'create_vlarray'),
    ('createHardLink', 'create_hard_link'),
    ('createSoftLink', 'create_soft_link'),
    ('createExternalLink', 'create_external_link'),
    ('_getNode', '_get_node'),
    ('getNode', 'get_node'),
    ('isVisibleNode', 'is_visible_node'),
    ('renameNode', 'rename_node'),
    ('', ''),
    ('', ''),
    ('', ''),
    ('', ''),
    ('', ''),
    ('', ''),
    ('', ''),
    ('', ''),
    ('', ''),
    ('', ''),
    ('', ''),
    ('', ''),
    ('', ''),
    ('', ''),
    ('', ''),
    ('', ''),
    ('', ''),
    ('', ''),
    ('', ''),
    ('', ''),
    ('', ''),
    ('', ''),
    ('', ''),
    ('', ''),
    ])

new2oldnames = dict([(v, k) for k, v in old2newnames.iteritems()])

