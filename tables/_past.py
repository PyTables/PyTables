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
    warnmsg = ("{0}() is pending deprecation, use {1}() instead. "
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
    ('moveNode', 'move_node'),
    ('copyNode', 'copy_node'),
    ('removeNode', 'remove_node'),
    ('getNodeAttr', 'get_node_attr'),
    ('setNodeAttr', 'set_node_attr'),
    ('delNodeAttr', 'del_node_attr'),
    ('copyNodeAttrs', 'copy_node_attrs'),
    ('copyChildren', 'copy_children'),
    ('listNodes', 'list_nodes'),
    ('iterNodes', 'iter_nodes'),
    ('walkNodes', 'walk_nodes'),
    ('walkGroups', 'walk_groups'),
    ('_checkOpen', '_check_open'),
    ('_isWritable', '_iswritable'),
    ('_checkWritable', '_check_writable'),
    ('_checkGroup', '_check_group'),
    ('isUndoEnabled', 'is_undo_enabled'),
    ('_checkUndoEnabled', '_check_undo_enabled'),
    ('_createTransactionGroup', '_create_transaction_group'),
    ('_createTransaction', '_create_transaction'),
    ('_createMark', '_create_mark'),
    ('enableUndo', 'enable_undo'),
    ('disableUndo', 'disable_undo'),
    ('_getMarkID', '_get_mark_id'),
    ('_getFinalAction', '_get_final_action'),
    ('getCurrentMark', 'get_current_mark'),
    ('_refNode', '_refnode'),
    ('_unrefNode', '_unrefnode'),
    ('_killNode', '_killnode'),
    ('_reviveNode', '_revivenode'),
    ('_updateNodeLocations', '_update_node_locations'),
    # from group.py
    ('_getValueFromContainer', '_get_value_from_container'),
    ('_g_postInitHook', '_g_post_init_hook'),
    ('_g_getChildGroupClass', '_g_get_child_group_class'),
    ('_g_getChildLeafClass', '_g_get_child_leaf_class'),
    ('_g_addChildrenNames', '_g_add_children_names'),
    ('_g_checkHasChild', '_g_check_has_child'),
    ('_f_walkNodes', '_f_walknodes'),
    ('_g_widthWarning', '_g_width_warning'),
    ('_g_refNode', '_g_refnode'),
    ('_g_unrefNode', '_g_unrefnode'),
    ('_g_copyChildren', '_g_copy_children'),
    ('_f_getChild', '_f_get_child'),
    ('_f_listNodes', '_f_list_nodes'),
    ('_f_iterNodes', '_f_iter_nodes'),
    ('_f_walkGroups', '_f_walk_groups'),
    ('_g_closeDescendents', '_g_close_descendents'),
    ('_f_copyChildren', '_f_copy_children'),
    ('_v_maxGroupWidth', '_v_max_group_width'),
    ('_v_objectID', '_v_objectid'),
    ('_g_loadChild', '_g_load_child'),
    ('childName', 'childname'),
    ('_c_shadowNameRE', '_c_shadow_name_re'),
    # from hdf5extension.p{yx,xd}
    ('hdf5Extension', 'hdf5extension'),
    ('_getFileId', '_get_file_id'),
    ('_flushFile', '_flush_file'),
    ('_closeFile', '_close_file'),
    ('_g_listAttr', '_g_list_attr'),
    ('_g_setAttr', '_g_setattr'),
    ('_g_getAttr', '_g_getattr'),
    ('_g_listGroup', '_g_list_group'),
    ('_g_getGChildAttr', '_g_get_gchild_attr'),
    ('_g_getLChildAttr', '_g_get_lchild_attr'),
    ('_g_flushGroup', '_g_flush_group'),
    ('_g_closeGroup', '_g_close_group'),
    ('_g_moveNode', '_g_move_node'),
    ('_convertTime64', '_convert_time64'),
    ('_createArray', '_create_array'),
    ('_createCArray', '_create_carray'),
    ('_openArray', '_open_array'),
    ('_readArray', '_read_array'),
    ('_g_readSlice', '_g_read_slice'),
    ('_g_readCoords', '_g_read_coords'),
    ('_g_readSelection', '_g_read_selection'),
    ('_g_writeSlice', '_g_write_slice'),
    ('_g_writeCoords', '_g_write_coords'),
    ('_g_writeSelection', '_g_write_selection'),
    # from idxutils.py
    ('calcChunksize', 'calc_chunksize'),
    ('infinityF', 'infinityf'),
    ('infinityMap', 'infinitymap'),
    ('infType', 'inftype'),
    ('StringNextAfter', 'string_next_after'),
    ('IntTypeNextAfter', 'int_type_next_after'),
    ('BoolTypeNextAfter', 'bool_type_next_after'),
    # from index.py
    ('defaultAutoIndex', 'default_auto_index'),
    ('defaultIndexFilters', 'default_index_filters'),
    ('_tableColumnPathnameOfIndex', '_table_column_pathname_of_index'),
    ('_is_CSI', '_is_csi'),
    ('is_CSI', 'is_csi'),
    ('appendLastRow', 'append_last_row'),
    ('read_sliceLR', 'read_slice_lr'),
    ('readSorted', 'read_sorted'),
    ('readIndices', 'read_indices'),
    ('_processRange', '_process_range'),
    ('searchLastRow', 'search_last_row'),
    ('getLookupRange', 'get_lookup_range'),
    ('_g_checkName', '_g_check_name'),
    # from indexes.py
    ('_searchBin', '_search_bin'),
    # from indexesextension
    ('indexesExtension', 'indexesextension'),
    ('initRead', 'initread'),
    ('_readIndexSlice', '_read_index_slice'),
    ('_initSortedSlice', '_init_sorted_slice'),
    ('_g_readSortedSlice', '_g_read_sorted_slice'),
    ('_readSortedSlice', '_read_sorted_slice'),
    ('getLRUbounds', 'get_lru_bounds'),
    ('getLRUsorted', 'get_lru_sorted'),
    ('_searchBinNA_b', '_search_bin_na_b'),
    ('_searchBinNA_ub', '_search_bin_na_ub'),
    ('_searchBinNA_s', '_search_bin_na_s'),
    ('_searchBinNA_us', '_search_bin_na_us'),
    ('_searchBinNA_i', '_search_bin_na_i'),
    ('_searchBinNA_ui', '_search_bin_na_ui'),
    ('_searchBinNA_ll', '_search_bin_na_ll'),
    ('_searchBinNA_ull', '_search_bin_na_ull'),
    ('_searchBinNA_e', '_search_bin_na_e'),
    ('_searchBinNA_f', '_search_bin_na_f'),
    ('_searchBinNA_d', '_search_bin_na_d'),
    ('_searchBinNA_g', '_search_bin_na_g'),
    # from leaf.py
    ('_processRangeRead', '_process_range_read'),
    ('_pointSelection', '_point_selection'),
    ('isVisible', 'isvisible'),
    ('getAttr', 'get_attr'),
    ('setAttr', 'set_attr'),
    ('delAttr', 'del_attr'),
    # from link.py
    ('_g_getLinkClass', '_g_get_link_class'),
    # from linkextension
    ('linkExtension', 'linkextension'),
    ('_getLinkClass', '_get_link_class'),
    ('_g_createHardLink', '_g_create_hard_link'),
    ('', ''),
    ('', ''),
    ('', ''),
    ('', ''),
    ('', ''),
    ])

new2oldnames = dict([(v, k) for k, v in old2newnames.iteritems()])

