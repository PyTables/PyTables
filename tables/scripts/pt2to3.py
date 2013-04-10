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

"""This utility helps you migrate from PyTables 2.x APIs to 3.x APIs, which
are PEP 8 compliant.

"""
import os
import re
import sys
import argparse

# Note that it is tempting to use the ast module here, but then this 
# breaks transforming cython files.  So instead we are going to do the 
# dumb thing with replace.

# old name, new name
names = [
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
    ]

def make_subs(ns):
    global names
    if ns.reverse:
        names = [(v, k) for k, v in names]
    s = '({0})'.format('|'.join([k for k, v in names]))
    if ns.ignore_previous:
        s += '(?!\s*?=\s*?previous_api\()'
        s += """(?!['"]{1,3}\s*?\))"""
    subs = re.compile(s, flags=re.MULTILINE)
    namesmap = dict(names)
    def repl(m):
        return namesmap.get(m.group(1), m.group(0))
    return subs, repl


def main():
    parser = argparse.ArgumentParser(description='PyTables 2.x -> 3.x API transition tool')
    parser.add_argument('-r', '--reverse', action='store_true', default=False, 
                        dest='reverse', help="reverts changes, going from 3.x -> 2.x.")
    parser.add_argument('-p', '--no-ignore-previous', action='store_false', 
                default=True, dest='ignore_previous', help="i previous_api() calls.")
    parser.add_argument('filename', help='path to input file.')
    ns = parser.parse_args()

    if not os.path.isfile(ns.filename):
        sys.exit('file {0!r} not found'.format(ns.filename))
    with open(ns.filename, 'r') as f:
        src = f.read()

    subs, repl = make_subs(ns)
    targ = subs.sub(repl, src)
    print targ

if __name__ == '__main__':
    main()
