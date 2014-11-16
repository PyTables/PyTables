==================================
Migrating from PyTables 2.x to 3.x
==================================

:Author: Antonio Valentino
:Author: Anthony Scopatz

This document describes the major changes in PyTables in going from the
2.x to 3.x series and what you need to know when migrating downstream
code bases.

Python 3 at Last!
=================

The PyTables 3.x series now ships with full compatibility for Python 3.1+.
Additionally, we plan on maintaining compatibility with Python 2.7 for the
foreseeable future.  Python 2.6 is no longer under actively supported but
may work in most cases.  Note that the entire 3.x series now relies on
numexpr v2.1+, which itself is the first version of numexpr support both
Python 2 & 3.

Numeric, Numarray, NetCDF3, & HDF5 1.6 No More!
===============================================

PyTables no longer supports numeric and numarray. Please use numpy instead.
Additionally, the ``tables.netcdf3`` module has been removed. Please refer
to the `netcdf4-python`_ project for further support. Lastly, the older
HDF5 1.6 API is no longer supported.  Please upgrade to HDF5 1.8+.


Major API Changes
=================

The PyTables developers, `by popular demand`_, have taken this opportunity
that a major version number upgrade affords to implement significant API
changes.  We have tried to do this in such a way that will not immediately
break most existing code, though in some breakages may still occur.

PEP 8 Compliance
****************
The PyTables 3.x series now follows `PEP 8`_ coding standard.  This makes
using PyTables more idiomatic with surrounding Python code that also adheres
to this standard.  The primary way that the 2.x series was *not* PEP 8
compliant was with respect to variable naming conventions.  Approximately
:ref:`450 API variables <api-name-changes>` were identified and updated for
PyTables 3.x.

To ease migration, PyTables ships with a new ``pt2to3`` command line tool.
This tool will run over a file and replace any instances of the old variable
names with the 3.x version of the name.  This tool covers the overwhelming
majority of cases was used to transition the PyTables code base itself!  However,
it may also accidentally also pick up variable names in 3rd party codes that
have *exactly* the same name as a PyTables' variable.  This is because ``pt2to3``
was implemented using regular expressions rather than a fancier AST-based
method. By using regexes, ``pt2to3`` works on Python and Cython code.


``pt2to3`` **help:**

.. code-block:: bash

    usage: pt2to3 [-h] [-r] [-p] [-o OUTPUT] [-i] filename

    PyTables 2.x -> 3.x API transition tool This tool displays to standard out, so
    it is common to pipe this to another file: $ pt2to3 oldfile.py > newfile.py

    positional arguments:
      filename              path to input file.

    optional arguments:
      -h, --help            show this help message and exit
      -r, --reverse         reverts changes, going from 3.x -> 2.x.
      -p, --no-ignore-previous
                            ignores previous_api() calls.
      -o OUTPUT             output file to write to.
      -i, --inplace         overwrites the file in-place.

Note that ``pt2to3`` only works on a single file, not a a directory.  However,
a simple BASH script may be written to run ``pt2to3`` over an entire directory
and all sub-directories:

.. code-block:: bash

    #!/bin/bash
    for f in $(find .)
    do
        echo $f
        pt2to3 $f > temp.txt
        mv temp.txt $f
    done

.. note::

    :program:`pt2to3` uses the :mod:`argparse` module that is part of the
    Python standard library since Python 2.7.
    Users of Python 2.6 should install :mod:`argparse` separately
    (e.g. via :program:`pip`).

The old APIs and variable names will continue to be supported for the short term,
where possible.  (The major backwards incompatible changes come from the renaming
of some function and method arguments and keyword arguments.)  Using the 2.x APIs
in the 3.x series, however, will issue warnings.  The following is the release
plan for the warning types:

* 3.0 - PendingDeprecationWarning
* 3.1 - DeprecationWarning
* >=3.2 - Remove warnings, previous_api(), and _past.py; keep pt2to3,

The current plan is to maintain the old APIs for at least 2 years, though this
is subject to change.

Consistent ``create_xxx()`` Signatures
***************************************

Also by popular demand, it is now possible to create all data sets (``Array``,
``CArray``, ``EArray``, ``VLArray``, and ``Table``) from existing Python objects.
Constructors for these classes now accept either of the following keyword arguments:

* an ``obj`` to initialize with data
* or both ``atom`` and ``shape`` to initialize an empty structure, if possible.

These keyword arguments are also now part of the function signature for the
corresponding ``create_xxx()`` methods on the ``File`` class.  These would be called
as follows::

    # All create methods will support the following
    create_xxx(where, name, obj=obj)

    # All non-variable length arrays support the following:
    create_xxx(where, name, atom=atom, shape=shape)

Using ``obj`` or ``atom`` and ``shape`` are mutually exclusive. Previously only
``Array`` could be created with an existing Python object using the ``object``
keyword argument.


.. _api-name-changes:

API Name Changes
****************

The following tables shows the old 2.x names that have been update to their
new values in the new 3.x series.  Please use the ``pt2to3`` tool to convert
between these.

================================ ================================
**2.x Name**                     **3.x Name**
================================ ================================
AtomFromHDF5Type                 atom_from_hdf5_type
AtomToHDF5Type                   atom_to_hdf5_type
BoolTypeNextAfter                bool_type_next_after
HDF5ClassToString                hdf5_class_to_string
HDF5ToNPExtType                  hdf5_to_np_ext_type
HDF5ToNPNestedType               hdf5_to_np_nested_type
IObuf                            iobuf
IObufcpy                         iobufcpy
IntTypeNextAfter                 int_type_next_after
NPExtPrefixesToPTKinds           npext_prefixes_to_ptkinds
PTSpecialKinds                   pt_special_kinds
PTTypeToHDF5                     pttype_to_hdf5
StringNextAfter                  string_next_after
__allowedInitKwArgs              __allowed_init_kwargs
__getRootGroup                   __get_root_group
__next__inKernel                 __next__inkernel
_actionLogName                   _action_log_name
_actionLogParent                 _action_log_parent
_actionLogPath                   _action_log_path
_addRowsToIndex                  _add_rows_to_index
_appendZeros                     _append_zeros
_autoIndex                       _autoindex
_byteShape                       _byte_shape
_c_classId                       _c_classid
_c_shadowNameRE                  _c_shadow_name_re
_cacheDescriptionData            _cache_description_data
_checkAndSetPair                 _check_and_set_pair
_checkAttributes                 _check_attributes
_checkBase                       _checkbase
_checkColumn                     _check_column
_checkGroup                      _check_group
_checkNotClosed                  _check_not_closed
_checkOpen                       _check_open
_checkShape                      _check_shape
_checkShapeAppend                _check_shape_append
_checkUndoEnabled                _check_undo_enabled
_checkWritable                   _check_writable
_check_sortby_CSI                _check_sortby_csi
_closeFile                       _close_file
_codeToOp                        _code_to_op
_column__createIndex             _column__create_index
_compileCondition                _compile_condition
_conditionCache                  _condition_cache
_convertTime64                   _convert_time64
_convertTime64_                  _convert_time64_
_convertTypes                    _convert_types
_createArray                     _create_array
_createCArray                    _create_carray
_createMark                      _create_mark
_createPath                      _create_path
_createTable                     _create_table
_createTransaction               _create_transaction
_createTransactionGroup          _create_transaction_group
_disableIndexingInQueries        _disable_indexing_in_queries
_doReIndex                       _do_reindex
_emptyArrayCache                 _empty_array_cache
_enableIndexingInQueries         _enable_indexing_in_queries
_enabledIndexingInQueries        _enabled_indexing_in_queries
_exprvarsCache                   _exprvars_cache
_f_copyChildren                  _f_copy_children
_f_delAttr                       _f_delattr
_f_getAttr                       _f_getattr
_f_getChild                      _f_get_child
_f_isVisible                     _f_isvisible
_f_iterNodes                     _f_iter_nodes
_f_listNodes                     _f_list_nodes
_f_setAttr                       _f_setattr
_f_walkGroups                    _f_walk_groups
_f_walkNodes                     _f_walknodes
_fancySelection                  _fancy_selection
_fillCol                         _fill_col
_flushBufferedRows               _flush_buffered_rows
_flushFile                       _flush_file
_flushModRows                    _flush_mod_rows
_g_addChildrenNames              _g_add_children_names
_g_checkGroup                    _g_check_group
_g_checkHasChild                 _g_check_has_child
_g_checkName                     _g_check_name
_g_checkNotContains              _g_check_not_contains
_g_checkOpen                     _g_check_open
_g_closeDescendents              _g_close_descendents
_g_closeGroup                    _g_close_group
_g_copyAsChild                   _g_copy_as_child
_g_copyChildren                  _g_copy_children
_g_copyRows                      _g_copy_rows
_g_copyRows_optim                _g_copy_rows_optim
_g_copyWithStats                 _g_copy_with_stats
_g_createHardLink                _g_create_hard_link
_g_delAndLog                     _g_del_and_log
_g_delLocation                   _g_del_location
_g_flushGroup                    _g_flush_group
_g_getAttr                       _g_getattr
_g_getChildGroupClass            _g_get_child_group_class
_g_getChildLeafClass             _g_get_child_leaf_class
_g_getGChildAttr                 _g_get_gchild_attr
_g_getLChildAttr                 _g_get_lchild_attr
_g_getLinkClass                  _g_get_link_class
_g_listAttr                      _g_list_attr
_g_listGroup                     _g_list_group
_g_loadChild                     _g_load_child
_g_logAdd                        _g_log_add
_g_logCreate                     _g_log_create
_g_logMove                       _g_log_move
_g_maybeRemove                   _g_maybe_remove
_g_moveNode                      _g_move_node
_g_postInitHook                  _g_post_init_hook
_g_postReviveHook                _g_post_revive_hook
_g_preKillHook                   _g_pre_kill_hook
_g_propIndexes                   _g_prop_indexes
_g_readCoords                    _g_read_coords
_g_readSelection                 _g_read_selection
_g_readSlice                     _g_read_slice
_g_readSortedSlice               _g_read_sorted_slice
_g_refNode                       _g_refnode
_g_removeAndLog                  _g_remove_and_log
_g_setAttr                       _g_setattr
_g_setLocation                   _g_set_location
_g_setNestedNamesDescr           _g_set_nested_names_descr
_g_setPathNames                  _g_set_path_names
_g_unrefNode                     _g_unrefnode
_g_updateDependent               _g_update_dependent
_g_updateLocation                _g_update_location
_g_updateNodeLocation            _g_update_node_location
_g_updateTableLocation           _g_update_table_location
_g_widthWarning                  _g_width_warning
_g_writeCoords                   _g_write_coords
_g_writeSelection                _g_write_selection
_g_writeSlice                    _g_write_slice
_getColumnInstance               _get_column_instance
_getConditionKey                 _get_condition_key
_getContainer                    _get_container
_getEnumMap                      _get_enum_map
_getFileId                       _get_file_id
_getFinalAction                  _get_final_action
_getInfo                         _get_info
_getLinkClass                    _get_link_class
_getMarkID                       _get_mark_id
_getNode                         _get_node
_getOrCreatePath                 _get_or_create_path
_getTypeColNames                 _get_type_col_names
_getUnsavedNrows                 _get_unsaved_nrows
_getValueFromContainer           _get_value_from_container
_hiddenNameRE                    _hidden_name_re
_hiddenPathRE                    _hidden_path_re
_indexNameOf                     _index_name_of
_indexNameOf_                    _index_name_of_
_indexPathnameOf                 _index_pathname_of
_indexPathnameOfColumn           _index_pathname_of_column
_indexPathnameOfColumn_          _index_pathname_of_column_
_indexPathnameOf_                _index_pathname_of_
_initLoop                        _init_loop
_initSortedSlice                 _init_sorted_slice
_isWritable                      _iswritable
_is_CSI                          _is_csi
_killNode                        _killnode
_lineChunkSize                   _line_chunksize
_lineSeparator                   _line_separator
_markColumnsAsDirty              _mark_columns_as_dirty
_newBuffer                       _new_buffer
_notReadableError                _not_readable_error
_npSizeType                      _npsizetype
_nxTypeFromNPType                _nxtype_from_nptype
_opToCode                        _op_to_code
_openArray                       _open_array
_openUnImplemented               _open_unimplemented
_pointSelection                  _point_selection
_processRange                    _process_range
_processRangeRead                _process_range_read
_pythonIdRE                      _python_id_re
_reIndex                         _reindex
_readArray                       _read_array
_readCoordinates                 _read_coordinates
_readCoords                      _read_coords
_readIndexSlice                  _read_index_slice
_readSelection                   _read_selection
_readSlice                       _read_slice
_readSortedSlice                 _read_sorted_slice
_refNode                         _refnode
_requiredExprVars                _required_expr_vars
_reservedIdRE                    _reserved_id_re
_reviveNode                      _revivenode
_saveBufferedRows                _save_buffered_rows
_searchBin                       _search_bin
_searchBinNA_b                   _search_bin_na_b
_searchBinNA_d                   _search_bin_na_d
_searchBinNA_e                   _search_bin_na_e
_searchBinNA_f                   _search_bin_na_f
_searchBinNA_g                   _search_bin_na_g
_searchBinNA_i                   _search_bin_na_i
_searchBinNA_ll                  _search_bin_na_ll
_searchBinNA_s                   _search_bin_na_s
_searchBinNA_ub                  _search_bin_na_ub
_searchBinNA_ui                  _search_bin_na_ui
_searchBinNA_ull                 _search_bin_na_ull
_searchBinNA_us                  _search_bin_na_us
_setAttributes                   _set_attributes
_setColumnIndexing               _set_column_indexing
_shadowName                      _shadow_name
_shadowParent                    _shadow_parent
_shadowPath                      _shadow_path
_sizeToShape                     _size_to_shape
_tableColumnPathnameOfIndex      _table_column_pathname_of_index
_tableFile                       _table_file
_tablePath                       _table_path
_table__autoIndex                _table__autoindex
_table__getautoIndex             _table__getautoindex
_table__setautoIndex             _table__setautoindex
_table__whereIndexed             _table__where_indexed
_transGroupName                  _trans_group_name
_transGroupParent                _trans_group_parent
_transGroupPath                  _trans_group_path
_transName                       _trans_name
_transParent                     _trans_parent
_transPath                       _trans_path
_transVersion                    _trans_version
_unrefNode                       _unrefnode
_updateNodeLocations             _update_node_locations
_useIndex                        _use_index
_vShape                          _vshape
_vType                           _vtype
_v__nodeFile                     _v__nodefile
_v__nodePath                     _v__nodepath
_v_colObjects                    _v_colobjects
_v_maxGroupWidth                 _v_max_group_width
_v_maxTreeDepth                  _v_maxtreedepth
_v_nestedDescr                   _v_nested_descr
_v_nestedFormats                 _v_nested_formats
_v_nestedNames                   _v_nested_names
_v_objectID                      _v_objectid
_whereCondition                  _where_condition
_writeCoords                     _write_coords
_writeSelection                  _write_selection
_writeSlice                      _write_slice
appendLastRow                    append_last_row
attrFromShadow                   attr_from_shadow
attrToShadow                     attr_to_shadow
autoIndex                        autoindex
bufcoordsData                    bufcoords_data
calcChunksize                    calc_chunksize
checkFileAccess                  check_file_access
checkNameValidity                check_name_validity
childName                        childname
chunkmapData                     chunkmap_data
classIdDict                      class_id_dict
className                        classname
classNameDict                    class_name_dict
containerRef                     containerref
convertToNPAtom                  convert_to_np_atom
convertToNPAtom2                 convert_to_np_atom2
copyChildren                     copy_children
copyClass                        copyclass
copyFile                         copy_file
copyLeaf                         copy_leaf
copyNode                         copy_node
copyNodeAttrs                    copy_node_attrs
countLoggedInstances             count_logged_instances
createArray                      create_array
createCArray                     create_carray
createCSIndex                    create_csindex
createEArray                     create_earray
createExternalLink               create_external_link
createGroup                      create_group
createHardLink                   create_hard_link
createIndex                      create_index
createIndexesDescr               create_indexes_descr
createIndexesTable               create_indexes_table
createNestedType                 create_nested_type
createSoftLink                   create_soft_link
createTable                      create_table
createVLArray                    create_vlarray
defaultAutoIndex                 default_auto_index
defaultIndexFilters              default_index_filters
delAttr                          del_attr
delAttrs                         _del_attrs
delNodeAttr                      del_node_attr
detectNumberOfCores              detect_number_of_cores
disableUndo                      disable_undo
dumpGroup                        dump_group
dumpLeaf                         dump_leaf
dumpLoggedInstances              dump_logged_instances
enableUndo                       enable_undo
enumFromHDF5                     enum_from_hdf5
enumToHDF5                       enum_to_hdf5
fetchLoggedInstances             fetch_logged_instances
flushRowsToIndex                 flush_rows_to_index
getAttr                          get_attr
getAttrs                         _get_attrs
getClassByName                   get_class_by_name
getColsInOrder                   get_cols_in_order
getCurrentMark                   get_current_mark
getEnum                          get_enum
getFilters                       get_filters
getHDF5Version                   get_hdf5_version
getIndices                       get_indices
getLRUbounds                     get_lru_bounds
getLRUsorted                     get_lru_sorted
getLookupRange                   get_lookup_range
getNestedField                   get_nested_field
getNestedFieldCache              get_nested_field_cache
getNestedType                    get_nested_type
getNode                          get_node
getNodeAttr                      get_node_attr
getPyTablesVersion               get_pytables_version
getTypeEnum                      get_type_enum
getWhereList                     get_where_list
hdf5Extension                    hdf5extension
hdf5Version                      hdf5_version
indexChunk                       indexchunk
indexValid                       indexvalid
indexValidData                   index_valid_data
indexValues                      indexvalues
indexValuesData                  index_values_data
indexesExtension                 indexesextension
infType                          inftype
infinityF                        infinityf
infinityMap                      infinitymap
initRead                         initread
isHDF5File                       is_hdf5_file
isPyTablesFile                   is_pytables_file
isUndoEnabled                    is_undo_enabled
isVisible                        isvisible
isVisibleName                    isvisiblename
isVisibleNode                    is_visible_node
isVisiblePath                    isvisiblepath
is_CSI                           is_csi
iterNodes                        iter_nodes
iterseqMaxElements               iterseq_max_elements
joinPath                         join_path
joinPaths                        join_paths
linkExtension                    linkextension
listLoggedInstances              list_logged_instances
listNodes                        list_nodes
loadEnum                         load_enum
logInstanceCreation              log_instance_creation
lrucacheExtension                lrucacheextension
metaIsDescription                MetaIsDescription
modifyColumn                     modify_column
modifyColumns                    modify_columns
modifyCoordinates                modify_coordinates
modifyRows                       modify_rows
moveFromShadow                   move_from_shadow
moveNode                         move_node
moveToShadow                     move_to_shadow
newNode                          new_node
newSet                           newset
newdstGroup                      newdst_group
objectID                         object_id
oldPathname                      oldpathname
openFile                         open_file
openNode                         open_node
parentNode                       parentnode
parentPath                       parentpath
reIndex                          reindex
reIndexDirty                     reindex_dirty
readCoordinates                  read_coordinates
readIndices                      read_indices
readSlice                        read_slice
readSorted                       read_sorted
readWhere                        read_where
read_sliceLR                     read_slice_lr
recreateIndexes                  recreate_indexes
redoAddAttr                      redo_add_attr
redoCreate                       redo_create
redoDelAttr                      redo_del_attr
redoMove                         redo_move
redoRemove                       redo_remove
removeIndex                      remove_index
removeNode                       remove_node
removeRows                       remove_rows
renameNode                       rename_node
rootUEP                          root_uep
searchLastRow                    search_last_row
setAttr                          set_attr
setAttrs                         _set_attrs
setBloscMaxThreads               set_blosc_max_threads
setInputsRange                   set_inputs_range
setNodeAttr                      set_node_attr
setOutput                        set_output
setOutputRange                   set_output_range
silenceHDF5Messages              silence_hdf5_messages
splitPath                        split_path
tableExtension                   tableextension
undoAddAttr                      undo_add_attr
undoCreate                       undo_create
undoDelAttr                      undo_del_attr
undoMove                         undo_move
undoRemove                       undo_remove
utilsExtension                   utilsextension
walkGroups                       walk_groups
walkNodes                        walk_nodes
whereAppend                      append_where
whereCond                        wherecond
whichClass                       which_class
whichLibVersion                  which_lib_version
willQueryUseIndexing             will_query_use_indexing
================================ ================================

----

  **Enjoy data!**

  -- The PyTables Developers


.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 78
.. End:


.. _by popular demand: http://sourceforge.net/mailarchive/message.php?msg_id=29584752

.. _PEP 8: http://www.python.org/dev/peps/pep-0008/

.. _netcdf4-python: http://code.google.com/p/netcdf4-python/
