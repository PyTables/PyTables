"""
Classes for describing columns for ``Table`` objects.

:Author: Francesc Alted
:Contact: faltet@pytables.com
:License: BSD
:Created: September 21, 2002
:Revision: $Id$

Variables
=========

`__docformat`__
    The format of documentation strings in this module.
`__version__`
    Repository version of this file.
"""

# Imports
# =======
import warnings
import sys
import copy

import numpy

from tables import atom
from tables.path import checkNameValidity


# Public variables
# ================
__docformat__ = 'reStructuredText'
"""The format of documentation strings in this module."""

__version__ = '$Revision$'
"""Repository version of this file."""


# Private functions
# =================
def same_position(oldmethod):
    """Decorate `oldmethod` to also compare the `_v_pos` attribute."""
    def newmethod(self, other):
        try:
            other_pos = other._v_pos
        except AttributeError:
            return False  # not a column definition
        return self._v_pos == other._v_pos and oldmethod(self, other)
    newmethod.__name__ = oldmethod.__name__
    newmethod.__doc__ = oldmethod.__doc__
    return newmethod


# Column classes
# ==============
class Col(atom.Atom):
    """
    Defines a non-nested column.

    `Col` instances are used as a means to declare the different
    properties of a non-nested column in a table or nested column.
    `Col` classes are descendants of their equivalent `Atom` classes,
    but their instances have an additional ``_v_pos`` attribute that is
    used to decide the position of the column inside its parent table or
    nested column (see the `IsDescription` class for more information on
    column positions).

    In the same fashion as `Atom`, you should use a particular `Col`
    descendant class whenever you know the exact type you will need when
    writing your code.  Otherwise, you may use one of the
    ``Col.from_*()`` factory methods.

    Public instance variables
    -------------------------

    In addition to the variables that they inherit from the `Atom`
    class, `Col` instances have the following attributes:

    _v_pos
        The *relative* position of this column with regard to its column
        siblings.

    Factory methods
    ---------------

    Each factory method inherited from the `Atom` class is available
    with the same signature, plus an additional `pos` parameter (placed
    in last position) which defaults to ``None`` and that may take an
    integer value.  This parameter might be used to specify the position
    of the column in the table.

    Besides, there are the next additional factory methods, available
    only for `Col` objects:</para>

    from_atom(atom[, pos])
        Create a `Col` definition from a PyTables ``atom``.

    Constructors
    ------------

    There are some common arguments for most `Col` -derived
    constructors:

    itemsize
        For types with a non-fixed size, this sets the size in bytes
        of individual items in the column.

    shape
        Sets the shape of the column.  An integer shape of ``N`` is
        equivalent to the tuple ``(N,)``.

    dflt
        Sets the default value for the column.

    pos
        Sets the position of column in table.  If unspecified, the
        position will be randomly selected.

    """

    # Avoid mangling atom class data.
    __metaclass__ = type

    _class_from_prefix = {}  # filled as column classes are created
    """Maps column prefixes to column classes."""

    # Class methods
    # ~~~~~~~~~~~~~
    @classmethod
    def prefix(class_):
        """Return the column class prefix."""
        cname = class_.__name__
        return cname[:cname.rfind('Col')]

    @classmethod
    def from_atom(class_, atom, pos=None):
        """
        Create a `Col` definition from a PyTables `atom`.

        An optional position may be specified as the `pos` argument.
        """
        prefix = atom.prefix()
        kwargs = atom._get_init_args()
        colclass = class_._class_from_prefix[prefix]
        return colclass(pos=pos, **kwargs)

    @classmethod
    def from_sctype(class_, sctype, shape=(), dflt=None, pos=None):
        """
        Create a `Col` definition from a NumPy scalar type `sctype`.

        Optional shape, default value and position may be specified as
        the `shape`, `dflt` and `pos` arguments, respectively.
        Information in the `sctype` not represented in a `Col` is
        ignored.
        """
        newatom = atom.Atom.from_sctype(sctype, shape, dflt)
        return class_.from_atom(newatom, pos=pos)

    @classmethod
    def from_dtype(class_, dtype, dflt=None, pos=None):
        """
        Create a `Col` definition from a NumPy `dtype`.

        Optional default value and position may be specified as the
        `dflt` and `pos` arguments, respectively.  The `dtype` must have
        a byte order which is irrelevant or compatible with that of the
        system.  Information in the `dtype` not represented in a `Col`
        is ignored.
        """
        newatom = atom.Atom.from_dtype(dtype, dflt)
        return class_.from_atom(newatom, pos=pos)

    @classmethod
    def from_type(class_, type, shape=(), dflt=None, pos=None):
        """
        Create a `Col` definition from a PyTables `type`.

        Optional shape, default value and position may be specified as
        the `shape`, `dflt` and `pos` arguments, respectively.
        """
        newatom = atom.Atom.from_type(type, shape, dflt)
        return class_.from_atom(newatom, pos=pos)

    @classmethod
    def from_kind(class_, kind, itemsize=None, shape=(), dflt=None, pos=None):
        """
        Create a `Col` definition from a PyTables `kind`.

        Optional item size, shape, default value and position may be
        specified as the `itemsize`, `shape`, `dflt` and `pos`
        arguments, respectively.  Bear in mind that not all columns
        support a default item size.
        """
        newatom = atom.Atom.from_kind(kind, itemsize, shape, dflt)
        return class_.from_atom(newatom, pos=pos)

    @classmethod
    def _subclass_from_prefix(class_, prefix):
        """Get a column subclass for the given `prefix`."""

        cname = '%sCol' % prefix
        class_from_prefix = class_._class_from_prefix
        if cname in class_from_prefix:
            return class_from_prefix[cname]
        atombase = getattr(atom, '%sAtom' % prefix)

        class NewCol(class_, atombase):
            """
            Defines a non-nested column of a particular type.

            The constructor accepts the same arguments as the equivalent
            `Atom` class, plus an additional ``pos`` argument for
            position information, which is assigned to the `_v_pos`
            attribute.
            """
            def __init__(self, *args, **kwargs):
                pos = kwargs.pop('pos', None)
                class_from_prefix = self._class_from_prefix
                atombase.__init__(self, *args, **kwargs)
                # The constructor of an abstract atom may have changed
                # the class of `self` to something different of `NewCol`
                # and `atombase` (that's why the prefix map is saved).
                if self.__class__ is not NewCol:
                    colclass = class_from_prefix[self.prefix()]
                    self.__class__ = colclass
                self._v_pos = pos

            __eq__ = same_position(atombase.__eq__)
            _is_equal_to_atom = same_position(atombase._is_equal_to_atom)

            if prefix == 'Enum':
                _is_equal_to_enumatom = same_position(
                    atombase._is_equal_to_enumatom )

        NewCol.__name__ = cname

        class_from_prefix[prefix] = NewCol
        return NewCol

    # Special methods
    # ~~~~~~~~~~~~~~~
    def __repr__(self):
        # Reuse the atom representation.
        atomrepr = super(Col, self).__repr__()
        lpar = atomrepr.index('(')
        rpar = atomrepr.rindex(')')
        atomargs = atomrepr[lpar + 1:rpar]
        classname = self.__class__.__name__
        return '%s(%s, pos=%s)' % (classname, atomargs, self._v_pos)

def _generate_col_classes():
    """Generate all column classes."""
    # Abstract classes are not in the class map.
    cprefixes = ['Int', 'UInt', 'Float', 'Time']
    for (kind, kdata) in atom.atom_map.items():
        if hasattr(kdata, 'kind'):  # atom class: non-fixed item size
            atomclass = kdata
            cprefixes.append(atomclass.prefix())
        else:  # dictionary: fixed item size
            for atomclass in kdata.values():
                cprefixes.append(atomclass.prefix())

    # Bottom-level complex classes are not in the type map, of course.
    # We still want the user to get the compatibility warning, though.
    cprefixes.extend(['Complex32', 'Complex64', 'Complex128'])

    for cprefix in cprefixes:
        newclass = Col._subclass_from_prefix(cprefix)
        yield newclass

# Create all column classes.
for _newclass in _generate_col_classes():
    exec '%s = _newclass' % _newclass.__name__
del _newclass


# Table description classes
# =========================
class Description(object):
    """
    This class represents descriptions of the structure of tables.

    An instance of this class is automatically bound to `Table` objects
    when they are created.  It provides a browseable representation of
    the structure of the table, made of non-nested (`Col`) and nested
    (`Description`) columns.  It also contains information that will
    allow you to build ``NestedRecArray`` objects suited for the
    different columns in a table (be they nested or not).

    Column definitions under a description can be accessed as attributes
    of it (*natural naming*).  For instance, if ``table.description`` is
    a ``Description`` instance with a colum named ``col1`` under it, the
    later can be accessed as ``table.description.col1``.  If ``col1`` is
    nested and contains a ``col2`` column, this can be accessed as
    ``table.description.col1.col2``.  Because of natural naming, the
    names of members start with special prefixes, like in the `Group`
    class.

    Public instance variables
    -------------------------

    _v_colObjects
        A dictionary mapping the names of the columns hanging directly
        from the associated table or nested column to their respective
        descriptions (`Col` or `Description` instances).

    _v_dflts
        A dictionary mapping the names of non-nested columns hanging
        directly from the associated table or nested column to their
        respective default values.

    _v_dtype
        The NumPy type which reflects the structure of this table or
        nested column.  You can use this as the ``dtype`` argument of
        NumPy array factories.

    _v_dtypes
        A dictionary mapping the names of non-nested columns hanging
        directly from the associated table or nested column to their
        respective NumPy types.

    _v_is_nested
        Whether the associated table or nested column contains further
        nested columns or not.

    _v_itemsize
        The size in bytes of an item in this table or nested column.

    _v_name
        The name of this description group.  The name of the root group
        is ``'/'``.

    _v_names
        A list of the names of the columns hanging directly from the
        associated table or nested column.  The order of the names
        matches the order of their respective columns in the containing
        table.

    _v_nestedDescr
        A nested list of pairs of ``(name, format)`` tuples for all the
        columns under this table or nested column.  You can use this as
        the ``dtype`` and ``descr`` arguments of NumPy array and
        `NestedRecArray` factories, respectively.

    _v_nestedFormats
        A nested list of the NumPy string formats (and shapes) of all
        the columns under this table or nested column.  You can use this
        as the ``formats`` argument of NumPy array and `NestedRecArray`
        factories.

    _v_nestedlvl
        The level of the associated table or nested column in the nested
        datatype.

    _v_nestedNames
        A nested list of the names of all the columns under this table
        or nested column.  You can use this for the ``names`` argument
        of `NestedRecArray` factory functions.

    _v_pathnames
        A list of the pathnames of all the columns under this table or
        nested column (in preorder).  If it does not contain nested
        columns, this is exactly the same as the `Description._v_names`
        attribute.

    _v_types
        A dictionary mapping the names of non-nested columns hanging
        directly from the associated table or nested column to their
        respective PyTables types.

    Public methods
    --------------

    _f_walk([type])
        Iterate over nested columns.
    """

    def __init__(self, classdict, nestedlvl=-1, validate=True):

        # Do a shallow copy of classdict just in case this is going to
        # be shared by other instances
        newdict = self.__dict__
        newdict["_v_name"] = "/"   # The name for root descriptor
        newdict["_v_names"] = []
        newdict["_v_dtypes"] = {}
        newdict["_v_types"] = {}
        newdict["_v_dflts"] = {}
        newdict["_v_colObjects"] = {}
        newdict["_v_is_nested"] = False
        nestedFormats = []
        nestedDType = []

        if not hasattr(newdict, "_v_nestedlvl"):
            newdict["_v_nestedlvl"] = nestedlvl + 1

        cols_with_pos = []  # colum (position, name) pairs
        cols_no_pos = []  # just column names

        # Check for special variables and convert column descriptions
        for (name, descr) in classdict.items():
            if name.startswith('_v_'):
                if name in newdict:
                    #print "Warning!"
                    # special methods &c: copy to newdict, warn about conflicts
                    warnings.warn("Can't set attr %r in description class %r" \
                                  % (name, self))
                else:
                    #print "Special variable!-->", name, classdict[name]
                    newdict[name] = descr
                continue  # This variable is not needed anymore

            columns = None
            if (type(descr) == type(IsDescription) and
                issubclass(descr, IsDescription)):
                #print "Nested object (type I)-->", name
                columns = descr().columns
            elif (type(descr.__class__) == type(IsDescription) and
                issubclass(descr.__class__, IsDescription)):
                #print "Nested object (type II)-->", name
                columns = descr.columns
            elif isinstance(descr, dict):
                #print "Nested object (type III)-->", name
                columns = descr
            else:
                #print "Nested object (type IV)-->", name
                descr = copy.copy(descr)
            # The copies above and below ensure that the structures
            # provided by the user will remain unchanged even if we
            # tamper with the values of ``_v_pos`` here.
            if columns is not None:
                descr = Description(copy.copy(columns), self._v_nestedlvl)
            classdict[name] = descr

            pos = getattr(descr, '_v_pos', None)
            if pos is None:
                cols_no_pos.append(name)
            else:
                cols_with_pos.append((pos, name))

        # Sort field names:
        #
        # 1. Fields with explicit positions, according to their
        #    positions (and their names if coincident).
        # 2. Fields with no position, in alfabetical order.
        cols_with_pos.sort()
        cols_no_pos.sort()
        keys = [name for (pos, name) in cols_with_pos] + cols_no_pos

        pos = 0
        # Get properties for compound types
        for k in keys:
            if validate:
                # Check for key name validity
                checkNameValidity(k)
            # Class variables
            object = classdict[k]
            newdict[k] = object    # To allow natural naming
            if not (isinstance(object, Col) or
                    isinstance(object, Description)):
                raise TypeError, \
"""Passing an incorrect value to a table column. Expected a Col (or
  subclass) instance and got: "%s". Please make use of the Col(), or
  descendant, constructor to properly initialize columns.
""" % object
            object._v_pos = pos  # Set the position of this object
            object._v_parent = self  # The parent description
            pos += 1
            newdict['_v_colObjects'][k] = object
            newdict['_v_names'].append(k)
            object.__dict__['_v_name'] = k
            if isinstance(object, Col):
                dtype = object.dtype
                newdict['_v_dtypes'][k] = dtype
                newdict['_v_types'][k] = object.type
                newdict['_v_dflts'][k] = object.dflt
                nestedFormats.append(object.recarrtype)
                baserecarrtype = dtype.base.str[1:]
                nestedDType.append((k, baserecarrtype, dtype.shape))
            else:  # A description
                nestedFormats.append(object._v_nestedFormats)
                nestedDType.append((k, object._v_dtype))

        # Assign the format list to _v_nestedFormats
        newdict['_v_nestedFormats'] = nestedFormats
        newdict['_v_dtype'] = numpy.dtype(nestedDType)
        # _v_itemsize is derived from the _v_dtype that already computes this
        newdict['_v_itemsize'] = newdict['_v_dtype'].itemsize
        if self._v_nestedlvl == 0:
            # Get recursively nested _v_nestedNames and _v_nestedDescr attrs
            self._g_setNestedNamesDescr()
            # Get pathnames for nested groups
            self._g_setPathNames()
            # Check the _v_byteorder has been used an issue an Error
            if hasattr(self, "_v_byteorder"):
                raise ValueError(
                    "Using a ``_v_byteorder`` in the description is obsolete. "
                    "Use the byteorder parameter in the constructor instead.")


    def _g_setNestedNamesDescr(self):
        """Computes the nested names and descriptions for nested datatypes.
        """
        names = self._v_names
        fmts = self._v_nestedFormats
        self._v_nestedNames = names[:]  # Important to do a copy!
        self._v_nestedDescr = [(names[i], fmts[i]) for i in range(len(names))]
        for i in range(len(names)):
            name = names[i]
            new_object = self._v_colObjects[name]
            if isinstance(new_object, Description):
                new_object._g_setNestedNamesDescr()
                # replace the column nested name by a correct tuple
                self._v_nestedNames[i] = (name, new_object._v_nestedNames)
                self._v_nestedDescr[i] = (name, new_object._v_nestedDescr)
                # set the _v_is_nested flag
                self._v_is_nested = True


    def _g_setPathNames(self):
        """Compute the pathnames for arbitrary nested descriptions.

        This method sets the ``_v_pathname`` and ``_v_pathnames``
        attributes of all the elements (both descriptions and columns)
        in this nested description.
        """

        def getColsInOrder(description):
            return [description._v_colObjects[colname]
                    for colname in description._v_names]

        def joinPaths(path1, path2):
            if not path1:
                return path2
            return '%s/%s' % (path1, path2)

        # The top of the stack always has a nested description
        # and a list of its child columns
        # (be they nested ``Description`` or non-nested ``Col`` objects).
        # In the end, the list contains only a list of column paths
        # under this one.
        #
        # For instance, given this top of the stack::
        #
        #   (<Description X>, [<Column A>, <Column B>])
        #
        # After computing the rest of the stack, the top is::
        #
        #   (<Description X>, ['a', 'a/m', 'a/n', ... , 'b', ...])

        stack = []

        # We start by pushing the top-level description
        # and its child columns.
        self._v_pathname = ''
        stack.append((self, getColsInOrder(self)))

        while stack:
            desc, cols = stack.pop()
            head = cols[0]

            # What's the first child in the list?
            if isinstance(head, Description):
                # A nested description.  We remove it from the list and
                # push it with its child columns.  This will be the next
                # handled description.
                head._v_pathname = joinPaths(desc._v_pathname, head._v_name)
                stack.append((desc, cols[1:]))  # alter the top
                stack.append((head, getColsInOrder(head)))  # new top
            elif isinstance(head, Col):
                # A non-nested column.  We simply remove it from the
                # list and append its name to it.
                head._v_pathname = joinPaths(desc._v_pathname, head._v_name)
                cols.append(head._v_name)  # alter the top
                stack.append((desc, cols[1:]))  # alter the top
            else:
                # Since paths and names are appended *to the end* of
                # children lists, a string signals that no more children
                # remain to be processed, so we are done with the
                # description at the top of the stack.
                assert isinstance(head, basestring)
                # Assign the computed set of descendent column paths.
                desc._v_pathnames = cols
                if len(stack) > 0:
                    # Compute the paths with respect to the parent node
                    # (including the path of the current description)
                    # and append them to its list.
                    descName = desc._v_name
                    colPaths = [joinPaths(descName, path) for path in cols]
                    colPaths.insert(0, descName)
                    parentCols = stack[-1][1]
                    parentCols.extend(colPaths)
                # (Nothing is pushed, we are done with this description.)


    def _f_walk(self, type='All'):
        """
        Iterate over nested columns.

        If `type` is ``'All'`` (the default), all column description
        objects (`Col` and `Description` instances) are yielded in
        top-to-bottom order (preorder).

        If `type` is ``'Col'`` or ``'Description'``, only column or
        descriptions of the specified type are yielded.
        """

        if type not in ["All", "Col", "Description"]:
            raise ValueError("""\
type can only take the parameters 'All', 'Col' or 'Description'.""")

        stack = [self]
        while stack:
            object = stack.pop(0)  # pop at the front so as to ensure the order
            if type in ["All", "Description"]:
                yield object  # yield description
            names = object._v_names
            for i in range(len(names)):
                new_object = object._v_colObjects[names[i]]
                if isinstance(new_object, Description):
                    stack.append(new_object)
                else:
                    if type in ["All", "Col"]:
                        yield new_object  # yield column


    def __repr__(self):
        """ Gives a detailed Description column representation.
        """
        rep = [ '%s\"%s\": %r' %  \
                ("  "*self._v_nestedlvl, k, self._v_colObjects[k])
                for k in self._v_names]
        return '{\n  %s}' % (',\n  '.join(rep))


    def __str__(self):
        """ Gives a brief Description representation.
        """
        return 'Description(%s)' % self._v_nestedDescr



class metaIsDescription(type):
    "Helper metaclass to return the class variables as a dictionary "

    def __new__(cls, classname, bases, classdict):
        """ Return a new class with a "columns" attribute filled
        """

        newdict = {"columns":{},
                   }
        for k in classdict.keys():
            #if not (k.startswith('__') or k.startswith('_v_')):
            # We let pass _v_ variables to configure class behaviour
            if not (k.startswith('__')):
                newdict["columns"][k] = classdict[k]

        # Return a new class with the "columns" attribute filled
        return type.__new__(cls, classname, bases, newdict)



class IsDescription(object):
    """
    Description of the structure of a table or nested column.

    This class is designed to be used as an easy, yet meaningful way to
    describe the structure of new `Table` datasets or nested columns
    through the definition of *derived classes*.  In order to define
    such a class, you must declare it as descendant of `IsDescription`,
    with as many attributes as columns you want in your table.  The name
    of each attribute will become the name of a column, and its value
    will hold a description of it.

    Ordinary columns can be described using instances of the `Col`
    class.  Nested columns can be described by using classes derived
    from `IsDescription`, instances of it, or name-description
    dictionaries.  Derived classes can be declared in place (in which
    case the column takes the name of the class) or referenced by name.

    Nested columns can have a ``_v_pos`` special attribute which sets
    the *relative* position of the column among sibling columns *also
    having explicit positions*.  The ``pos`` constructor argument of
    `Col` intances is used for the same purpose.  Columns with no
    explicit position will be placed afterwards in alphanumeric order.

    Once you have created a description object, you can pass it to the
    `Table` constructor, where all the information it contains will be
    used to define the table structure.

    Special attributes
    ------------------

    These are the special attributes that the user can specify *when
    declaring* an `IsDescription` subclass to complement its *metadata*.

    _v_pos
        Sets the position of a possible nested column description among
        its sibling columns.

    Class variables
    ---------------

    The following attributes are *automatically created* when an
    `IsDescription` subclass is declared.  Please note that declared
    columns can no longer be accessed as normal class variables after
    its creation.

    columns
        Maps the name of each column in the description to its own
        descriptive object.
    """
    __metaclass__ = metaIsDescription



def descr_from_dtype(dtype_):
    """
    Get a description instance and byteorder from a (nested) NumPy dtype.
    """

    fields = {}
    fbyteorder = '|'
    for (name, (dtype, pos)) in dtype_.fields.items():
        kind = dtype.base.kind
        byteorder = dtype.base.byteorder
        if byteorder in '><=':
            if fbyteorder not in ['|', byteorder]:
                raise NotImplementedError(
                    "record arrays with mixed byteorders "
                    "are not supported yet, sorry" )
            fbyteorder = byteorder
        # Non-nested column
        if kind in 'biufSc':
            col = Col.from_dtype(dtype, pos=pos)
        # Nested column
        elif kind == 'V' and dtype.shape in [(), (1,)]:
            col, _ = descr_from_dtype(dtype)
            col._v_pos = pos
        else:
            raise NotImplementedError(
                "record arrays with columns with type description ``%s`` "
                "are not supported yet, sorry" % dtype )
        fields[name] = col

    return Description(fields), fbyteorder



if __name__=="__main__":
    """Test code"""

    class Info(IsDescription):
        _v_pos = 2
        Name = UInt32Col()
        Value = Float64Col()

    class Test(IsDescription):
        """A description that has several columns"""
        x = Col.from_type("int32", 2, 0, pos=0)
        y = Col.from_kind('float', dflt=1, shape=(2,3))
        z = UInt8Col(dflt=1)
        color = StringCol(2, dflt=" ")
        #color = UInt32Col(2)
        Info = Info()
        class info(IsDescription):
            _v_pos = 1
            name = UInt32Col()
            value = Float64Col(pos=0)
            y2 = Col.from_kind('float', dflt=1, shape=(2,3), pos=1)
            z2 = UInt8Col(dflt=1)
            class info2(IsDescription):
                y3 = Col.from_kind('float', dflt=1, shape=(2,3))
                z3 = UInt8Col(dflt=1)
                name = UInt32Col()
                value = Float64Col()
                class info3(IsDescription):
                    name = UInt32Col()
                    value = Float64Col()
                    y4 = Col.from_kind('float', dflt=1, shape=(2,3))
                    z4 = UInt8Col(dflt=1)

#     class Info(IsDescription):
#         _v_pos = 2
#         Name = StringCol(itemsize=2)
#         Value = ComplexCol(itemsize=16)

#     class Test(IsDescription):
#         """A description that has several columns"""
#         x = Col.from_type("int32", 2, 0, pos=0)
#         y = Col.from_kind('float', dflt=1, shape=(2,3))
#         z = UInt8Col(dflt=1)
#         color = StringCol(2, dflt=" ")
#         Info = Info()
#         class info(IsDescription):
#             _v_pos = 1
#             name = StringCol(itemsize=2)
#             value = ComplexCol(itemsize=16, pos=0)
#             y2 = Col.from_kind('float', dflt=1, shape=(2,3), pos=1)
#             z2 = UInt8Col(dflt=1)
#             class info2(IsDescription):
#                 y3 = Col.from_kind('float', dflt=1, shape=(2,3))
#                 z3 = UInt8Col(dflt=1)
#                 name = StringCol(itemsize=2)
#                 value = ComplexCol(itemsize=16)
#                 class info3(IsDescription):
#                     name = StringCol(itemsize=2)
#                     value = ComplexCol(itemsize=16)
#                     y4 = Col.from_kind('float', dflt=1, shape=(2,3))
#                     z4 = UInt8Col(dflt=1)

    # example cases of class Test
    klass = Test()
    #klass = Info()
    desc = Description(klass.columns)
    print "Description representation (short) ==>", desc
    print "Description representation (long) ==>", repr(desc)
    print "Column names ==>", desc._v_names
    print "Column x ==>", desc.x
    print "Column Info ==>", desc.Info
    print "Column Info.value ==>", desc.Info.Value
    print "Nested column names  ==>", desc._v_nestedNames
    print "Defaults ==>", desc._v_dflts
    print "Nested Formats ==>", desc._v_nestedFormats
    print "Nested Descriptions ==>", desc._v_nestedDescr
    print "Nested Descriptions (info) ==>", desc.info._v_nestedDescr
    print "Total size ==>", desc._v_dtype.itemsize


    # check _f_walk
    for object in desc._f_walk():
        if isinstance(object, Description):
            print "******begin object*************",
            print "name -->", object._v_name
            #print "name -->", object._v_dtype.name
            #print "object childs-->", object._v_names
            #print "object nested childs-->", object._v_nestedNames
            print "totalsize-->", object._v_dtype.itemsize
        else:
            #pass
            print "leaf -->", object._v_name, object.dtype



## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
