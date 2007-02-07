"""
Atom classes for describing dataset contents.

:Author: Ivan Vilata i Balaguer
:Contact: ivilata at carabos dot com
:License: BSD
:Created: December 16, 2004
:Revision: $Id$

See the docstrings of `Atom` classes for more info.

Variables
=========

`__docformat`__
    The format of documentation strings in this module.
`__version__`
    Repository version of this file.
`all_types`
    Set of all PyTables types.
`atom_map`
    Maps atom kinds to item sizes and atom classes.

    If there is a fixed set of possible item sizes for a given kind,
    the kind maps to another mapping from item size in bytes to atom
    class.  Otherwise, the kind maps directly to the atom class.

`deftype_from_kind`
    Maps atom kinds to their default atom type (if any).
"""

# Imports
# =======
import re
import inspect
import cPickle

import numpy

from tables.misc.enum import Enum


# Public variables
# ================
__docformat__ = 'reStructuredText'
"""The format of documentation strings in this module."""

__version__ = '$Revision$'
"""Repository version of this file."""

all_types = set()  # filled as atom classes are created
"""Set of all PyTables types."""

atom_map = {}  # filled as atom classes are created
"""
Maps atom kinds to item sizes and atom classes.

If there is a fixed set of possible item sizes for a given kind, the
kind maps to another mapping from item size in bytes to atom class.
Otherwise, the kind maps directly to the atom class.
"""

deftype_from_kind = {}  # filled as atom classes are created
"""Maps atom kinds to their default atom type (if any)."""


# Public functions
# ================
_type_re = re.compile(r'^([a-z]+)([0-9]*)$')
def split_type(type):
    """
    Split a PyTables `type` into a PyTables kind and an item size.

    Returns a tuple of ``(kind, itemsize)``.  If no item size is
    present in the `type` (in the form of a precision), the returned
    item size is `None`.

    >>> split_type('int32')
    ('int', 4)
    >>> split_type('string')
    ('string', None)
    >>> split_type('int20')
    Traceback (most recent call last):
      ...
    ValueError: precision must be a multiple of 8: 20
    >>> split_type('foo bar')
    Traceback (most recent call last):
      ...
    ValueError: malformed type: 'foo bar'
    """

    match = _type_re.match(type)
    if not match:
        raise ValueError("malformed type: %r" % type)
    kind, precision = match.groups()
    itemsize = None
    if precision:
        precision = int(precision)
        itemsize, remainder = divmod(precision, 8)
        if remainder:  # 0 could be a valid item size
            raise ValueError( "precision must be a multiple of 8: %d"
                              % precision )
    return (kind, itemsize)


# Private functions
# =================
def _normalize_shape(shape):
    """Check that the `shape` is safe to be used and return it as a tuple."""

    if type(shape) in (int, long):
        if shape < 1:
            raise ValueError( "shape value must be greater than 0: %d"
                              % shape )
        elif shape == 1:
            shape = ()  # 1 is a shorthand for ()
        else:
            shape = (shape,)  # N is a shorthand for (N,)
    elif type(shape) in (tuple, list):
        shape = tuple(shape)
    else:
        raise TypeError( "shape must be an integer, tuple or list: %r"
                         % (shape,) )

    ## XXX Get from HDF5 library if possible.
    # HDF5 does not support ranks greater than 32
    if len(shape) > 32:
        raise ValueError(
            "shapes with rank > 32 are not supported: %r" % (shape,) )

    return shape

def _normalize_default(value, dtype):
    """Return `value` as a valid default of NumPy type `dtype`."""
    # Create NumPy objects as defaults
    # This is better in order to serialize them as attributes
    if value is None:
        value = 0
    basedtype = dtype.base
    try:
        default = numpy.array(value, dtype=basedtype)
    except ValueError:
        array = numpy.array(value)
        if array.shape != basedtype.shape:
            raise
        # Maybe nested dtype with "scalar" value.
        default = numpy.array(value, dtype=basedtype.base)
    # 0-dim arrays will be representented as NumPy scalars
    # (PyTables attribute convention)
    if default.shape == ():
        default = default[()]
    return default


# Helper classes
# ==============
class MetaAtom(type):

    """
    Atom metaclass.

    This metaclass ensures that data about atom classes gets inserted
    into the suitable registries.
    """

    def __init__(class_, name, bases, dict_):
        super(MetaAtom, class_).__init__(name, bases, dict_)

        kind = dict_.get('kind')
        itemsize = dict_.get('itemsize')
        type_ = dict_.get('type')
        deftype = dict_.get('_deftype')

        if kind and deftype:
            deftype_from_kind[kind] = deftype

        if type_:
            all_types.add(type_)

        if kind and itemsize and not hasattr(itemsize, '__int__'):
            # Atom classes with a non-fixed item size do have an
            # ``itemsize``, but it's not a number (e.g. property).
            atom_map[kind] = class_
            return

        if kind:  # first definition of kind, make new entry
            atom_map[kind] = {}

        if itemsize and hasattr(itemsize, '__int__'):  # fixed
            kind = class_.kind  # maybe from superclasses
            atom_map[kind][int(itemsize)] = class_


# Atom classes
# ============
class Atom(object):

    """
    Defines the type of atomic cells stored in a dataset.

    The meaning of *atomic* is that individual elements of a cell can
    not be extracted directly by indexation (i.e. ``__getitem__()``)
    of the dataset; e.g. if a dataset has shape (2, 2) and its atoms
    have shape (3,), to get the third element of the cell at (1, 0)
    one should use ``dataset[1,0][2]`` instead of ``dataset[1,0,2]``.

    Atoms have the following common attributes:

    `kind`
        The PyTables kind of the atom (a string).
    `type`
        The PyTables type of the atom (a string).
    `shape`
        The shape of the atom (a tuple, ``()`` for scalar atoms).
    `dflt`
        The default value of the atom.
    `size`
        Total size in bytes of the atom.
    `itemsize`
        Size in bytes of a sigle item in the atom.
    `dtype`
        The NumPy ``dtype`` that most closely matches this atom.
    `recarrtype`
        String type to be used in ``numpy.rec.array()``.
    """

    # Register data for all subclasses.
    __metaclass__ = MetaAtom

    # Class methods
    # ~~~~~~~~~~~~~
    @classmethod
    def prefix(class_):
        """Return the atom class prefix."""
        cname = class_.__name__
        return cname[:cname.rfind('Atom')]

    @classmethod
    def from_sctype(class_, sctype, shape=1, dflt=None):
        """
        Create an `Atom` from a NumPy scalar type `sctype`.

        Optional shape and default value may be specified as the
        `shape` and `dflt` arguments, respectively.  Information in
        the `sctype` not represented in an `Atom` is ignored.

        >>> import numpy
        >>> Atom.from_sctype(numpy.int16, shape=(2, 2))
        Int16Atom(shape=(2, 2), dflt=0)
        >>> Atom.from_sctype('S5', dflt='hello')
        Traceback (most recent call last):
          ...
        ValueError: unknown NumPy scalar type: 'S5'
        >>> Atom.from_sctype('Float64')
        Float64Atom(shape=(), dflt=0.0)
        """
        if ( not isinstance(sctype, type)
             or not issubclass(sctype, numpy.generic) ):
            if sctype not in numpy.sctypeDict:
                raise ValueError("unknown NumPy scalar type: %r" % (sctype,))
            sctype = numpy.sctypeDict[sctype]
        return class_.from_dtype(numpy.dtype((sctype, shape)), dflt)

    @classmethod
    def from_dtype(class_, dtype, dflt=None):
        """
        Create an `Atom` from a NumPy `dtype`.

        An optional default value may be specified as the `dflt`
        argument.  Information in the `dtype` not represented in an
        `Atom` is ignored.

        >>> import numpy
        >>> Atom.from_dtype(numpy.dtype((numpy.int16, (2, 2))))
        Int16Atom(shape=(2, 2), dflt=0)
        >>> Atom.from_dtype(numpy.dtype('S5'), dflt='hello')
        StringAtom(itemsize=5, shape=(), dflt='hello')
        >>> Atom.from_dtype(numpy.dtype('Float64'))
        Float64Atom(shape=(), dflt=0.0)
        """
        basedtype = dtype.base
        if basedtype.names:
            raise ValueError( "compound data types are not supported: %r"
                              % dtype )
        if basedtype.shape != ():
            raise ValueError( "nested data types are not supported: %r"
                              % dtype )
        if basedtype.kind == 'S':  # can not reuse something like 'string80'
            itemsize = basedtype.itemsize
            return class_.from_kind('string', itemsize, dtype.shape, dflt)
        # Most NumPy types have direct correspondence with PyTables types.
        return class_.from_type(basedtype.name, dtype.shape, dflt)

    @classmethod
    def from_type(class_, type, shape=1, dflt=None):
        """
        Create an `Atom` from a PyTables `type`.

        Optional shape and default value may be specified as the
        `shape` and `dflt` arguments, respectively.

        >>> Atom.from_type('bool')
        BoolAtom(shape=(), dflt=False)
        >>> Atom.from_type('int16', shape=(2, 2))
        Int16Atom(shape=(2, 2), dflt=0)
        >>> Atom.from_type('string40', dflt='hello')
        Traceback (most recent call last):
          ...
        ValueError: unknown type: 'string40'
        >>> Atom.from_type('Float64')
        Traceback (most recent call last):
          ...
        ValueError: unknown type: 'Float64'
        """
        if type not in all_types:
            raise ValueError("unknown type: %r" % (type,))
        kind, itemsize = split_type(type)
        return class_.from_kind(kind, itemsize, shape, dflt)

    @classmethod
    def from_kind(class_, kind, itemsize=None, shape=1, dflt=None):
        """
        Create an `Atom` from a PyTables `kind`.

        Optional item size, shape and default value may be specified
        as the `itemsize`, `shape` and `dflt` arguments, respectively.
        Bear in mind that not all atoms support a default item size.

        >>> Atom.from_kind('int', itemsize=2, shape=(2, 2))
        Int16Atom(shape=(2, 2), dflt=0)
        >>> Atom.from_kind('int', shape=(2, 2))
        Int32Atom(shape=(2, 2), dflt=0)
        >>> Atom.from_kind('string', itemsize=5, dflt='hello')
        StringAtom(itemsize=5, shape=(), dflt='hello')
        >>> Atom.from_kind('string', dflt='hello')
        Traceback (most recent call last):
          ...
        ValueError: no default item size for kind ``string``
        >>> Atom.from_kind('Float')
        Traceback (most recent call last):
          ...
        ValueError: unknown kind: 'Float'
        """

        kwargs = {'shape': shape}
        if kind not in atom_map:
            raise ValueError("unknown kind: %r" % (kind,))
        # If no `itemsize` is given, try to get the default type of the
        # kind (which has a fixed item size).
        if itemsize is None:
            if kind not in deftype_from_kind:
                raise ValueError( "no default item size for kind ``%s``"
                                  % kind )
            type_ = deftype_from_kind[kind]
            kind, itemsize = split_type(type_)
        kdata = atom_map[kind]
        # Look up the class and set a possible item size.
        if hasattr(kdata, 'kind'):  # atom class: non-fixed item size
            atomclass = kdata
            kwargs['itemsize'] = itemsize
        else:  # dictionary: fixed item size
            if itemsize not in kdata:
                isizes = sorted(kdata.keys())
                raise ValueError( "invalid item size for kind ``%s``: %r; "
                                  "it must be one of ``%r``"
                                  % (kind, itemsize, isizes) )
            atomclass = kdata[itemsize]
        # Only set a `dflt` argument if given (`None` may not be understood).
        if dflt is not None:
            kwargs['dflt'] = dflt

        return atomclass(**kwargs)

    # Properties
    # ~~~~~~~~~~
    size = property(
        lambda self: self.dtype.itemsize,
        None, None, "Total size in bytes of the atom." )
    recarrtype = property(
        lambda self: str(self.dtype.shape) + self.dtype.base.str[1:],
        None, None, "String type to be used in ``numpy.rec.array()``." )

    # Special methods
    # ~~~~~~~~~~~~~~~
    def __init__(self, nptype, shape, dflt):
        if not hasattr(self, 'type'):
            raise NotImplementedError( "``%s`` is an abstract class; "
                                       "please use one of its subclasses"
                                       % self.__class__.__name__ )
        self.shape = shape = _normalize_shape(shape)
        self.dtype = dtype = numpy.dtype((nptype, shape))
        self.dflt = _normalize_default(dflt, dtype)

    def __repr__(self):
        args = 'shape=%s, dflt=%r' % (self.shape, self.dflt)
        if not hasattr(self.__class__.itemsize, '__int__'):  # non-fixed
            args = 'itemsize=%s, %s' % (self.itemsize, args)
        return '%s(%s)' % (self.__class__.__name__, args)

    # Public methods
    # ~~~~~~~~~~~~~~
    def copy(self, **override):
        """
        Get a copy of the atom, possibly overriding some arguments.

        Constructor arguments to be overridden must be passed as
        keyword arguments.

        >>> atom1 = StringAtom(itemsize=12)
        >>> atom2 = atom1.copy()
        >>> print atom1
        StringAtom(itemsize=12, shape=(), dflt='')
        >>> print atom2
        StringAtom(itemsize=12, shape=(), dflt='')
        >>> atom1 is atom2
        False
        >>> atom3 = atom1.copy(itemsize=100, shape=(2, 2))
        >>> print atom3
        StringAtom(itemsize=100, shape=(2, 2), dflt='')
        >>> atom1.copy(foobar=42)
        Traceback (most recent call last):
          ...
        TypeError: __init__() got an unexpected keyword argument 'foobar'
        """
        newargs = self._get_init_args()
        newargs.update(override)
        return self.__class__(**newargs)

    # Private methods
    # ~~~~~~~~~~~~~~~
    def _get_init_args(self):
        """
        Get a dictionary of instance constructor arguments.

        This implementation works on classes which use the same names
        for both constructor arguments and instance attributes.
        """
        return dict( (arg, getattr(self, arg))
                     for arg in inspect.getargspec(self.__init__)[0]
                     if arg != 'self' )


class StringAtom(Atom):
    """Defines an atom of type ``string``."""
    kind = 'string'
    itemsize = property(
        lambda self: self.dtype.base.itemsize,
        None, None, "Size in bytes of a sigle item in the atom." )
    type = 'string'
    _defvalue = ''

    def __init__(self, itemsize, shape=1, dflt=_defvalue):
        if not hasattr(itemsize, '__int__') or int(itemsize) < 0:
            raise ValueError( "invalid item size for kind ``%s``: %r; "
                              "it must be a positive integer"
                              % ('string', itemsize) )
        Atom.__init__(self, 'S%d' % itemsize, shape, dflt)


class BoolAtom(Atom):
    """Defines an atom of type ``bool``."""
    kind = 'bool'
    itemsize = 1
    type = 'bool'
    _deftype = 'bool8'
    _defvalue = False
    def __init__(self, shape=1, dflt=_defvalue):
        Atom.__init__(self, self.type, shape, dflt)


class IntAtom(Atom):
    """Defines an atom of a signed integral type."""
    kind = 'int'
    signed = True
    _deftype = 'int32'
    _defvalue = 0

class UIntAtom(Atom):
    """Defines an atom of an unsigned integral type."""
    kind = 'uint'
    signed = False
    _deftype = 'uint32'
    _defvalue = 0

class FloatAtom(Atom):
    """Defines an atom of a floating point type."""
    kind = 'float'
    _deftype = 'float64'
    _defvalue = 0.0


def _create_numeric_class(baseclass, itemsize):
    """
    Create a numeric atom class with the given `baseclass` and an
    `itemsize`.
    """
    prefix = '%s%d' % (baseclass.prefix(), itemsize * 8)
    type_ = prefix.lower()
    classdict = { 'itemsize': itemsize, 'type': type_,
                  '__doc__': "Defines an atom of type ``%s``." % type_ }
    def __init__(self, shape=1, dflt=baseclass._defvalue):
        Atom.__init__(self, self.type, shape, dflt)
    classdict['__init__'] = __init__
    return type('%sAtom' % prefix, (baseclass,), classdict)


def _generate_integral_classes():
    """Generate all integral classes."""
    for baseclass in [IntAtom, UIntAtom]:
        for itemsize in [1, 2, 4, 8]:
            newclass = _create_numeric_class(baseclass, itemsize)
            yield newclass

def _generate_floating_classes():
    """Generate all floating classes."""
    for itemsize in [4, 8]:
        newclass = _create_numeric_class(FloatAtom, itemsize)
        yield newclass

# Create all numeric atom classes.
for _classgen in [_generate_integral_classes, _generate_floating_classes]:
    for _newclass in _classgen():
        exec '%s = _newclass' % _newclass.__name__
del _classgen, _newclass


class ComplexAtom(Atom):
    """Defines an atom of a complex type."""

    # This definition is a little more complex (no pun intended)
    # because, although the complex kind is a normal numerical one,
    # the usage of bottom-level classes is artificially forbidden.
    # Everything will be back to normality when people has stopped
    # using the old bottom-level complex classes.

    kind = 'complex'
    itemsize = property(
        lambda self: self.dtype.base.itemsize,
        None, None, "Size in bytes of a sigle item in the atom." )
    _deftype = 'complex128'
    _defvalue = 0j

    # Only instances have a `type` attribute, so complex types must be
    # registered by hand.
    all_types.add('complex64')
    all_types.add('complex128')

    def __init__(self, itemsize, shape=1, dflt=_defvalue):
        isizes = [8, 16]
        if itemsize not in isizes:
            raise ValueError( "invalid item size for kind ``%s``: %r; "
                              "it must be one of ``%r``"
                              % ('complex', itemsize, isizes) )
        self.type = '%s%d' % (self.kind, itemsize * 8)
        Atom.__init__(self, self.type, shape, dflt)

class _ComplexErrorAtom(ComplexAtom):
    """Reminds the user to stop using the old complex atom names."""
    __metaclass__ = type  # do not register anything about this class
    def __init__(self, shape=1, dflt=ComplexAtom._defvalue):
        raise TypeError(
            "to avoid confusions with PyTables 1.X complex atom names, "
            "please use ``ComplexAtom(itemsize=N)``, "
            "where N=8 for single precision complex atoms, "
            "and N=16 for double precision complex atoms" )
Complex32Atom = Complex64Atom = Complex128Atom = _ComplexErrorAtom


class TimeAtom(Atom):

    """
    Defines an atom of time type.

    There are two distinct supported types of time: a 32 bit integer
    value and a 64 bit floating point value.  Both of them reflect the
    number of seconds since the Epoch.  This atom has the property of
    being stored using the HDF5 time datatypes.
    """
    kind = 'time'
    _deftype = 'time32'

class Time32Atom(TimeAtom):
    """Defines an atom of type ``time32``."""
    itemsize = 4
    type = 'time32'
    _defvalue = 0
    def __init__(self, shape=1, dflt=_defvalue):
        Atom.__init__(self, 'int32', shape, dflt)

class Time64Atom(TimeAtom):
    """Defines an atom of type ``time64``."""
    itemsize = 8
    type = 'time64'
    _defvalue = 0.0
    def __init__(self, shape=1, dflt=_defvalue):
        Atom.__init__(self, 'float64', shape, dflt)


class EnumAtom(Atom):

    """
    Description of an atom of an enumerated type.

    Instances of this class describe the atom type used to store
    enumerated values.  Those values belong to an enumerated type,
    defined by the first argument (``enum``) in the constructor of the
    atom, which accepts the same kinds of arguments as the ``Enum``
    class.  The enumerated type is stored in the ``enum`` attribute of
    the atom.

    A default value must be specified as the second argument
    (``dflt``) in the constructor; it must be the *name* (a string) of
    one of the enumerated values in the enumerated type.  When the
    atom is created, the corresponding concrete value is broadcast and
    stored in the ``dflt`` attribute (setting different default values
    for items in a multidimensional atom is not supported yet).  If
    the name does not match any value in the enumerated type, a
    ``KeyError`` is raised.

    Another atom must be specified as the ``base`` argument in order
    to determine the base type used for storing the values of
    enumerated values in memory and disk.  This *storage atom* is kept
    in the ``base`` attribute of the created atom.  As a shorthand,
    you may specify a PyTables type instead of the storage atom,
    implying that this has a scalar shape.

    The storage atom should be able to represent each and every
    concrete value in the enumeration.  If it is not, a ``TypeError``
    is raised.  The default value of the storage atom is ignored.

    The ``type`` attribute of enumerated atoms is always ``'enum'``.

    Examples
    --------

    The next C ``enum`` construction::

      enum myEnum {
        T0,
        T1,
        T2
      };

    would correspond to the following PyTables declaration:

    >>> myEnumAtom = EnumAtom(['T0', 'T1', 'T2'], 'T0', 'int32')

    Please note the ``dflt`` argument with a value of ``'T0'``.  Since
    the concrete value matching ``T0`` is unknown right now (we have
    not used explicit concrete values), using the name is the only
    option left for defining a default value for the atom.

    The chosen representation of values for this enumerated atom uses
    unsigned 32-bit integers, which surely wastes quite a lot of
    memory.  Another size could be selected by using the ``base``
    argument (this time with a full-blown storage atom):

    >>> myEnumAtom = EnumAtom(['T0', 'T1', 'T2'], 'T0', UInt8Atom())

    You can also define multidimensional arrays for data elements:

    >>> myEnumAtom = EnumAtom(
    ...    ['T0', 'T1', 'T2'], 'T0', base='uint32', shape=(3,2))

    for 3x2 arrays of ``uint32``.
    """

    # Registering this class in the class map may be a little wrong,
    # since the ``Atom.from_kind()`` method fails miserably with
    # enumerations, as they don't support an ``itemsize`` argument.
    # However, resetting ``__metaclass__`` to ``type`` doesn't seem to
    # work and I don't feel like creating a subclass of ``MetaAtom``.

    kind = 'enum'
    type = 'enum'

    # Properties
    # ~~~~~~~~~~
    itemsize = property(
        lambda self: self.dtype.base.itemsize,
        None, None, "Size in bytes of a sigle item in the atom." )

    # Private methods
    # ~~~~~~~~~~~~~~~
    def _checkBase(self, base):
        """Check the `base` storage atom."""

        if base.kind == 'enum':
            raise TypeError( "can not use an enumerated atom "
                             "as a storage atom: %r" % base )

        # Check whether the storage atom can represent concrete values
        # in the enumeration...
        basedtype = base.dtype
        pyvalues = [value for (name, value) in self.enum]
        try:
            npgenvalues = numpy.array(pyvalues)
        except ValueError:
            raise TypeError("concrete values are not uniformly-shaped")
        try:
            npvalues = numpy.array(npgenvalues, dtype=basedtype.base)
        except ValueError:
            raise TypeError( "storage atom type is incompatible with "
                             "concrete values in the enumeration" )
        if npvalues.shape[1:] != basedtype.shape:
            raise TypeError( "storage atom shape does not match that of "
                             "concrete values in the enumeration" )
        if npvalues.tolist() != npgenvalues.tolist():
            raise TypeError( "storage atom type lacks precision for "
                             "concrete values in the enumeration" )

        # ...with some implementation limitations.
        if not npvalues.dtype.kind in ['i', 'u']:
            raise NotImplementedError( "only integer concrete values "
                                       "are supported for the moment, sorry" )
        if len(npvalues.shape) > 1:
            raise NotImplementedError( "only scalar concrete values "
                                       "are supported for the moment, sorry" )

    def _get_init_args(self):
        """Get a dictionary of instance constructor arguments."""
        return dict( enum=self.enum, dflt=self._defname,
                     base=self.base, shape=self.shape )

    # Special methods
    # ~~~~~~~~~~~~~~~
    def __init__(self, enum, dflt, base, shape=1):
        if not isinstance(enum, Enum):
            enum = Enum(enum)
        self.enum = enum

        if type(base) is str:
            base = Atom.from_type(base)
        self._checkBase(base)
        self.base = base

        default = enum[dflt]  # check default value
        self._defname = dflt  # kept for representation purposes

        # These are kept to ease dumping this particular
        # representation of the enumeration to storage.
        names, values = [], []
        for (name, value) in enum:
            names.append(name)
            values.append(value)
        basedtype = self.base.dtype

        self._names = names
        self._values = numpy.array(values, dtype=basedtype.base)

        Atom.__init__(self, basedtype, shape, default)

    def __repr__(self):
        return ( 'EnumAtom(enum=%r, dflt=%r, base=%r, shape=%r)'
                 % (self.enum, self._defname, self.base, self.shape) )


# Pseudo-atom classes
# ===================
class PseudoAtom(object):
    """
    Pseudo-atoms can only be used in ``VLArray`` nodes.

    They can be recognised because they also have `kind`, `type` and
    `shape` attributes, but no `size`, `itemsize` or `dflt` ones.
    Instead, they have a `base` atom which defines the elements used
    for storage.
    """
    def __repr__(self):
        return '%s()' % self.__class__.__name__

    def toarray(self, object_):
        """Convert an `object_` into an array of base atoms."""
        raise NotImplementedError

    def fromarray(self, array):
        """Convert an `array` of base atoms into an object."""
        raise NotImplementedError

class _BufferedAtom(PseudoAtom):
    """Pseudo-atom which stores data as a buffer (array of bytes)."""
    shape = ()
    base = UInt8Atom()

    def toarray(self, object_):
        buffer_ = self._tobuffer(object_)
        array = numpy.ndarray( buffer=buffer_, dtype=self.base.dtype,
                               shape=len(buffer_) )
        return array

    def _tobuffer(self, object_):
        """Convert an `object_` into a buffer."""
        raise NotImplementedError

class VLStringAtom(_BufferedAtom):
    """
    Defines an atom of type ``vlstring``

    This supports storing variable length strings as components of a
    ``VLArray``.  Unicode strings are stored with UTF-8 encoding.
    """
    kind = 'vlstring'
    type = 'vlstring'

    def _tobuffer(self, object_):
        if not hasattr(object_, 'encode'):
            raise TypeError( "object does not look like a string: %r"
                             % object_ )
        return object_.encode('utf-8')

    def fromarray(self, array):
        return array.tostring().decode('utf-8')

class ObjectAtom(_BufferedAtom):
    """
    Defines an atom of type ``object``

    This supports storing arbitrary pickled objects as components of a
    ``VLArray``.
    """
    kind = 'object'
    type = 'object'

    def _tobuffer(self, object_):
        return cPickle.dumps(object_, 0)

    def fromarray(self, array):
        # We have to check for an empty array because of a possible
        # bug in HDF5 which makes it claim that a dataset has one
        # record when in fact it is empty.
        if array.size == 0:
            return None
        return cPickle.loads(array.tostring())


# Main part
# =========
def _test():
    """Run ``doctest`` on this module."""
    import doctest
    doctest.testmod()

if __name__ == '__main__':
    _test()
