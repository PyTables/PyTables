# -*- coding: utf-8 -*-

########################################################################
#
# License: BSD
# Created: October 2, 2004
# Author:  Ivan Vilata i Balaguer - reverse:net.selidor@ivan
#
# $Id$
#
########################################################################

"""A file interface to nodes for PyTables databases.

The FileNode module provides a file interface for using inside of
PyTables database files.  Use the new_node() function to create a brand
new file node which can be read and written as any ordinary Python
file.  Use the open_node() function to open an existing (i.e. created
with new_node()) node for read-only or read-write access.  Read acces
is always available.  Write access (enabled on new files and files
opened with mode 'a+') only allows appending data to a file node.

See :ref:`filenode_usersguide` for instructions on use.

"""

import os
import warnings

import numpy

import tables
from tables._past import previous_api


NodeType = 'file'
"""Value for NODE_TYPE node system attribute."""

NodeTypeVersions = [1, 2]
"""Supported values for NODE_TYPE_VERSION node system attribute."""


def new_node(h5file, **kwargs):
    """Creates a new file node object in the specified PyTables file object.

    Additional named arguments where and name must be passed to specify where
    the file node is to be created. Other named arguments such as title and
    filters may also be passed.

    The special named argument expectedsize, indicating an estimate of the file
    size in bytes, may also be passed. It returns the file node object.

    """

    return RAFileNode(None, h5file, **kwargs)

newNode = previous_api(new_node)


def open_node(node, mode='r'):
    """Opens an existing file node.

    Returns a file node object from the existing specified PyTables node.
    If mode is not specified or it is 'r', the file can only be read,
    and the pointer is positioned at the beginning of the file.
    If mode is 'a+', the file can be read and appended, and the pointer
    is positioned at the end of the file.

    """

    if mode == 'r':
        return ROFileNode(node)
    elif mode == 'a+':
        return RAFileNode(node, None)
    else:
        raise IOError("invalid mode: %s" % (mode,))

openNode = previous_api(open_node)


class ReadableMixin:
    """Mix-in class which provides reading methods for readable file nodes.

    It also defines the 'line_separator' property, which contains the string
    used as a line separator, and defaults to os.linesep.
    It can be set to any reasonably-sized string you want.

    This class requires support for:

    * 'offset' and 'node' attributes
    * _check_not_closed() and seek() methods

    """

    # The number of bytes readline() reads at a time.
    _line_chunksize = 128

    # The line separator string.
    _line_separator = os.linesep.encode('ascii')

    # The line separator string property methods.
    def get_line_separator(self):
        """Returns the line separator string."""

        return self._line_separator

    getLineSeparator = previous_api(get_line_separator)

    def set_line_separator(self, value):
        """Sets the line separator string.

        Raises ValueError if the string is empty or too long.

        """

        if not isinstance(value, bytes):
            raise TypeError('the line separator must be a string of bytes')

        if value == b'':
            raise ValueError("line separator string is empty")
        elif len(value) > self._line_chunksize:
            raise ValueError("sorry, line separator string is too long")
        else:
            self._line_separator = value

    setLineSeparator = previous_api(set_line_separator)

    def del_line_separator(self):
        "Deletes the 'line_separator' property."

        del self._line_separator

    delLineSeparator = previous_api(del_line_separator)

    # The line separator string property.
    line_separator = property(
        get_line_separator, set_line_separator, del_line_separator,
        """A property containing the line separator string.""")

    lineSeparator = previous_api(line_separator)

    def __iter__(self):
        return self

    def next(self):
        """Gets the next line of text.

        Raises StopIteration when finished.
        See file.next.__doc__ for more information.

        """

        # The use of this method is compatible with the use of readline().
        line = self.readline()
        if len(line) == 0:
            raise StopIteration
        return line

    def read(self, size=None):
        """Reads at most 'size' bytes.

        See file.read.__doc__ for more information.

        """

        self._check_not_closed()

        # 2004-08-03: Reading from beyond the last row raises an IndexError.
        #   Moreover, the pointer should not be incremented.
        if self.offset >= self.node.nrows:
            return ''

        start = self.offset
        if size is None or size < 0:
            # Read the entire file.
            # 2004-08-03: A None value would only read one row.
            stop = self.node.nrows
        else:
            # Read the specified number of rows, if available.
            # 2004-08-04: Reading beyond the last row is allowed.
            stop = self.offset + size

        data = self.node.read(start, stop).tostring()
        self.offset += len(data)
        return data

    def readline(self, size=-1):
        """Reads the next text line.

        See file.readline.__doc__ for more information.

        """

        self._check_not_closed()

        # Set the remaining bytes to read to the specified size.
        remsize = size

        lseplen = len(self._line_separator)
        partial = []
        finished = False

        while not finished:
            # Read a string limited by the remaining number of bytes.
            if size <= 0:
                ibuff = self.read(self._line_chunksize)
            else:
                ibuff = self.read(min(remsize, self._line_chunksize))
            ibufflen = len(ibuff)
            remsize -= ibufflen

            if ibufflen >= lseplen:
                # Separator fits, look for EOL string.
                eolindex = ibuff.find(self._line_separator)
            elif ibufflen == 0:
                # EOF was immediately reached.
                finished = True
                continue
            else:  # ibufflen < lseplen
                # EOF was hit and separator does not fit. ;)
                partial.append(ibuff)
                finished = True
                continue

            if eolindex >= 0:
                # Found an EOL. If there are trailing characters,
                # cut the input buffer and seek back;
                # else add the whole input buffer.
                trailing = ibufflen - lseplen - eolindex  # Bytes beyond EOL.
                if trailing > 0:
                    obuff = ibuff[:-trailing]
                    self.seek(-trailing, 1)
                    remsize += trailing
                else:
                    obuff = ibuff
                finished = True
            elif lseplen > 1 and (size <= 0 or remsize > 0):
                # Seek back a little since the end of the read string
                # may have fallen in the middle of the line separator.
                obuff = ibuff[:-lseplen + 1]
                self.seek(-lseplen + 1, 1)
                remsize += lseplen - 1
            else:  # eolindex<0 and (lseplen<=1 or (size>0 and remsize<=0))
                # Did not find an EOL, add the whole input buffer.
                obuff = ibuff

            # Append (maybe cut) buffer.
            partial.append(obuff)

            # If a size has been specified and the remaining count
            # reaches zero, the reading is finished.
            if size > 0 and remsize <= 0:
                finished = True

        return b''.join(partial)

    def readlines(self, sizehint=-1):
        """Reads the text lines into a list.

        See file.readlines.__doc__ for more information.

        """

        # Set the remaining bytes to read to the size hint.
        remsize = sizehint

        lines = []
        finished = False

        while not finished:
            # Read a line limited by the remaining number of bytes.
            if sizehint <= 0:
                line = self.readline()
            else:
                line = self.readline(remsize)
            remsize -= len(line)

            # An empty line finishes the reading.
            if len(line) > 0:
                lines.append(line)
            else:
                finished = True
                continue

            # If a size hint has been specified and the remaining count
            # reaches zero, the reading is finished.
            if sizehint > 0 and remsize <= 0:
                finished = True

        return lines

    def xreadlines(self):
        """For backward compatibility.

        See file.xreadlines.__doc__ for more information.

        """

        return self


class NotReadableMixin:
    """Mix-in class which provides reading methods for non-readable file nodes.

    This class requires support for:

    * _check_not_closed() method

    """

    def _not_readable_error(self):
        """_not_readable_error() -> None

        Raises a common IOError exception for non-readable file nodes.

        """

        raise IOError("the file is not readable")

    _notReadableError = previous_api(_not_readable_error)

    # The definition of those methods may seem odd
    # but it is the way Python (2.3) files work.

    def __iter__(self):
        return self

    def next(self):
        """Gets the next line of text.

        :returns:
            a string of bytes

        Raises IOError.
        See file.next.__doc__ for more information.

        """

        self._check_not_closed()
        self._not_readable_error()

    def read(self, size=None):
        """Reads at most 'size' bytes.

        :returns:
            a string of bytes

        Raises IOError.
        See file.read.__doc__ for more information.

        """

        self._check_not_closed()
        self._not_readable_error()

    def readline(self, size=-1):
        """Reads the next text line.

        :returns:
            a string of bytes

        Raises IOError.
        See file.readline.__doc__ for more information.

        """

        self._check_not_closed()
        self._not_readable_error()

    def readlines(self, sizehint=-1):
        """Reads the text lines.

        :returns:
            a list of strings of bytes

        Raises IOError.
        See file.readlines.__doc__ for more information.

        """

        self._check_not_closed()
        self._not_readable_error()

    def xreadlines(self):
        """xreadlines() -> self.  For backward compatibility.

        See file.xreadlines.__doc__ for more information.

        """

        return self


class NotWritableMixin:
    """Mix-in class which provides writing methods for non-writable file nodes.

    This class requires support for:

        * _check_not_closed() method

    """

    # The definition of those methods may seem odd
    # but it is the way Python (2.3) files work.

    def _notWritableError(self):
        """_notWritableError() -> None

        Raises a common IOError exception for non-writable file nodes.

        """

        raise IOError("the file is not writable")

    def truncate(self, size=None):
        """Truncates the file node to at most 'size' bytes.

        This raises an IOError when called on read-only nodes.

        See file.truncate.__doc__ for more information.

        """

        self._check_not_closed()
        self._notWritableError()

    def write(self, string):
        """Writes the string to the file.

        This raises an IOError when called on read-only nodes.

        See file.write.__doc__ for more information.

        """

        self._check_not_closed()
        self._notWritableError()

    def writelines(self, sequence):
        """Writes the strings to the file.

        This raises an IOError when called on read-only nodes.

        See file.writelines.__doc__ for more information.

        """

        self._check_not_closed()
        self._notWritableError()


class AppendableMixin:
    """Mix-in class which provides writing methods for appendable file nodes.

    This class requires support for:

    * 'offset', 'node', '_vType' and '_vShape' attributes
    * _check_not_closed() method

    """

    def _append_zeros(self, size):
        """_append_zeros(size) -> None.  Appends a string of zeros.

        Appends a string of 'size' zeros to the array,
        without moving the file pointer.

        """

        # Appending an empty array would raise an error.
        if size == 0:
            return

        # XXX This may be redone to avoid a potentially large in-memory array.
        self.node.append(
            numpy.zeros(dtype=self._vType, shape=self._vShape(size)))

    _appendZeros = previous_api(_append_zeros)

    def truncate(self, size=None):
        """Truncates the file node to at most 'size' bytes.

        Currently, this method only makes sense to grow the file node,
        since data can not be rewritten nor deleted.
        See file.truncate.__doc__ for more information.

        """

        self._check_not_closed()

        if size is None:
            size = self.offset
        if size < self.node.nrows:
            raise IOError("truncating is only allowed for growing a file")
        self._append_zeros(size - self.node.nrows)

    def write(self, string):
        """Writes the string to the file.

        Writing an empty string does nothing, but requires the file to be open.
        See file.write.__doc__ for more information.

        """

        self._check_not_closed()

        # This mimics the behaviour of normal Python (2.3) files,
        # where writing an empty string does absolutely nothing
        # (not even moving the pointer of append-only files).
        if len(string) == 0:
            return

        # Is the pointer beyond the real end of data?
        end2off = self.offset - self.node.nrows
        if end2off > 0:
            # Zero-fill the gap between the end of data and the pointer.
            self._append_zeros(end2off)

        # Move the pointer to the end of the (newly written) data.
        self.offset = self.node.nrows

        # Append data.
        self.node.append(numpy.ndarray(buffer=string, dtype=self._vType,
                                       shape=self._vShape(len(string))))

        # Move the pointer to the end of the written data.
        self.offset = self.node.nrows

    def writelines(self, sequence):
        """Writes the sequence of strings to the file.

        See file.writelines.__doc__ for more information.

        """

        for line in sequence:
            self.write(line)


class FileNode(object):
    """This is the ancestor of ROFileNode and RAFileNode (see below).

    Instances of these classes are returned when new_node() or
    open_node() are called. It represents a new file node associated
    with a PyTables node, providing a standard Python file interface
    to it.

    This abstract class provides only an implementation of the reading methods
    needed to implement a file-like object over a PyTables node. The attribute
    set of the node becomes available via the attrs property. You can add
    attributes there, but try to avoid attribute names in all caps or starting
    with '_', since they may clash with internal attributes.

    The node used as storage is also made available via the read-only attribute
    node. Please do not tamper with this object if it's avoidable, since you
    may break the operation of the file node object.

    The line_separator property contains the string used as a line separator,
    and defaults to os.linesep. It can be set to any reasonably-sized string
    you want.

    The constructor sets the closed, softspace and _line_separator attributes
    to their initial values, as well as the node attribute to None.
    Sub-classes should set the node, mode and offset attributes.

    Version 1 implements the file storage as a UInt8 uni-dimensional EArray.
    Version 2 uses an UInt8 N vector EArray.

    """

    # The atom representing a byte in the array, for each version.
    _byteShape = [
        None,
        (0, 1),
        (0,),
    ]

    # A lambda to turn a size into a shape, for each version.
    _sizeToShape = [
        None,
        lambda l: (l, 1),
        lambda l: (l, ),
    ]

    # The attribute set property methods.
    def get_attrs(self):
        """Returns the attribute set of the file node."""

        return self.node.attrs

    getAttrs = previous_api(get_attrs)

    def set_attrs(self, value):
        """set_attrs(string) -> None.  Raises ValueError."""

        raise ValueError("changing the whole attribute set is not allowed")

    setAttrs = previous_api(set_attrs)

    def del_attrs(self):
        """del_attrs() -> None.  Raises ValueError."""

        raise ValueError("deleting the whole attribute set is not allowed")

    delAttrs = previous_api(del_attrs)

    # The attribute set property.
    attrs = property(
        get_attrs, set_attrs, del_attrs,
        "A property pointing to the attribute set of the file node.")

    def __init__(self):
        super(FileNode, self).__init__()

        # The constructor of the subclass must set the value of
        # the instance attributes 'node', 'mode', 'offset' and '_version'.
        # It also has to set or check the node attributes.
        self.closed = False
        self.sofstpace = 0

        self.node = None
        self.mode = None
        self.offset = None
        self._version = None

    def __del__(self):
        if self.node is not None:
            self.close()

    def _set_attributes(self, node):
        """_set_attributes(node) -> None.  Adds file node-specific attributes.

        Sets the system attributes 'NODE_TYPE' and 'NODE_TYPE_VERSION'
        in the specified PyTables node (leaf).

        """

        attrs = node.attrs
        # System attributes are now writable.  ivb(2004-12-30)
        # attrs._g_setattr('NODE_TYPE', NodeType)
        # attrs._g_setattr('NODE_TYPE_VERSION', NodeTypeVersions[-1])
        attrs.NODE_TYPE = NodeType
        attrs.NODE_TYPE_VERSION = NodeTypeVersions[-1]

    _setAttributes = previous_api(_set_attributes)

    def _check_attributes(self, node):
        """Checks file node-specific attributes.

        Checks for the presence and validity
        of the system attributes 'NODE_TYPE' and 'NODE_TYPE_VERSION'
        in the specified PyTables node (leaf).
        ValueError is raised if an attribute is missing or incorrect.

        """

        attrs = node.attrs
        ltype = getattr(attrs, 'NODE_TYPE', None)
        ltypever = getattr(attrs, 'NODE_TYPE_VERSION', None)

        if ltype != NodeType:
            raise ValueError("invalid type of node object: %s" % (ltype,))
        if ltypever not in NodeTypeVersions:
            raise ValueError(
                "unsupported type version of node object: %s" % (ltypever,))

    _checkAttributes = previous_api(_check_attributes)

    def _check_not_closed(self):
        """Checks if file node is open.

        Checks whether the file node is open or has been closed.
        In the second case, a ValueError is raised.
        If the host PyTables has been closed, ValueError is also raised.

        """

        if self.closed:
            raise ValueError("I/O operation on closed file")
        if getattr(self.node, '_v_file', None) is None:
            raise ValueError("host PyTables file is already closed!")

    _checkNotClosed = previous_api(_check_not_closed)

    def close(self):
        """Flushes the file and closes it.

        After calling this method the node attribute becomes None and
        the attrs property is no longer available.

        """

        # Only flush the first time the file is closed,
        # taking care of not doing it if the host PyTables file
        # has already been closed.
        if not self.closed:
            if getattr(self.node, '_v_file', None) is None:
                warnings.warn("host PyTables file is already closed!")
            else:
                self.flush()

        # Set the flag every time the method is called.
        self.closed = True
        # Release node object to allow closing the file.
        self.node = None

    def flush(self):
        """Flushes the file node.

        See file.flush.__doc__ for more information.

        """

        raise NotImplementedError

    def seek(self, offset, whence=0):
        """Moves to a new file position.

        See file.seek.__doc__ for more information.

        """

        self._check_not_closed()

        if whence == 0:
            # Absolute positioning.
            newoffset = offset
        elif whence == 1:
            # Offset from pointer positioning.
            newoffset = self.offset + offset
        elif whence == 2:
            # Offset from (real) end positioning.
            newoffset = self.node.nrows + offset
        else:
            raise ValueError("invalid positioning mode")

        if newoffset < 0:
            # Positioning before the beginning is not allowed.
            raise IOError("can not seek before beginning of file")
        else:
            # Positioning beyond the end is allowed.
            self.offset = newoffset

    def tell(self):
        """Gets the current file position.

        See file.tell.__doc__ for more information.

        """

        self._check_not_closed()
        return self.offset


class ROFileNode(ReadableMixin, NotWritableMixin, FileNode):
    """Creates a new read-only file node.

    Creates a new read-only file node associated with the specified
    PyTables node, providing a standard Python file interface to it.
    The node has to have been created on a previous occasion
    using the new_node() function.

    This constructor is not intended to be used directly.
    Use the open_node() function in read-only mode ('r') instead.

    """

    # Since FileNode provides all methods for read-only access,
    # only the constructor method and failing writing methods are needed.
    def __init__(self, node):
        super(ROFileNode, self).__init__()
        self._check_attributes(node)

        self.node = node
        self.mode = 'r'
        self.offset = 0L
        self._version = node.attrs.NODE_TYPE_VERSION

    def __del__(self):
        super(ROFileNode, self).__del__()

    def flush(self):
        """Flushes the file node.

        See file.flush.__doc__ for more information.

        """

        self._check_not_closed()
        # Do nothing.


class RAFileNode(ReadableMixin, AppendableMixin, FileNode):
    """Creates a new read-write file node.

    The first syntax opens the specified PyTables node, while the
    second one creates a new node in the specified PyTables file.
    In the second case, additional named arguments 'where' and 'name'
    must be passed to specify where the file node is to be created.
    Other named arguments such as 'title' and 'filters' may also be
    passed.  The special named argument 'expectedsize', indicating an
    estimate of the file size in bytes, may also be passed.

    Write access means reading as well as appending data is allowed.

    This constructor is not intended to be used directly.
    Use the new_node() or open_node() functions instead.

    """

    __allowed_init_kwargs = [
        'where', 'name', 'title', 'filters', 'expectedsize']

    def __init__(self, node, h5file, **kwargs):
        super(RAFileNode, self).__init__()

        if node is not None:
            # Open an existing node and get its version.
            self._check_attributes(node)
            self._version = node.attrs.NODE_TYPE_VERSION
        elif h5file is not None:
            # Check for allowed keyword arguments,
            # to avoid unwanted arguments falling through to array constructor.
            for kwarg in kwargs:
                if kwarg not in self.__allowed_init_kwargs:
                    raise TypeError(
                        "%s keyword argument is not allowed" % repr(kwarg))

            # Turn 'expectedsize' into 'expectedrows'.
            if 'expectedsize' in kwargs:
                # These match since one byte is stored per row.
                expectedrows = kwargs['expectedsize']
                kwargs = kwargs.copy()
                del kwargs['expectedsize']
                kwargs['expectedrows'] = expectedrows

            # Create a new array in the specified PyTables file.
            self._version = NodeTypeVersions[-1]
            shape = self._byteShape[self._version]
            node = h5file.create_earray(
                atom=tables.UInt8Atom(), shape=shape, **kwargs)

            # Set the node attributes, else remove the array itself.
            try:
                self._set_attributes(node)
            except RuntimeError:
                h5file.remove_node(kwargs['where'], kwargs['name'])
                raise

        # Set required attributes (besides of '_version').
        self.node = node
        self.mode = 'a+'
        self.offset = 0L

        # Cache some dictionary lookups regarding file version.
        # self._version is a NumPy scalar and when Python < 2.5
        # this cannot be used as an index.
        # Will force a conversion to an integer.
        version = int(self._version)
        self._vType = tables.UInt8Atom().dtype.base.type
        self._vShape = self._sizeToShape[version]

    def __del__(self):
        super(RAFileNode, self).__del__()

    def flush(self):
        """Flushes the file node.

        See file.flush.__doc__ for more information.

        """

        self._check_not_closed()
        self.node.flush()


## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## End:
