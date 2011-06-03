########################################################################
#
#       License: BSD
#       Created: November 25, 2009
#       Author:  Francesc Alted - faltet@pytables.com
#
#       $Id$
#
########################################################################

"""Create links in the HDF5 file.

This module implements containers for soft and external links.  Hard
links doesn't need a container as such as they are the same as regular
nodes (groups or leaves).

Classes:

    SoftLink
    ExternalLink

Functions:


Misc variables:

    __version__

"""
import os
import tables as t
from tables import linkExtension
from tables.node import Node
from tables.utils import lazyattr
from tables.attributeset import AttributeSet
import tables.file

try:
    from tables.linkExtension import ExternalLink
except ImportError:
    are_extlinks_available = False
else:
    are_extlinks_available = True


__version__ = "$Revision$"



def _g_getLinkClass(parent_id, name):
    """Guess the link class."""
    return linkExtension._getLinkClass(parent_id, name)


class Link(Node):
    """Abstract base class for all PyTables links.

    A link is a node that refers to another node.  The `Link` class
    inherits from `Node` class and the links that inherits from `Link`
    are `SoftLink` and `ExternalLink`.  There is not a `HardLink`
    subclass because hard links behave like a regular `Group` or `Leaf`.

    Contrarily to other nodes, links cannot have HDF5 attributes.  This
    is an HDF5 library limitation that might be solved in future
    releases.

    """

    # Properties
    @lazyattr
    def _v_attrs(self):
        """A `NoAttrs` instance replacing the typical `AttributeSet`
        instance of other node objects.  The purpose of `NoAttrs` is to
        make clear that HDF5 attributes are not supported in link nodes.
        """
        class NoAttrs(AttributeSet):
            def __getattr__(self, name):
                raise KeyError("you cannot get attributes from this "
                               "`%s` instance" % self.__class__.__name__)
            def __setattr__(self, name, value):
                raise KeyError("you cannot set attributes to this "
                               "`%s` instance" % self.__class__.__name__)
            def _g_close(self):
                pass
        return NoAttrs(self)


    def __init__(self, parentNode, name, target=None, _log = False):
        self._v_new = target is not None
        self.target = target
        """The path string to the pointed node."""
        super(Link, self).__init__(parentNode, name, _log)


    # Public and tailored versions for copy, move, rename and remove methods

    def copy(self, newparent=None, newname=None,
             overwrite=False, createparents=False):
        """Copy this link and return the new one.

        See `Node._f_copy` for a complete explanation of the arguments.
        Please note that there is no `recursive` flag since links do not
        have child nodes.
        """
        newnode = self._f_copy(newparent=newparent, newname=newname,
                               overwrite=overwrite, createparents=createparents)
        # Insert references to a `newnode` via `newname`
        newnode._v_parent._g_refNode(newnode, newname, True)
        return newnode


    def move(self, newparent=None, newname=None, overwrite=False):
        """Move or rename this link.

        See `Node._f_move` for a complete explanation of the arguments.
        """
        return self._f_move(newparent=newparent, newname=newname,
                            overwrite=overwrite)


    def remove(self):
        """Remove this link from the hierarchy.
        """
        return self._f_remove()


    def rename(self, newname=None, overwrite=False):
        """Rename this link in place.

        See `Node._f_rename` for a complete explanation of the arguments.
        """
        return self._f_rename(newname=newname, overwrite=overwrite)


    def __repr__(self):
        return str(self)



class SoftLink(linkExtension.SoftLink, Link):
    """Represents a soft link (aka symbolic link).

    A soft link is a reference to another node in the *same* file
    hierarchy.  Getting access to the pointed node (this action is
    called *dereferencing*) is done via the `__call__` special method (see
    below).
    """

    # Class identifier.
    _c_classId = 'SOFTLINK'

    def __call__(self):
        """
        Dereference `self.target` and return the object.

        Example of use::

            >>> f=tables.openFile('data/test.h5')
            >>> print f.root.link0
            /link0 (SoftLink) -> /another/path
            >>> print f.root.link0()
            /another/path (Group) ''
        """
        target = self.target
        # Check for relative pathnames
        if not self.target.startswith('/'):
            target = self._v_parent._g_join(self.target)
        return self._v_file._getNode(target)


    def __str__(self):
        """
        Return a short string representation of the link.

        Example of use::

            >>> f=tables.openFile('data/test.h5')
            >>> print f.root.link0
            /link0 (SoftLink) -> /path/to/node
        """

        classname = self.__class__.__name__
        target = self.target
        # Check for relative pathnames
        if not self.target.startswith('/'):
            target = self._v_parent._g_join(self.target)
        if target in self._v_file:
            dangling = ""
        else:
            dangling = " (dangling)"
        return "%s (%s) -> %s%s" % (self._v_pathname, classname,
                                    self.target, dangling)



if are_extlinks_available:

    # Declare this only if the extension is available
    class ExternalLink(linkExtension.ExternalLink, Link):
        """Represents an external link.

        An external link is a reference to a node in *another* file.
        Getting access to the pointed node (this action is called
        *dereferencing*) is done via the `__call__` special method (see
        below).

        .. Warning:: External links are only supported when PyTables is
           compiled against HDF5 1.8.x series.  When using PyTables with
           HDF5 1.6.x, the *parent* group containing external link
           objects will be mapped to an `Unknown` instance and you won't
           be able to access *any* node hanging of this parent group.
           It follows that if the parent group containing the external
           link is the root group, you won't be able to read *any*
           information contained in the file when using HDF5 1.6.x.

        """

        # Class identifier.
        _c_classId = 'EXTERNALLINK'

        def __init__(self, parentNode, name, target=None, _log = False):
            self.extfile = None
            """The external file handler, if the link has been
            dereferenced.  In case the link has not been dereferenced
            yet, its value is None."""
            super(ExternalLink, self).__init__(parentNode, name, target, _log)


        def _get_filename_node(self):
            """Return the external filename and nodepath from `self.target`."""
            # This is needed for avoiding the 'C:\\file.h5' filepath notation
            filename, target = self.target.split(':/')
            return filename, '/'+target


        def __call__(self, **kwargs):
            """
            Dereference `self.target` and return the object.

            You can pass all the arguments (except `filename`, of course)
            supported by the `openFile()` function so as to open the
            referenced external file.

            Example of use::

                >>> f=tables.openFile('data1/test1.h5')
                >>> print f.root.link2
                /link2 (ExternalLink) -> data2/test2.h5:/path/to/node
                >>> plink2 = f.root.link2('a')  # open in 'a'ppend mode
                >>> print plink2
                /path/to/node (Group) ''
                >>> print plink2._v_filename
                'data2/test2.h5'        # belongs to referenced file
            """
            filename, target = self._get_filename_node()

            if not os.path.isabs(filename):
                # Resolve the external link with respect to the this
                # file's directory.  See #306.
                base_directory = os.path.dirname(self._v_file.filename)
                filename = os.path.join(base_directory, filename)

            # Fetch the external file and save a reference to it.
            # Check first in already opened files.
            open_files = tables.file._open_files
            if filename in open_files:
                self.extfile = open_files[filename]
            else:
                self.extfile = t.openFile(filename, **kwargs)
            return self.extfile._getNode(target)


        def umount(self):
            """Safely unmount `self.extfile`, if opened."""
            extfile = self.extfile
            # Close external file, if open
            if extfile is not None and extfile.isopen:
                extfile.close()
                self.extfile = None


        def _f_close(self):
            """Especific close for external links."""
            self.umount()
            super(ExternalLink, self)._f_close()


        def __str__(self):
            """
            Return a short string representation of the link.

            Example of use::

                >>> f=tables.openFile('data1/test1.h5')
                >>> print f.root.link2
                /link2 (ExternalLink) -> data2/test2.h5:/path/to/node
            """

            classname = self.__class__.__name__
            return "%s (%s) -> %s" % (self._v_pathname, classname, self.target)



## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
