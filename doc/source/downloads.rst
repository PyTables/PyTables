=========
Downloads
=========

Stable Versions
---------------

The stable versions of PyTables can be downloaded from the file `download
area`_ on SourceForge.net.  The full distribution contains a copy of this
documentation in HTML.  The documentation in both HTML and PDF formats can
also be downloaded separately from the same URL.

A *pure source* version of the package (mainly intended for developers and
packagers) is available on the `tags page`_ on GitHub.  It contains all files
under SCM but not the (generated) files, HTML doc and *cythonized* C
extensions, so it is smaller that the standard package (about 3.5MB).

Windows binaries can be obtained from many different distributions, like
`Python(x,y)`_, ActiveState_, or Enthought_.
In addition, Christoph Gohlke normally does an excellent job by providing
binaries for many interesting software on his
`website <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_.

You may be interested to install the latest released stable version::

    $ pip install tables

Or, you may prefer to install the stable version in Git repository
using :program:`pip`. For example, for the stable 3.1 series, you can do::

    $ pip install --install-option='--prefix=<PREFIX>' \
    -e git+https://github.com/PyTables/PyTables.git@v.3.1#egg=tables

.. _`download area`: http://sourceforge.net/projects/pytables/files/pytables
.. _`tags page`: https://github.com/PyTables/PyTables/tags
.. _`Python(x,y)`: http://code.google.com/p/pythonxy
.. _ActiveState: http://www.activestate.com/activepython
.. _Enthought: https://www.enthought.com/products/epd


Bleeding Edge Versions
----------------------

The latest, coolest, and possibly buggiest ;-) sources can be obtained from
the new github repository:

https://github.com/PyTables/PyTables

A `snapshot <https://github.com/PyTables/PyTables/archive/develop.zip>`_ of
the code in development is also available on the `GitHub project page`_.

.. _`GitHub project page`: https://github.com/PyTables/PyTables

