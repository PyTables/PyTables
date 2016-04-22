===================================
Welcome to PyTables' documentation!
===================================

PyTables is a package for managing hierarchical datasets and designed
to efficiently and easily cope with extremely large amounts of data.
You can download PyTables and use it for free. You can access documentation,
some examples of use and presentations here.

PyTables is built on top of the HDF5 library, using the Python language
and the NumPy package. It features an object-oriented interface that,
combined with C extensions for the performance-critical parts of the
code (generated using Cython), makes it a fast, yet extremely easy to
use tool for interactively browse, process and search very large amounts
of data. One important feature of PyTables is that it optimizes memory and
disk resources so that data takes much less space (specially if on-flight
compression is used) than other solutions such as relational or object
oriented databases.

You can also find more information by reading the PyTables :doc:`FAQ`.

PyTables development is a continuing effort and we are always looking for
more developers, testers, and users.  If you are interested in being
involved with this project, please contact us via `github`_ or the
`mailing list`_.

.. image:: images/NumFocusSponsoredStamp.png
   :alt: NumFocus Sponsored Stamp
   :align: center
   :width: 300
   :target: http://www.numfocus.org

Since August 2015, PyTables is a `NumFOCUS project`_, which means that
your donations are fiscally sponsored under the NumFOCUS umbrella.  Please
consider donating to NumFOCUS.


--------
Contents
--------

.. toctree::
    :maxdepth: 1

    User’s Guide <usersguide/index>
    Cookbook <cookbook/index>
    FAQ
    other_material
    Migrating from 2.x to 3.x <MIGRATING_TO_3.x>
    downloads
    Release Notes <release_notes>
    project_pointers
    Development <development>
    Development Team <dev_team>


=============
Helpful Links
=============

* :ref:`genindex`
* :ref:`search`


.. _github: https://github.com/PyTables/PyTables
.. _`mailing list`: https://groups.google.com/group/pytables-users
.. _`NumFOCUS project`: http://www.numfocus.org/open-source-projects.html
