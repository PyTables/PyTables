:source: http://www.pytables.org/moin/UserDocuments/PyTables%20%26%20py2exe
:revision: 8
:date: 2008-04-21 11:12:45
:author: localhost

.. todo:: update the code example to numpy

=============================================================
How to integrate PyTables in your application by using py2exe
=============================================================

This document shortly describes how to build an executable when using PyTables.
Py2exe_ is a third party product that converts python scripts into standalone
windows application/programs.
For more information about py2exe please visit http://www.py2exe.org.

To be able to use py2exe you have to download and install it.
Please follow the instructions at http://www.py2exe.org.

Let’s assume that you have written a python script as in the attachment
:download:`py2exe_howto/pytables_test.py`

.. literalinclude:: py2exe_howto/pytables_test.py
   :linenos:

To wrap this script into an executable you have to create a setup script and a
configuration script in your program directory.

The setup script will look like this::

    from distutils.core import setup
    import py2exe
    setup(console=['pytables_test.py'])

The configuration script (:file:`setup.cfg`) specifies which modules to be
included and excluded::

    [py2exe]
    excludes= Tkconstants,Tkinter,tcl
    includes= encodings.*, tables.*, numarray.*

As you can see I have included everything from tables (tables.*) and numarray
(numarray.*).

Now you are ready to build the executable file (:file:`pytable_test.exe`).
During the build process a subfolder called *dist* will be created.
This folder contains everything needed for your program.
All dependencies (dll's and such stuff) will be copied into this folder.
When you distribute your application you have to distribute all files and
folders inside the *dist* folder.

Below you can see how to start the build process (`python setup.py py2exe`)::

    c:pytables_test> python setup.py py2exe
    ...
    BUILDING EXECUTABLE
    ...

After the build process I enter the *dist* folder and start
:file:`pytables_test.exe`.

::

    c:pytables_test> cd dist

    c:pytables_testdist> pytables_test.exe
    tutorial.h5 (File) 'Test file'
    Last modif.: 'Tue Apr 04 23:09:17 2006'
    Object Tree:
    / (RootGroup) 'Test file'
    /detector (Group) 'Detector information'
    /detector/readout (Table(0,)) 'Readout example'

    [25.0, 36.0, 49.0]

DONE!


-----


.. target-notes::

.. _py2exe: http://www.py2exe.org

