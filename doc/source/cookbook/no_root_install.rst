:source: http://www.pytables.org/moin/UserDocuments/InstallingPyTablesWhenNotRoot
:revision: 50
:date: 2008-04-21 11:12:44
:author: localhost


.. todo:: update to use new SW versions


Installing PyTables when you're not root
========================================

By `Koen van de Sande <http://www.tibed.net>`_.

.. warning:: contents of this recipe recipe may be outdated.

This guide describes how to install PyTables and its dependencies on Linux or
other \*nix systems when your user account is not root.
Installing the HDF5_ shared libraries and Python extensions NumArray and
NumPy requires some non-trivial steps to work.
We describe all steps needed.
They only assumption is that you have Python 2.3 or higher and a C/C++ compiler
(gcc) installed.


Installing HDF5
---------------

* First go to or make a temporary folder where we can download and compile
  software.
  We'll assume you're in this temporary folder in the rest of this section.
* Download `hdf5-1.6.5.tar.gz` from ftp://ftp.hdfgroup.org/HDF5/current16/src/::

    wget ftp://ftp.hdfgroup.org/HDF5/current16/src/hdf5-1.6.5.tar.gz

* Extract the archive to the current folder::

    tar xzvf hdf5-1.6.5.tar.gz

* Go to the extracted HDF5 folder::

    cd hdf5-1.6.5

* Run the configure script::

    ./configure

* Run make::

    make install

* We've now compiled HDF5_ into the `hdf5` folder inside the source tree.
  We'll need to move this to its final location.
  For this guide, we'll make a `software` folder inside your home directory
  to store installed libraries::

    mkdir ~/software

* Move the files to the right location::

    mv hdf5 ~/software/


Installing NumArray
-------------------

* From the `NumArray SourceForge page
  <http://sourceforge.net/projects/numpy/files>`_ download
  NumArray 1.5.2 to our temporary folder.
* Extract the archive::

    tar xzvf numarray-1.5.2.tar.gz

* Go to the NumArray folder::

    cd numarray-1.5.2

* Build and install the Python module into our software folder (it will
  actually end up in `~/software/lib/python`::

    python setup.py install --home=~/software

  We will also need to copy the header files of NumArray so PyTables can use
  them later on for compilation.
  Skipping this step will lead to compilation errors for PyTables.
* Go into the header file folder::

    cd include

* Copy the header files. We'll put them together with the HDF5_ header files::

    cp -r numarray ~/software/hdf5/include/


Installing NumPy (optional)
---------------------------

It is not required to install NumPy; PyTables will work with just NumArray
installed.
However, I do recommend that you install NumPy as well, because PyTables
can optionally use it.

* From the `NumPy SourceForge page
  <http://sourceforge.net/projects/numpy/files>`_ download
  NumPy 1.0 (at time of writing) to our temporary folder.
* Extract the archive::

    tar xzvf numpy-1.0.tar.gz

* Go to the NumPy folder::

    cd numpy-1.0
* Build and install the Python module into our software folder::

    python setup.py install --home=~/software


Python wrapper script
---------------------

We've installed all dependencies of PyTables.
We need to create a wrapper script for Python to let PyTables actually find
all these dependencies.
Had we installed them as root, they'd be trivial to find, but now we need to
help a bit.

* Create a script with the following contents (I've called this script `p` on
  my machine)::

    #!/bin/bash
    export PYTHONPATH=~/software/lib/python
    export HDF5_DIR=~/software/hdf5
    export LD_LIBRARY_PATH=~/software/lib/python/tables:~/software/hdf5/lib
    python $*

* Make the script executable::

    chmod 755 p

* Place the script somewhere on your path (for example, inside a folder
  called `bin` inside your home dir, which is normally added to the path
  automatically).
  If you do not add this script to your path, you'll have to replace `p` in
  scripts below by the full path (and name of) your script, e.g.
  `~/pytablespython.sh` if you called it `pytablespython.sh` and put it in
  your home dir.
* Test your Python wrapper script::

    p

* It should now start Python. And you should be able to import `numarray`
  (and optionally `numpy`) without errors::

    Python 2.3.4 (#1, Feb  2 2005, 12:11:53)
    [GCC 3.4.2 20041017 (Red Hat 3.4.2-6.fc3)] on linux2
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import numarray
    >>> import numpy
    >>>


.. note::

    you could do this differently by defining these environment settings
    somewhere in your startup scripts, but this wrapper script approach is
    cleaner.


Installing PyTables
-------------------

* From the `SourceForge page <http://sourceforge.net/projects/pytables/files>`_
  download PyTables 1.3.3 (at time of writing) to our temporary folder.
* Extract the archive::

    tar xzvf pytables-1.3.3.tar.gz

* Go to the PyTables folder::

    cd pytables-1.3.3

* Install PyTables using our wrapper script::

    p setup.py install --home=~/software

* If you get the following error then you are not using the wrapper script
  properly!

  ::

    .. ERROR:: Can't find a local numarray Python installation.
       Please, read carefully the ``README`` file and remember that
       PyTables needs the numarray package to compile and run.}}}


Running Python with PyTables support
------------------------------------

* Use your Python wrapper script to start Python::

    p

* You can now import `tables` without errors::

    Python 2.3.4 (#1, Feb  2 2005, 12:11:53)
    [GCC 3.4.2 20041017 (Red Hat 3.4.2-6.fc3)] on linux2
    Type "help", "copyright", "credits" or "license" for more information.
    >>> import tables
    >>> tables.__version__
    '1.3.3'
    >>>


Concluding remarks
------------------

* It is safe to remove the temporary folder we have used in this guide,
  there are no dependencies on it.
* This guide was written for and tested with HDF5 1.6.5, PyTables 1.3.3 and
  NumArray 1.5.2.


Enjoy working with PyTables!

*Koen*


-----


.. target-notes::

.. _HDF5: http://www.hdfgroup.org/HDF5

