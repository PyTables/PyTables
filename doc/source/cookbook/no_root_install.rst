:author: localhost
:date: 2008-04-21 11:12:44

.. todo:: update to use new SW versions


Installing PyTables when you're not root
========================================

By `Koen van de Sande <http://www.tibed.net>`_.

.. warning:: contents of this recipe may be outdated.

This guide describes how to install PyTables and its dependencies on Linux or
other \*nix systems when your user account is not root.
Installing the HDF5_ shared libraries and Python extension
NumPy requires some non-trivial steps to work.
We describe all steps needed.
They only assumption is that you have Python 3.6 or higher and a C/C++
compiler (gcc) installed.


Installing HDF5
---------------

* First go to or make a temporary folder where we can download and compile
  software.
  We'll assume you're in this temporary folder in the rest of this section.
* Download `hdf5-1.12.1.tar.gz` from https://www.hdfgroup.org/downloads/hdf5
* Extract the archive to the current folder::

    tar xzvf hdf5-1.12.1.tar.gz

* Go to the extracted HDF5 folder::

    cd hdf5-1.12.1

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


Installing NumPy
----------------

* From the `NumPy page on PyPI <https://pypi.org/project/numpy/>`_
  download NumPy 1.21.5 (at time of writing) to our temporary folder.
* Extract the archive::

    tar xzvf numpy-1.21.5.tar.gz

* Go to the NumPy folder::

    cd numpy-1.21.5
* Build and install the Python module into our software folder::

    python3 setup.py install --home=~/software


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
    python3 $*

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

* It should now start Python. And you should be able to import `numpy`
  without errors::

    >>> import numpy


.. note::

    you could do this differently by defining these environment settings
    somewhere in your startup scripts, but this wrapper script approach is
    cleaner.


Installing PyTables
-------------------

* From the `PyPI page <https://pypi.org/project/tables/>`_
  download PyTables 3.7.0 (at time of writing) to our temporary folder.
* Extract the archive::

    tar xzvf pytables-3.7.0.tar.gz

* Go to the PyTables folder::

    cd pytables-3.7.0

* Install PyTables using our wrapper script::

    p setup.py install --home=~/software


Running Python with PyTables support
------------------------------------

* Use your Python wrapper script to start Python::

    p

* You can now import `tables` without errors::

    >>> import tables
    >>> tables.__version__
    '3.7.0'


Concluding remarks
------------------

* It is safe to remove the temporary folder we have used in this guide,
  there are no dependencies on it.
* This guide was written for and tested with HDF5 1.12.1, PyTables 3.7.6 and
  Numpy 1.21.5.


Enjoy working with PyTables!

*Koen*


-----


.. target-notes::

.. _HDF5: http://www.hdfgroup.org/HDF5
