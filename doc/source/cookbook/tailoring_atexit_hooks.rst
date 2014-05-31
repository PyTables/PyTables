:source: http://www.pytables.org/moin/UserDocuments/AtexitHooks
:revision: 2
:date: 2012-06-01 09:31:14
:author: AskJakobsen

========================
Tailoring `atexit` hooks
========================

In some situations you may want to tailor the typical messages that PyTables
outputs::

    Closing remaining open files: /tmp/prova.h5... done

The responsible of this behaviour is the :func:`tables.file.close_open_files`
function that is being registered via :func:`atexit.register` Python function.
Although you can't de-register already registered cleanup functions, you can
register new ones to tailor the existing behaviour.
For example, if you  register this function::

    def my_close_open_files(verbose):
        open_files = tables.file._open_files

        are_open_files = len(open_files) > 0

        if verbose and are_open_files:
            sys.stderr.write("Closing remaining open files:")

        if StrictVersion(tables.__version__) >= StrictVersion("3.1.0"):
            # make a copy of the open_files.handlers container for the iteration
            handlers = list(open_files.handlers)
        else:
            # for older versions of pytables, setup the handlers list from the
            # keys
            keys = open_files.keys()
            handlers = []
            for key in keys:
                handlers.append(open_files[key])

        for fileh in handlers:
            if verbose:
                sys.stderr.write("%s..." % fileh.filename)

            fileh.close()

            if verbose:
                sys.stderr.write("done")

        if verbose and are_open_files:
            sys.stderr.write("\n")

    import sys, atexit
    from distutils.version import StrictVersion
    atexit.register(my_close_open_files, False)

then, you won't get the closing messages anymore because the new registered
function is executed before the existing one.
If you want the messages back again, just set the verbose parameter to true.

You can also use the `atexit` hooks to perform other cleanup functions as well.

