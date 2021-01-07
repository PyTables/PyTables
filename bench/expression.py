from pathlib import Path
from time import perf_counter as clock

import numpy as np
import tables as tb

OUT_DIR = Path("/scratch2/faltet/")   # the directory for data output

shape = (1000, 1000 * 1000)   # shape for input arrays
expr = "a*b+1"   # Expression to be computed

nrows, ncols = shape


def tables(docompute, dowrite, complib, verbose):

    # Filenames
    ifilename = OUT_DIR / "expression-inputs.h5"
    ofilename = OUT_DIR / "expression-outputs.h5"

    # Filters
    shuffle = True
    if complib == 'blosc':
        filters = tb.Filters(complevel=1, complib='blosc', shuffle=shuffle)
    elif complib == 'lzo':
        filters = tb.Filters(complevel=1, complib='lzo', shuffle=shuffle)
    elif complib == 'zlib':
        filters = tb.Filters(complevel=1, complib='zlib', shuffle=shuffle)
    else:
        filters = tb.Filters(complevel=0, shuffle=False)
    if verbose:
        print("Will use filters:", filters)

    if dowrite:
        f = tb.open_file(ifilename, 'w')

        # Build input arrays
        t0 = clock()
        root = f.root
        a = f.create_carray(root, 'a', tb.Float32Atom(),
                            shape, filters=filters)
        b = f.create_carray(root, 'b', tb.Float32Atom(),
                            shape, filters=filters)
        if verbose:
            print("chunkshape:", a.chunkshape)
            print("chunksize:", np.prod(a.chunkshape) * a.dtype.itemsize)
        #row = np.linspace(0, 1, ncols)
        row = np.arange(0, ncols, dtype='float32')
        for i in range(nrows):
            a[i] = row * (i + 1)
            b[i] = row * (i + 1) * 2
        f.close()
        print(f"[tables.Expr] Time for creating inputs: {clock() - t0:.3f}")

    if docompute:
        f = tb.open_file(ifilename, 'r')
        fr = tb.open_file(ofilename, 'w')
        a = f.root.a
        b = f.root.b
        r1 = f.create_carray(fr.root, 'r1', tb.Float32Atom(), shape,
                             filters=filters)
        # The expression
        e = tb.Expr(expr)
        e.set_output(r1)
        t0 = clock()
        e.eval()
        if verbose:
            print("First ten values:", r1[0, :10])
        f.close()
        fr.close()
        print(f"[tables.Expr] Time for computing & save: {clock() - t0:.3f}")


def memmap(docompute, dowrite, verbose):

    afilename = OUT_DIR / "memmap-a.bin"
    bfilename = OUT_DIR / "memmap-b.bin"
    rfilename = OUT_DIR / "memmap-output.bin"
    if dowrite:
        t0 = clock()
        a = np.memmap(afilename, dtype='float32', mode='w+', shape=shape)
        b = np.memmap(bfilename, dtype='float32', mode='w+', shape=shape)

        # Fill arrays a and b
        #row = np.linspace(0, 1, ncols)
        row = np.arange(0, ncols, dtype='float32')
        for i in range(nrows):
            a[i] = row * (i + 1)
            b[i] = row * (i + 1) * 2
        del a, b  # flush data
        print(f"[numpy.memmap] Time for creating inputs: {clock() - t0:.3f}")

    if docompute:
        t0 = clock()
        # Reopen inputs in read-only mode
        a = np.memmap(afilename, dtype='float32', mode='r', shape=shape)
        b = np.memmap(bfilename, dtype='float32', mode='r', shape=shape)
        # Create the array output
        r = np.memmap(rfilename, dtype='float32', mode='w+', shape=shape)
        # Do the computation row by row
        for i in range(nrows):
            r[i] = eval(expr, {'a': a[i], 'b': b[i]})
        if verbose:
            print("First ten values:", r[0, :10])
        del a, b
        del r  # flush output data
        print(f"[numpy.memmap] Time for compute & save: {clock() - t0:.3f}")


def do_bench(what, documpute, dowrite, complib, verbose):
    if what == "tables":
        tables(docompute, dowrite, complib, verbose)
    if what == "memmap":
        memmap(docompute, dowrite, verbose)


if __name__ == "__main__":
    import sys
    import getopt

    usage = """usage: %s [-T] [-M] [-c] [-w] [-v] [-z complib]
           -T use tables.Expr
           -M use numpy.memmap
           -c do the computation only
           -w write inputs only
           -v verbose mode
           -z select compression library ('zlib' or 'lzo').  Default is None.
""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'TMcwvz:')
    except:
        sys.stderr.write(usage)
        sys.exit(1)

    # default options
    usepytables = False
    usememmap = False
    docompute = False
    dowrite = False
    verbose = False
    complib = None

    # Get the options
    for option in opts:
        if option[0] == '-T':
            usepytables = True
        elif option[0] == '-M':
            usememmap = True
        elif option[0] == '-c':
            docompute = True
        elif option[0] == '-w':
            dowrite = True
        elif option[0] == '-v':
            verbose = True
        elif option[0] == '-z':
            complib = option[1]
            if complib not in ('blosc', 'lzo', 'zlib'):
                print("complib must be 'lzo' or 'zlib' "
                       "and you passed: '%s'" % complib)
                sys.exit(1)

    # If not a backend selected, abort
    if not usepytables and not usememmap:
        print("Please select a backend:")
        print("PyTables.Expr: -T")
        print("NumPy.memmap: -M")
        sys.exit(1)

    # Select backend and do the benchmark
    if usepytables:
        what = "tables"
    if usememmap:
        what = "memmap"
    do_bench(what, docompute, dowrite, complib, verbose)
