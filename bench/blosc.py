import sys
from pathlib import Path
from time import perf_counter as clock
import numpy as np
import tables as tb


niter = 3
dirname = "/scratch2/faltet/blosc-data/"
#expression = "a**2 + b**3 + 2*a*b + 3"
#expression = "a+b"
#expression = "a**2 + 2*a/b + 3"
#expression = "(a+b)**2 - (a**2 + b**2 + 2*a*b) + 1.1"
expression = "3*a-2*b+1.1"
shuffle = True


def create_file(kind, prec, synth):
    prefix_orig = 'cellzome/cellzome-'
    iname = dirname + prefix_orig + 'none-' + prec + '.h5'
    f = tb.open_file(iname, "r")

    if prec == "single":
        type_ = tb.Float32Atom()
    else:
        type_ = tb.Float64Atom()

    if synth:
        prefix = 'synth/synth-'
    else:
        prefix = 'cellzome/cellzome-'

    for clevel in range(10):
        oname = '%s/%s-%s%d-%s.h5' % (dirname, prefix, kind, clevel, prec)
        # print "creating...", iname
        f2 = tb.open_file(oname, "w")

        if kind in ["none", "numpy"]:
            filters = None
        else:
            filters = tb.Filters(
                complib=kind, complevel=clevel, shuffle=shuffle)

        for name in ['maxarea', 'mascotscore']:
            col = f.get_node('/', name)
            r = f2.create_carray('/', name, type_, col.shape, filters=filters)
            if synth:
                r[:] = np.arange(col.nrows, dtype=type_.dtype)
            else:
                r[:] = col[:]
        f2.close()
        if clevel == 0:
            size = 1.5 * Path(oname).stat().st_size
    f.close()
    return size


def create_synth(kind, prec):

    prefix_orig = 'cellzome/cellzome-'
    iname = dirname + prefix_orig + 'none-' + prec + '.h5'
    f = tb.open_file(iname, "r")

    if prec == "single":
        type_ = tb.Float32Atom()
    else:
        type_ = tb.Float64Atom()

    prefix = 'synth/synth-'
    for clevel in range(10):
        oname = '%s/%s-%s%d-%s.h5' % (dirname, prefix, kind, clevel, prec)
        # print "creating...", iname
        f2 = tb.open_file(oname, "w")

        if kind in ["none", "numpy"]:
            filters = None
        else:
            filters = tb.Filters(
                complib=kind, complevel=clevel, shuffle=shuffle)

        for name in ['maxarea', 'mascotscore']:
            col = f.get_node('/', name)
            r = f2.create_carray('/', name, type_, col.shape, filters=filters)
            if name == 'maxarea':
                r[:] = np.arange(col.nrows, dtype=type_.dtype)
            else:
                r[:] = np.arange(col.nrows, 0, dtype=type_.dtype)

        f2.close()
        if clevel == 0:
            size = 1.5 * Path(oname).stat().st_size
    f.close()
    return size


def process_file(kind, prec, clevel, synth):

    if kind == "numpy":
        lib = "none"
    else:
        lib = kind
    if synth:
        prefix = 'synth/synth-'
    else:
        prefix = 'cellzome/cellzome-'
    iname = '%s/%s-%s%d-%s.h5' % (dirname, prefix, kind, clevel, prec)
    f = tb.open_file(iname, "r")
    a_ = f.root.maxarea
    b_ = f.root.mascotscore

    oname = '%s/%s-%s%d-%s-r.h5' % (dirname, prefix, kind, clevel, prec)
    f2 = tb.open_file(oname, "w")
    if lib == "none":
        filters = None
    else:
        filters = tb.Filters(complib=lib, complevel=clevel, shuffle=shuffle)
    if prec == "single":
        type_ = tb.Float32Atom()
    else:
        type_ = tb.Float64Atom()
    r = f2.create_carray('/', 'r', type_, a_.shape, filters=filters)

    if kind == "numpy":
        a2, b2 = a_[:], b_[:]
        t0 = clock()
        r = eval(expression, {'a': a2, 'b': b2})
        print(f"{clock() - t0:5.2f}")
    else:
        expr = tb.Expr(expression, {'a': a_, 'b': b_})
        expr.set_output(r)
        expr.eval()
    f.close()
    f2.close()
    size = Path(iname).stat().st_size + Path(oname).stat().st_size
    return size


if __name__ == '__main__':
    if len(sys.argv) > 3:
        kind = sys.argv[1]
        prec = sys.argv[2]
        if sys.argv[3] == "synth":
            synth = True
        else:
            synth = False
    else:
        print("3 parameters required")
        sys.exit(1)

    # print "kind, precision, synth:", kind, prec, synth

    # print "Creating input files..."
    size_orig = create_file(kind, prec, synth)

    # print "Processing files for compression levels in range(10)..."
    for clevel in range(10):
        t0 = clock()
        ts = []
        for i in range(niter):
            size = process_file(kind, prec, clevel, synth)
            ts.append(clock() - t0)
            t0 = clock()
        ratio = size_orig / size
        print(f"{min(ts):5.2f}, {ratio:5.2f}")
