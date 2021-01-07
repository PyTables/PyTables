"""Testbed for open/close PyTables files.

This uses the HotShot profiler.

"""

import os
import sys
import getopt
import pstats
import cProfile as prof
from pathlib import Path
from time import perf_counter as clock

import tables as tb

filename = None
niter = 1


def show_stats(explain, tref):
    "Show the used memory"
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith("VmSize:"):
            vmsize = int(line.split()[1])
        elif line.startswith("VmRSS:"):
            vmrss = int(line.split()[1])
        elif line.startswith("VmData:"):
            vmdata = int(line.split()[1])
        elif line.startswith("VmStk:"):
            vmstk = int(line.split()[1])
        elif line.startswith("VmExe:"):
            vmexe = int(line.split()[1])
        elif line.startswith("VmLib:"):
            vmlib = int(line.split()[1])
    print("WallClock time:", clock() - tref)
    print("Memory usage: ******* %s *******" % explain)
    print(f"VmSize: {vmsize:>7} kB\tVmRSS: {vmrss:>7} kB")
    print(f"VmData: {vmdata:>7} kB\tVmStk: {vmstk:>7} kB")
    print(f"VmExe:  {vmexe:>7} kB\tVmLib: {vmlib:>7} kB")


def check_open_close():
    for i in range(niter):
        print(
            "------------------ open_close #%s -------------------------" % i)
        tref = clock()
        fileh = tb.open_file(filename)
        fileh.close()
        show_stats("After closing file", tref)


def check_only_open():
    for i in range(niter):
        print("------------------ only_open #%s -------------------------" % i)
        tref = clock()
        fileh = tb.open_file(filename)
        show_stats("Before closing file", tref)
        fileh.close()


def check_full_browse():
    for i in range(niter):
        print("------------------ full_browse #%s -----------------------" % i)
        tref = clock()
        fileh = tb.open_file(filename)
        for node in fileh:
            pass
        fileh.close()
        show_stats("After full browse", tref)


def check_partial_browse():
    for i in range(niter):
        print("------------------ partial_browse #%s --------------------" % i)
        tref = clock()
        fileh = tb.open_file(filename)
        for node in fileh.root.ngroup0.ngroup1:
            pass
        fileh.close()
        show_stats("After closing file", tref)


def check_full_browse_attrs():
    for i in range(niter):
        print("------------------ full_browse_attrs #%s -----------------" % i)
        tref = clock()
        fileh = tb.open_file(filename)
        for node in fileh:
            # Access to an attribute
            klass = node._v_attrs.CLASS
        fileh.close()
        show_stats("After full browse", tref)


def check_partial_browse_attrs():
    for i in range(niter):
        print("------------------ partial_browse_attrs #%s --------------" % i)
        tref = clock()
        fileh = tb.open_file(filename)
        for node in fileh.root.ngroup0.ngroup1:
            # Access to an attribute
            klass = node._v_attrs.CLASS
        fileh.close()
        show_stats("After closing file", tref)


def check_open_group():
    for i in range(niter):
        print("------------------ open_group #%s ------------------------" % i)
        tref = clock()
        fileh = tb.open_file(filename)
        group = fileh.root.ngroup0.ngroup1
        # Access to an attribute
        klass = group._v_attrs.CLASS
        fileh.close()
        show_stats("After closing file", tref)


def check_open_leaf():
    for i in range(niter):
        print("------------------ open_leaf #%s -----------------------" % i)
        tref = clock()
        fileh = tb.open_file(filename)
        leaf = fileh.root.ngroup0.ngroup1.array9
        # Access to an attribute
        klass = leaf._v_attrs.CLASS
        fileh.close()
        show_stats("After closing file", tref)


if __name__ == '__main__':

    usage = """usage: %s [-v] [-p] [-n niter] [-O] [-o] [-B] [-b] [-g] [-l] [-A] [-a] [-E] [-S] datafile
              -v verbose  (total dump of profiling)
              -p do profiling
              -n number of iterations for reading
              -O Check open_close
              -o Check only_open
              -B Check full browse
              -b Check partial browse
              -A Check full browse and reading one attr each node
              -a Check partial browse and reading one attr each node
              -g Check open nested group
              -l Check open nested leaf
              -E Check everything
              -S Check everything as subprocess
              \n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'vpn:OoBbAaglESs')
    except:
        sys.stderr.write(usage)
        sys.exit(0)

    progname = sys.argv[0]
    args = sys.argv[1:]

    # if we pass too much parameters, abort
    if len(pargs) != 1:
        sys.stderr.write(usage)
        sys.exit(0)

    # default options
    verbose = 0
    silent = 0  # if silent, does not print the final statistics
    profile = 0
    all_checks = 0
    all_system_checks = 0
    func = []

    # Checking options
    options = ['-O', '-o', '-B', '-b', '-A', '-a', '-g', '-l']

    # Dict to map options to checking functions
    option2func = {
        '-O': 'check_open_close',
        '-o': 'check_only_open',
        '-B': 'check_full_browse',
        '-b': 'check_partial_browse',
        '-A': 'check_full_browse_attrs',
        '-a': 'check_partial_browse_attrs',
        '-g': 'check_open_group',
        '-l': 'check_open_leaf',
    }

    # Get the options
    for option in opts:
        if option[0] == '-v':
            verbose = 1
        elif option[0] == '-p':
            profile = 1
        elif option[0] in option2func:
            func.append(option2func[option[0]])
        elif option[0] == '-E':
            all_checks = 1
            for opt in options:
                func.append(option2func[opt])
        elif option[0] == '-S':
            all_system_checks = 1
        elif option[0] == '-s':
            silent = 1
        elif option[0] == '-n':
            niter = int(option[1])

    filename = pargs[0]

    tref = clock()
    if all_system_checks:
        args.remove('-S')  # We don't want -S in the options list again
        for opt in options:
            opts = r"{} \-s {} {}".format(progname, opt, " ".join(args))
            # print "opts-->", opts
            os.system("python2.4 %s" % opts)
    else:
        if profile:
            for ifunc in func:
                prof.run(ifunc + '()', ifunc + '.prof')
                stats = pstats.Stats(ifunc + '.prof')
                stats.strip_dirs()
                stats.sort_stats('time', 'calls')
                if verbose:
                    stats.print_stats()
                else:
                    stats.print_stats(20)
        else:
            for ifunc in func:
                eval(ifunc + '()')

    if not silent:
        print("------------------ End of run -------------------------")
        show_stats("Final statistics (after closing everything)", tref)
