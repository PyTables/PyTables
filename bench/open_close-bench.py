"""Testbed for open/close PyTables files. This uses the HotShot profiler."""

import sys, os, getopt, pstats
import profile as prof
import time
import subprocess  # From Python 2.4 on
import tables

filename = None
niter = 1

def show_stats(explain, tref):
    "Show the used memory"
    # Build the command to obtain memory info (only for Linux 2.6.x)
    cmd = "cat /proc/%s/status" % os.getpid()
    sout = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout
    for line in sout:
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
    sout.close()
    print "WallClock time:", time.time() - tref
    print "Memory usage: ******* %s *******" % explain
    print "VmSize: %7s kB\tVmRSS: %7s kB" % (vmsize, vmrss)
    print "VmData: %7s kB\tVmStk: %7s kB" % (vmdata, vmstk)
    print "VmExe:  %7s kB\tVmLib: %7s kB" % (vmexe, vmlib)

def check_open_close():
    for i in range(niter):
        print "------------------ open_close #%s -------------------------" % i
        tref = time.time()
        fileh=tables.openFile(filename)
        fileh.close()
        show_stats("After closing file", tref)

def check_only_open():
    for i in range(niter):
        print "------------------ only_open #%s -------------------------" % i
        tref = time.time()
        fileh=tables.openFile(filename)
        show_stats("Before closing file", tref)
        fileh.close()

def check_full_browse():
    for i in range(niter):
        print "------------------ full_browse #%s -----------------------" % i
        tref = time.time()
        fileh=tables.openFile(filename)
        for node in fileh:
            pass
        fileh.close()
        show_stats("After full browse", tref)

def check_partial_browse():
    for i in range(niter):
        print "------------------ partial_browse #%s --------------------" % i
        tref = time.time()
        fileh=tables.openFile(filename)
        for node in fileh.root.group0.group1:
            pass
        fileh.close()
        show_stats("After closing file", tref)

def check_full_browse_attrs():
    for i in range(niter):
        print "------------------ full_browse_attrs #%s -----------------" % i
        tref = time.time()
        fileh=tables.openFile(filename)
        for node in fileh:
            # Access to an attribute
            klass = node._v_attrs.CLASS
        fileh.close()
        show_stats("After full browse", tref)

def check_partial_browse_attrs():
    for i in range(niter):
        print "------------------ partial_browse_attrs #%s --------------" % i
        tref = time.time()
        fileh=tables.openFile(filename)
        for node in fileh.root.group0.group1:
            # Access to an attribute
            klass = node._v_attrs.CLASS
        fileh.close()
        show_stats("After closing file", tref)

def check_open_group():
    for i in range(niter):
        print "------------------ open_group #%s ------------------------" % i
        tref = time.time()
        fileh=tables.openFile(filename)
        group = fileh.root.group0.group1
        # Access to an attribute
        klass = group._v_attrs.CLASS
        fileh.close()
        show_stats("After closing file", tref)

def check_open_leaf():
    for i in range(niter):
        print "------------------ open_leaf #%s -----------------------" % i
        tref = time.time()
        fileh=tables.openFile(filename)
        leaf = fileh.root.group0.group1.array999
        # Access to an attribute
        klass = leaf._v_attrs.CLASS
        fileh.close()
        show_stats("After closing file", tref)


if __name__ == '__main__':

    usage = """usage: %s [-v] [-p] [-n niter] [-O] [-o] [-B] [-b] [-g] [-l] [-A] [-a] [-E] datafile
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
              \n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'vpn:OoBbAaglE')
    except:
        sys.stderr.write(usage)
        sys.exit(0)

    # if we pass too much parameters, abort
    if len(pargs) <> 1:
        sys.stderr.write(usage)
        sys.exit(0)

    # default options
    verbose = 0
    profile = 0
    open_close = 0
    only_open = 0
    full_browse = 0
    partial_browse = 0
    open_group = 0
    open_leaf = 0
    all_checks = 0
    func = []

    print "opts-->", opts
    # Get the options
    for option in opts:
        if option[0] == '-v':
            verbose = 1
        elif option[0] == '-p':
            profile = 1
        elif option[0] == '-O':
            open_close = 1
            func.append('check_open_close')
        elif option[0] == '-o':
            only_open = 1
            func.append('check_only_open')
        elif option[0] == '-B':
            full_browse = 1
            func.append('check_full_browse')
        elif option[0] == '-b':
            partial_browse = 1
            func.append('check_partial_browse')
        elif option[0] == '-A':
            full_browse = 1
            func.append('check_full_browse_attrs')
        elif option[0] == '-a':
            partial_browse = 1
            func.append('check_partial_browse_attrs')
        elif option[0] == '-g':
            open_group = 1
            func.append('check_open_group')
        elif option[0] == '-l':
            open_leaf = 1
            func.append('check_open_leaf')
        elif option[0] == '-E':
            all_checks = 1
            func.extend(['check_open_close', 'check_only_open',
                         'check_full_browse', 'check_partial_browse',
                         'check_full_browse_attrs', 'check_partial_browse_attrs',
                         'check_open_group', 'check_open_leaf'])
        elif option[0] == '-n':
            niter = int(option[1])

    filename = pargs[0]


    tref = time.time()
    if profile:
        for ifunc in func:
            prof.run(ifunc+'()', ifunc+'.prof')
            stats = pstats.Stats(ifunc+'.prof')
            stats.strip_dirs()
            stats.sort_stats('time', 'calls')
            if verbose:
                stats.print_stats()
            else:
                stats.print_stats(20)
    else:
        for ifunc in func:
            eval(ifunc+'()')

    print "------------------ End of run -------------------------"
    show_stats("Final statistics (after closing everything)", tref)
