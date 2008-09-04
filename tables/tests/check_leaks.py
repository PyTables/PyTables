import os, popen2, time
import tables

tref = time.time()
trel = tref

def show_mem(explain):
    global tref, trel

    cmd = "cat /proc/%s/status" % os.getpid()
    sout, sin = popen2.popen2(cmd)
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
    sin.close()
    print "\nMemory usage: ******* %s *******" % explain
    print "VmSize: %7s kB\tVmRSS: %7s kB" % (vmsize, vmrss)
    print "VmData: %7s kB\tVmStk: %7s kB" % (vmdata, vmstk)
    print "VmExe:  %7s kB\tVmLib: %7s kB" % (vmexe, vmlib)
    print "WallClock time:", time.time() - tref,
    print "  Delta time:", time.time() - trel
    trel = time.time()


def write_group(file, nchildren, niter):
    for i in range(niter):
        fileh = tables.openFile(file, mode = "w")
        for child in range(nchildren):
            fileh.createGroup(fileh.root, 'group' + str(child),
                              "child: %d" % child)
        show_mem("After creating. Iter %s" % i)
        fileh.close()
        show_mem("After close")


def read_group(file, nchildren, niter):
    for i in range(niter):
        fileh = tables.openFile(file, mode = "r")
        for child in range(nchildren):
            node = fileh.getNode(fileh.root, 'group' + str(child))
            flavor = node._v_attrs.CLASS
#         for child in fileh.walkNodes():
#             pass
        show_mem("After reading metadata. Iter %s" % i)
        fileh.close()
        show_mem("After close")


def write_array(file, nchildren, niter):
    for i in range(niter):
        fileh = tables.openFile(file, mode = "w")
        for child in range(nchildren):
            fileh.createArray(fileh.root, 'array' + str(child),
                              [1,1], "child: %d" % child)
        show_mem("After creating. Iter %s" % i)
        fileh.close()
        show_mem("After close")


def read_array(file, nchildren, niter):
    for i in range(niter):
        fileh = tables.openFile(file, mode = "r")
        for child in range(nchildren):
            node = fileh.getNode(fileh.root, 'array' + str(child))
            flavor = node._v_attrs.FLAVOR
            data = node[:]  # Read data
        show_mem("After reading data. Iter %s" % i)
#         for child in range(nchildren):
#             node = fileh.getNode(fileh.root, 'array' + str(child))
#             flavor = node._v_attrs.FLAVOR
            #flavor = node._v_attrs
#         for child in fileh.walkNodes():
#             pass
#         show_mem("After reading metadata. Iter %s" % i)
        fileh.close()
        show_mem("After close")


def write_carray(file, nchildren, niter):
    for i in range(niter):
        fileh = tables.openFile(file, mode = "w")
        for child in range(nchildren):
            fileh.createCArray(fileh.root, 'array' + str(child),
                               tables.IntAtom(), (2,), "child: %d" % child)
        show_mem("After creating. Iter %s" % i)
        fileh.close()
        show_mem("After close")


def read_carray(file, nchildren, niter):
    for i in range(niter):
        fileh = tables.openFile(file, mode = "r")
        for child in range(nchildren):
            node = fileh.getNode(fileh.root, 'array' + str(child))
            flavor = node._v_attrs.FLAVOR
            data = node[:]  # Read data
            #print "data-->", data
        show_mem("After reading data. Iter %s" % i)
        fileh.close()
        show_mem("After close")

def write_earray(file, nchildren, niter):
    for i in range(niter):
        fileh = tables.openFile(file, mode = "w")
        for child in range(nchildren):
            ea = fileh.createEArray(fileh.root, 'array' + str(child),
                                    tables.IntAtom(), shape=(0,),
                                    title="child: %d" % child)
            ea.append([1,2,3])
        show_mem("After creating. Iter %s" % i)
        fileh.close()
        show_mem("After close")


def read_earray(file, nchildren, niter):
    for i in range(niter):
        fileh = tables.openFile(file, mode = "r")
        for child in range(nchildren):
            node = fileh.getNode(fileh.root, 'array' + str(child))
            flavor = node._v_attrs.FLAVOR
            data = node[:]  # Read data
            #print "data-->", data
        show_mem("After reading data. Iter %s" % i)
        fileh.close()
        show_mem("After close")

def write_vlarray(file, nchildren, niter):
    for i in range(niter):
        fileh = tables.openFile(file, mode = "w")
        for child in range(nchildren):
            vl = fileh.createVLArray(fileh.root, 'array' + str(child),
                                     tables.IntAtom(), "child: %d" % child)
            vl.append([1,2,3])
        show_mem("After creating. Iter %s" % i)
        fileh.close()
        show_mem("After close")


def read_vlarray(file, nchildren, niter):
    for i in range(niter):
        fileh = tables.openFile(file, mode = "r")
        for child in range(nchildren):
            node = fileh.getNode(fileh.root, 'array' + str(child))
            flavor = node._v_attrs.FLAVOR
            data = node[:]  # Read data
            #print "data-->", data
        show_mem("After reading data. Iter %s" % i)
        fileh.close()
        show_mem("After close")

def write_table(file, nchildren, niter):

    class Record(tables.IsDescription):
        var1 = tables.IntCol(pos=1)
        var2 = tables.StringCol(length=1, pos=2)
        var3 = tables.FloatCol(pos=3)

    for i in range(niter):
        fileh = tables.openFile(file, mode = "w")
        for child in range(nchildren):
            t = fileh.createTable(fileh.root, 'table' + str(child),
                                  Record, "child: %d" % child)
            t.append([[1,"2",3.]])
        show_mem("After creating. Iter %s" % i)
        fileh.close()
        show_mem("After close")


def read_table(file, nchildren, niter):
    for i in range(niter):
        fileh = tables.openFile(file, mode = "r")
        for child in range(nchildren):
            node = fileh.getNode(fileh.root, 'table' + str(child))
            klass = node._v_attrs.CLASS
            data = node[:]  # Read data
            #print "data-->", data
        show_mem("After reading data. Iter %s" % i)
        fileh.close()
        show_mem("After close")


def write_xtable(file, nchildren, niter):

    class Record(tables.IsDescription):
        var1 = tables.IntCol(pos=1)
        var2 = tables.StringCol(length=1, pos=2)
        var3 = tables.FloatCol(pos=3)

    for i in range(niter):
        fileh = tables.openFile(file, mode = "w")
        for child in range(nchildren):
            t = fileh.createTable(fileh.root, 'table' + str(child),
                                  Record, "child: %d" % child)
            t.append([[1,"2",3.]])
            t.cols.var1.createIndex()
        show_mem("After creating. Iter %s" % i)
        fileh.close()
        show_mem("After close")


def read_xtable(file, nchildren, niter):
    for i in range(niter):
        fileh = tables.openFile(file, mode = "r")
        for child in range(nchildren):
            node = fileh.getNode(fileh.root, 'table' + str(child))
            #klass = node._v_attrs.CLASS
            #data = node[:]  # Read data
            #print "data-->", data
        show_mem("After reading data. Iter %s" % i)
        fileh.close()
        show_mem("After close")


if __name__ == '__main__':
    import sys, getopt, pstats
    import profile as prof

    usage = """usage: %s [-v] [-p] [-a] [-c] [-e] [-l] [-t] [-x] [-g] [-r] [-w] [-c nchildren] [-n iter] file
            -v verbose
            -p profile
            -a create/read arrays  (default)
            -c create/read carrays
            -e create/read earrays
            -l create/read vlrrays
            -t create/read tables
            -x create/read indexed tables
            -g create/read groups
            -r only read test
            -w only write test
            -n number of children (4000 is the default)
            -i number of iterations (default is 3)
            \n"""
    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'vpaceltxgrwn:i:')
    except:
        sys.stderr.write(usage)
        sys.exit(0)

    # if we pass too much parameters, abort
    if len(pargs) != 1:
        sys.stderr.write(usage)
        sys.exit(0)

    # default options
    verbose = 0
    profile = 0
    array = 1
    carray = 0
    earray = 0
    vlarray = 0
    table = 0
    xtable = 0
    group = 0
    write = 0
    read = 0
    nchildren = 1000
    niter = 5

    # Get the options
    for option in opts:
        if option[0] == '-v':
            verbose = 1
        elif option[0] == '-p':
            profile = 1
        elif option[0] == '-a':
            carray = 1
        elif option[0] == '-c':
            array = 0
            carray = 1
        elif option[0] == '-e':
            array = 0
            earray = 1
        elif option[0] == '-l':
            array = 0
            vlarray = 1
        elif option[0] == '-t':
            array = 0
            table = 1
        elif option[0] == '-x':
            array = 0
            xtable = 1
        elif option[0] == '-g':
            array = 0
            cgroup = 1
        elif option[0] == '-w':
            write = 1
        elif option[0] == '-r':
            read = 1
        elif option[0] == '-n':
            nchildren = int(option[1])
        elif option[0] == '-i':
            niter = int(option[1])

    # Catch the hdf5 file passed as the last argument
    file = pargs[0]

    if array:
        fwrite = 'write_array'
        fread = 'read_array'
    elif carray:
        fwrite = 'write_carray'
        fread = 'read_carray'
    elif earray:
        fwrite = 'write_earray'
        fread = 'read_earray'
    elif vlarray:
        fwrite = 'write_vlarray'
        fread = 'read_vlarray'
    elif table:
        fwrite = 'write_table'
        fread = 'read_table'
    elif xtable:
        fwrite = 'write_xtable'
        fread = 'read_xtable'
    elif group:
        fwrite = 'write_group'
        fread = 'read_group'

    show_mem("Before open")
    if write:
        if profile:
            prof.run(str(fwrite)+'(file, nchildren, niter)', 'write_file.prof')
            stats = pstats.Stats('write_file.prof')
            stats.strip_dirs()
            stats.sort_stats('time', 'calls')
            if verbose:
                stats.print_stats()
            else:
                stats.print_stats(20)
        else:
            eval(fwrite+'(file, nchildren, niter)')
    if read:
        if profile:
            prof.run(fread+'(file, nchildren, niter)', 'read_file.prof')
            stats = pstats.Stats('read_file.prof')
            stats.strip_dirs()
            stats.sort_stats('time', 'calls')
            if verbose:
                stats.print_stats()
            else:
                stats.print_stats(20)
        else:
            eval(fread+'(file, nchildren, niter)')
