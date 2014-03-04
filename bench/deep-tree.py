# Small benchmark for compare creation times with parameter
# PYTABLES_SYS_ATTRS active or not.

from __future__ import print_function
import os
import subprocess
from time import time
import random
#import numpy
import tables

random.seed(2)


def show_stats(explain, tref):
    "Show the used memory (only works for Linux 2.6.x)."
    # Build the command to obtain memory info
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
    print("Memory usage: ******* %s *******" % explain)
    print("VmSize: %7s kB\tVmRSS: %7s kB" % (vmsize, vmrss))
    print("VmData: %7s kB\tVmStk: %7s kB" % (vmdata, vmstk))
    print("VmExe:  %7s kB\tVmLib: %7s kB" % (vmexe, vmlib))
    tnow = time()
    print("WallClock time:", round(tnow - tref, 3))
    return tnow


def populate(f, nlevels):
    g = f.root
    #arr = numpy.zeros((10,), "f4")
    #descr = {'f0': tables.Int32Col(), 'f1': tables.Float32Col()}
    for i in range(nlevels):
        #dset = f.create_array(g, "DS1", arr)
        #dset = f.create_array(g, "DS2", arr)
        f.create_carray(g, "DS1", tables.IntAtom(), (10,))
        f.create_carray(g, "DS2", tables.IntAtom(), (10,))
        #dset = f.create_table(g, "DS1", descr)
        #dset = f.create_table(g, "DS2", descr)
        f.create_group(g, 'group2_')
        g = f.create_group(g, 'group')


def getnode(f, nlevels, niter, range_):
    for i in range(niter):
        nlevel = random.randrange(
            (nlevels - range_) / 2, (nlevels + range_) / 2)
        groupname = ""
        for i in range(nlevel):
            groupname += "/group"
        groupname += "/DS1"
        f.get_node(groupname)


if __name__ == '__main__':
    nlevels = 1024
    niter = 256
    range_ = 128
    nodeCacheSlots = 64
    pytables_sys_attrs = True
    profile = True
    doprofile = True
    verbose = False

    if doprofile:
        import pstats
        import cProfile as prof

    if profile:
        tref = time()
    if profile:
        show_stats("Abans de crear...", tref)
    f = tables.open_file("/tmp/PTdeep-tree.h5", 'w',
                         node_cache_slots=nodeCacheSlots,
                         pytables_sys_attrs=pytables_sys_attrs)
    if doprofile:
        prof.run('populate(f, nlevels)', 'populate.prof')
        stats = pstats.Stats('populate.prof')
        stats.strip_dirs()
        stats.sort_stats('time', 'calls')
        if verbose:
            stats.print_stats()
        else:
            stats.print_stats(20)
    else:
        populate(f, nlevels)
    f.close()
    if profile:
        show_stats("Despres de crear", tref)

    if profile:
        tref = time()
    if profile:
        show_stats("Abans d'obrir...", tref)
    f = tables.open_file("/tmp/PTdeep-tree.h5", 'r',
                         node_cache_slots=nodeCacheSlots,
                         pytables_sys_attrs=pytables_sys_attrs)
    if profile:
        show_stats("Abans d'accedir...", tref)
    if doprofile:
        prof.run('getnode(f, nlevels, niter, range_)', 'getnode.prof')
        stats = pstats.Stats('getnode.prof')
        stats.strip_dirs()
        stats.sort_stats('time', 'calls')
        if verbose:
            stats.print_stats()
        else:
            stats.print_stats(20)
    else:
        getnode(f, nlevels, niter, range_)
    if profile:
        show_stats("Despres d'accedir", tref)
    f.close()
    if profile:
        show_stats("Despres de tancar", tref)
