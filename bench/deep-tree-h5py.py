from __future__ import print_function
import os
import subprocess
from time import time
import random
import numpy
import h5py

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
    g = f
    arr = numpy.zeros((10,), "f4")
    for i in range(nlevels):
        g["DS1"] = arr
        g["DS2"] = arr
        g.create_group('group2_')
        g = g.create_group('group')


def getnode(f, nlevels, niter, range_):
    for i in range(niter):
        nlevel = random.randrange(
            (nlevels - range_) / 2, (nlevels + range_) / 2)
        groupname = ""
        for i in range(nlevel):
            groupname += "/group"
        groupname += "/DS1"
        f[groupname]


if __name__ == '__main__':
    nlevels = 1024
    niter = 1000
    range_ = 256
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
    f = h5py.File("/tmp/deep-tree.h5", 'w')
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

#     if profile: tref = time()
#     if profile: show_stats("Abans d'obrir...", tref)
#     f = h5py.File("/tmp/deep-tree.h5", 'r')
#     if profile: show_stats("Abans d'accedir...", tref)
#     if doprofile:
#         prof.run('getnode(f, nlevels, niter, range_)', 'deep-tree.prof')
#         stats = pstats.Stats('deep-tree.prof')
#         stats.strip_dirs()
#         stats.sort_stats('time', 'calls')
#         if verbose:
#             stats.print_stats()
#         else:
#             stats.print_stats(20)
#     else:
#         getnode(f, nlevels, niter, range_)
#     if profile: show_stats("Despres d'accedir", tref)
#     f.close()
#     if profile: show_stats("Despres de tancar", tref)

#     f = h5py.File("/tmp/deep-tree.h5", 'r')
#     g = f
#     for i in range(nlevels):
#         dset = g["DS1"]
#         dset = g["DS2"]
#         group2 = g['group2_']
#         g = g['group']
#     f.close()
