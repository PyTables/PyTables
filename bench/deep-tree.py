# Small benchmark for compare creation times with parameter
# PYTABLES_SYS_ATTRS active or not.

import os, subprocess, gc
from time import time
import tables

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
    print "Memory usage: ******* %s *******" % explain
    print "VmSize: %7s kB\tVmRSS: %7s kB" % (vmsize, vmrss)
    print "VmData: %7s kB\tVmStk: %7s kB" % (vmdata, vmstk)
    print "VmExe:  %7s kB\tVmLib: %7s kB" % (vmexe, vmlib)
    tnow = time()
    print "WallClock time:", round(tnow - tref, 3)
    return tnow


if __name__=='__main__':
    nlevels = 2048
    profile=True

    if profile: tref = time()
    if profile: show_stats("Without PYTABLES_SYS_ATTRS...", tref)
    f = tables.openFile("/tmp/PTdeep-tree.h5", 'w', PYTABLES_SYS_ATTRS=False)
    g = f.root
    for i in range(nlevels):
        dset = f.createArray(g, "DS1", (10,10))
        dset = f.createArray(g, "DS2", (10,10))
        group2 = f.createGroup(g, 'group2_')
        g = f.createGroup(g, 'group')
    f.close()
    if profile: show_stats("After WITHOUT PYTABLES_SYS_ATTRS", tref)

    if profile: tref = time()
    if profile: show_stats("With PYTABLES_SYS_ATTRS...", tref)
    f = tables.openFile("/tmp/PTdeep-tree.h5", 'w', PYTABLES_SYS_ATTRS=True)
    g = f.root
    for i in range(nlevels):
        dset = f.createArray(g, "DS1", (10,10))
        dset = f.createArray(g, "DS2", (10,10))
        group2 = f.createGroup(g, 'group2_')
        g = f.createGroup(g, 'group')
    f.close()
    if profile: show_stats("After WITH PYTABLES_SYS_ATTRS", tref)
