###########################################################################
# Benchmark for undo/redo. Run this program without parameters
# for mode of use. When using profiling, it is recommended to use it
# with Python 2.4.
#
# Francesc Alted
# 2005-03-09
###########################################################################

import os
import tempfile
import Numeric
from time import time
import tables

verbose = 0

class BasicBenchmark(object):

    def __init__(self, filename, testname, vecsize, nobjects, niter):

        self.file = filename
        self.test = testname
        self.vecsize = vecsize
        self.nobjects = nobjects
        self.niter = niter

        # Initialize the arrays
        self.a1 = Numeric.arange(0,1*self.vecsize)
        self.a2 = Numeric.arange(1*self.vecsize,2*self.vecsize)
        self.a3 = Numeric.arange(2*self.vecsize,3*self.vecsize)

    def setUp(self):

        # Create an HDF5 file
        self.fileh = tables.openFile(self.file, mode = "w")
        # open the do/undo
        self.fileh.enableUndo()

    def tearDown(self):
        self.fileh.disableUndo()
        self.fileh.close()
        # Remove the temporary file
        #os.remove(self.file)

    def createNode(self):
        """Checking a undo/redo createArray"""

        for i in range(self.nobjects):
            # Create a new array
            self.fileh.createArray('/', 'array'+str(i), self.a1)
            # Put a mark
            self.fileh.mark()
        # Unwind all marks sequentially
        for i in range(self.niter):
            t1 = time()
            for i in range(self.nobjects):
                self.fileh.undo()
                if verbose: print "u",
            if verbose: print
            undo = time() - t1
            # Rewind all marks sequentially
            t1 = time()
            for i in range(self.nobjects):
                self.fileh.redo()
                if verbose: print "r",
            if verbose: print
            redo = time() - t1

            print "Time for Undo, Redo (createNode):", undo, "s, ", redo, "s"

    def copyChildren(self):
        """Checking a undo/redo copyChildren"""

        # Create a group
        self.fileh.createGroup('/', 'agroup')
        # Create several objects there
        for i in range(10):
            # Create a new array
            self.fileh.createArray('/agroup', 'array'+str(i), self.a1)
        # Excercise copyChildren
        for i in range(self.nobjects):
            # Create another group for destination
            self.fileh.createGroup('/', 'anothergroup'+str(i))
            # Copy children from /agroup to /anothergroup+i
            self.fileh.copyChildren('/agroup', '/anothergroup'+str(i))
            # Put a mark
            self.fileh.mark()
        # Unwind all marks sequentially
        for i in range(self.niter):
            t1 = time()
            for i in range(self.nobjects):
                self.fileh.undo()
                if verbose: print "u",
            if verbose: print
            undo = time() - t1
            # Rewind all marks sequentially
            t1 = time()
            for i in range(self.nobjects):
                self.fileh.redo()
                if verbose: print "r",
            if verbose: print
            redo = time() - t1

            print "Time for Undo, Redo (copyChildren):", undo, "s, ", redo, "s"


    def setAttr(self):
        """Checking a undo/redo for setting attributes"""

        # Create a new array
        self.fileh.createArray('/', 'array', self.a1)
        for i in range(self.nobjects):
            # Set an attribute
            setattr(self.fileh.root.array.attrs, "attr"+str(i), str(self.a1))
            # Put a mark
            self.fileh.mark()
        # Unwind all marks sequentially
        for i in range(self.niter):
            t1 = time()
            for i in range(self.nobjects):
                self.fileh.undo()
                if verbose: print "u",
            if verbose: print
            undo = time() - t1
            # Rewind all marks sequentially
            t1 = time()
            for i in range(self.nobjects):
                self.fileh.redo()
                if verbose: print "r",
            if verbose: print
            redo = time() - t1

            print "Time for Undo, Redo (setAttr):", undo, "s, ", redo, "s"

    def runall(self):

        if testname == "all":
            tests = [self.createNode,self.copyChildren,self.setAttr]
        elif testname == "createNode":
            tests = [self.createNode]
        elif testname == "copyChildren":
            tests = [self.copyChildren]
        elif testname == "setAttr":
            tests = [self.setAttr]
        for meth in tests:
            self.setUp()
            meth()
            self.tearDown()


if __name__ == '__main__':
    import sys, getopt

    usage = """usage: %s [-v] [-p] [-t test] [-s vecsize] [-n niter] datafile
              -v verbose  (total dump of profiling)
              -p do profiling
              -t {createNode|copyChildren|setAttr|all} run the specified test
              -s the size of vectors that are undone/redone
              -n number of objects in operations
              -i number of iterations for reading\n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'vpt:s:n:i:')
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
    testname = "all"
    vecsize = 10
    nobjects = 1
    niter = 1


    # Get the options
    for option in opts:
        if option[0] == '-v':
            verbose = 1
        elif option[0] == '-p':
            profile = 1
        elif option[0] == '-t':
            testname = option[1]
            if testname not in ['createNode','copyChildren','setAttr','all']:
                sys.stderr.write(usage)
                sys.exit(0)
        elif option[0] == '-s':
            vecsize = int(option[1])
        elif option[0] == '-n':
            nobjects = int(option[1])
        elif option[0] == '-i':
            niter = int(option[1])

    filename = pargs[0]


    bench = BasicBenchmark(filename, testname, vecsize, nobjects, niter)
    if profile:
        import hotshot, hotshot.stats
        prof = hotshot.Profile("do_undo.prof")
        prof.runcall(bench.runall)
        prof.close()
        stats = hotshot.stats.load("do_undo.prof")
        stats.strip_dirs()
        stats.sort_stats('time', 'calls')
        if verbose:
            stats.print_stats()
        else:
            stats.print_stats(20)
    else:
        bench.runall()

## Local Variables:
## mode: python
## End:
