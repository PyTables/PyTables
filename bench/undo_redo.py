"""Benchmark for undo/redo.
Run this program without parameters for mode of use."""

from time import perf_counter as clock
import numpy as np
import tables as tb

verbose = 0


class BasicBenchmark:

    def __init__(self, filename, testname, vecsize, nobjects, niter):

        self.file = filename
        self.test = testname
        self.vecsize = vecsize
        self.nobjects = nobjects
        self.niter = niter

        # Initialize the arrays
        self.a1 = np.arange(0, 1 * self.vecsize)
        self.a2 = np.arange(1 * self.vecsize, 2 * self.vecsize)
        self.a3 = np.arange(2 * self.vecsize, 3 * self.vecsize)

    def setUp(self):

        # Create an HDF5 file
        self.fileh = tb.open_file(self.file, mode="w")
        # open the do/undo
        self.fileh.enable_undo()

    def tearDown(self):
        self.fileh.disable_undo()
        self.fileh.close()
        # Remove the temporary file
        # os.remove(self.file)

    def createNode(self):
        """Checking a undo/redo create_array."""

        for i in range(self.nobjects):
            # Create a new array
            self.fileh.create_array('/', 'array' + str(i), self.a1)
            # Put a mark
            self.fileh.mark()
        # Unwind all marks sequentially
        for i in range(self.niter):
            t1 = clock()
            for i in range(self.nobjects):
                self.fileh.undo()
                if verbose:
                    print("u", end=' ')
            if verbose:
                print()
            undo = clock() - t1
            # Rewind all marks sequentially
            t1 = clock()
            for i in range(self.nobjects):
                self.fileh.redo()
                if verbose:
                    print("r", end=' ')
            if verbose:
                print()
            redo = clock() - t1

            print("Time for Undo, Redo (createNode):", undo, "s, ", redo, "s")

    def copy_children(self):
        """Checking a undo/redo copy_children."""

        # Create a group
        self.fileh.create_group('/', 'agroup')
        # Create several objects there
        for i in range(10):
            # Create a new array
            self.fileh.create_array('/agroup', 'array' + str(i), self.a1)
        # Excercise copy_children
        for i in range(self.nobjects):
            # Create another group for destination
            self.fileh.create_group('/', 'anothergroup' + str(i))
            # Copy children from /agroup to /anothergroup+i
            self.fileh.copy_children('/agroup', '/anothergroup' + str(i))
            # Put a mark
            self.fileh.mark()
        # Unwind all marks sequentially
        for i in range(self.niter):
            t1 = clock()
            for i in range(self.nobjects):
                self.fileh.undo()
                if verbose:
                    print("u", end=' ')
            if verbose:
                print()
            undo = clock() - t1
            # Rewind all marks sequentially
            t1 = clock()
            for i in range(self.nobjects):
                self.fileh.redo()
                if verbose:
                    print("r", end=' ')
            if verbose:
                print()
            redo = clock() - t1

            print(("Time for Undo, Redo (copy_children):", undo, "s, ",
                  redo, "s"))

    def set_attr(self):
        """Checking a undo/redo for setting attributes."""

        # Create a new array
        self.fileh.create_array('/', 'array', self.a1)
        for i in range(self.nobjects):
            # Set an attribute
            setattr(self.fileh.root.array.attrs, "attr" + str(i), str(self.a1))
            # Put a mark
            self.fileh.mark()
        # Unwind all marks sequentially
        for i in range(self.niter):
            t1 = clock()
            for i in range(self.nobjects):
                self.fileh.undo()
                if verbose:
                    print("u", end=' ')
            if verbose:
                print()
            undo = clock() - t1
            # Rewind all marks sequentially
            t1 = clock()
            for i in range(self.nobjects):
                self.fileh.redo()
                if verbose:
                    print("r", end=' ')
            if verbose:
                print()
            redo = clock() - t1

            print("Time for Undo, Redo (set_attr):", undo, "s, ", redo, "s")

    def runall(self):

        if testname == "all":
            tests = [self.createNode, self.copy_children, self.set_attr]
        elif testname == "createNode":
            tests = [self.createNode]
        elif testname == "copy_children":
            tests = [self.copy_children]
        elif testname == "set_attr":
            tests = [self.set_attr]
        for meth in tests:
            self.setUp()
            meth()
            self.tearDown()


if __name__ == '__main__':
    import sys
    import getopt

    usage = """usage: %s [-v] [-p] [-t test] [-s vecsize] [-n niter] datafile
              -v verbose  (total dump of profiling)
              -p do profiling
              -t {createNode|copy_children|set_attr|all} run the specified test
              -s the size of vectors that are undone/redone
              -n number of objects in operations
              -i number of iterations for reading\n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'vpt:s:n:i:')
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
            if testname not in ['createNode', 'copy_children', 'set_attr',
                                'all']:
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
        import hotshot
        import hotshot.stats
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

# Local Variables:
# mode: python
# End:
