"""Testbed for open/close PyTables files. This uses the HotShot profiler."""

import sys, hotshot, hotshot.stats, getopt
import tables

filename = None
niter = 1

def check_open_close():
    #global filename, niter
    for i in range(niter):
	fileh=tables.openFile(filename)
	fileh.close()

if __name__ == '__main__':

    usage = """usage: %s [-v] [-p] [-n niter] datafile
              -v verbose  (total dump of profiling)
	      -p do profiling
	      -n number of iterations for reading\n""" % sys.argv[0]

    try:
	opts, pargs = getopt.getopt(sys.argv[1:], 'vpn:')
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

    # Get the options
    for option in opts:
	if option[0] == '-v':
	    verbose = 1
	elif option[0] == '-p':
	    profile = 1
	elif option[0] == '-n':
	    niter = int(option[1])

    filename = pargs[0]

    if profile:
	prof = hotshot.Profile("bench-open.prof")
	prof.runcall(check_open_close)
	prof.close()
	stats = hotshot.stats.load("bench-open.prof")
	stats.strip_dirs()
	stats.sort_stats('time', 'calls')
	if verbose:
	    stats.print_stats()
	else:
	    stats.print_stats(20)
    else:
	check_open_close()

