"""
convert netCDF file to HDF5 using Scientific.IO.NetCDF and PyTables.
Jeff Whitaker <jeffrey.s.whitaker@noaa.gov>

Added some flags to select filters, as well as some small improvements.
Francesc Alted <faltet@pytables.com>

This requires Scientific from
http://starship.python.net/~hinsen/ScientificPython
"""

import sys, os.path, getopt, time

import tables.netcdf3

from tables.leaf import Filters


def nctoh5(ncfilename, h5filename, filters, verbose, overwritefile):
    # open h5 file
    if overwritefile:
        h5file = tables.netcdf3.NetCDFFile(h5filename, mode = "w")
    else:
        h5file = tables.netcdf3.NetCDFFile(h5filename, mode = "a")
    # convert to netCDF
    nobjects, nbytes = h5file.nctoh5(ncfilename,filters=filters)
    # ncdump-like output
    if verbose:
        print 'contents of hdf5 file:'
        print '----------------------'
        print h5file
    # Close the file
    h5file.close()
    return nobjects, nbytes


def main():
    if not tables.netcdf3.ScientificIONetCDF_imported:
        sys.stderr.write(
            'You need Scientific Python installed in order to use this utility.\n')
        sys.exit(1)

    usage = """usage: %s [-h] [-v] [-o] [--complevel=(0-9)] [--complib=lib] [--shuffle=(0|1)] [--fletcher32=(0|1)] netcdffilename hdf5filename
     -h -- Print usage message.
     -v -- Show more information.
     -o -- Overwite destination file.
     --complevel=(0-9) -- Set a compression level (0 for no compression, which
         is the default).
     --complib=lib -- Set the compression library to be used during the copy.
         lib can be set to "zlib", "lzo" or "ucl". Defaults to "zlib".
     --shuffle=(0|1) -- Activate or not the shuffling filter (default is active
         if complevel>0).
     --fletcher32=(0|1) -- Whether to activate or not the fletcher32 filter (not
         active by default).
    \n""" % os.path.basename(sys.argv[0])

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'hvo',
                                    ['complevel=',
                                     'complib=',
                                     'shuffle=',
                                     'fletcher32=',
                                     ])
    except:
        (type, value, traceback) = sys.exc_info()
        print "Error parsing the options. The error was:", value
        sys.stderr.write(usage)
        sys.exit(0)

    # default options
    verbose = 0
    overwritefile = 0
    complevel = None
    complib = None
    shuffle = 0
    fletcher32 = 0

    # Get the options
    for option in opts:
        if option[0] == '-h':
            sys.stderr.write(usage)
            sys.exit(0)
        elif option[0] == '-v':
            verbose = 1
        elif option[0] == '-o':
            overwritefile = 1
        elif option[0] == '--complevel':
            complevel = int(option[1])
        elif option[0] == '--complib':
            complib = option[1]
        elif option[0] == '--shuffle':
            shuffle = int(option[1])
        elif option[0] == '--fletcher32':
            fletcher32 = int(option[1])
        else:
            print option[0], ": Unrecognized option"
            sys.stderr.write(usage)
            sys.exit(0)

    # if we pass a number of files different from 2, abort
    if len(pargs) <> 2:
        print "You need to pass both source and destination!."
        sys.stderr.write(usage)
        sys.exit(0)

    # Catch the files passed as the last arguments
    ncfilename = pargs[0]
    h5filename = pargs[1]


    # Build the Filters instance
    if complevel==None and complib==None and shuffle==0 and fletcher32==0:
        filters = None
    else:
        if complevel is None: complevel = 0
        if complib is None: complib = "zlib"
        if fletcher32 is None: fletcher32 = 0
        filters = Filters(complevel=complevel, complib=complib,
                          shuffle=shuffle, fletcher32=fletcher32)

    # Some timing
    t1 = time.time()
    cpu1 = time.clock()
    # Copy the file
    if verbose:
        print "+=+"*20
        print "Starting conversion from %s to %s" % (ncfilename, h5filename)
        if filters == None:
            print "Using default filters (complevel=6,complib='zlib',shuffle=1,fletcher32=0)"
        else:
            print "Applying filters:", filters
        print "+=+"*20

    # Do the conversion
    (nobjects, nbytes) = nctoh5(ncfilename, h5filename, filters, verbose, overwritefile)

    # Gather some statistics
    t2 = time.time()
    cpu2 = time.clock()
    tcopy = round(t2-t1, 3)
    cpucopy = round(cpu2-cpu1, 3)
    tpercent = int(round(cpucopy/tcopy, 2)*100)
    if verbose:
        print "Number of variables copied:", nobjects
        print "KBytes copied:", round(nbytes/1024.,3)
        print "Time copying: %s s (real) %s s (cpu)  %s%%" % \
              (tcopy, cpucopy, tpercent)
        print "Copied variable/sec: ", round(nobjects / float(tcopy),1)
        print "Copied KB/s :", int(nbytes / (tcopy * 1024))
