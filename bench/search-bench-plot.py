import tables as tb
from pylab import *


def get_values(filename, complib=''):
    f = tb.open_file(filename)
    nrows = f.root.small.create_best.cols.nrows[:]
    corrected_sizes = nrows / 10 ** 6
    if mb_units:
        corrected_sizes = 16 * nrows / 10 ** 6
    if insert:
        values = corrected_sizes / f.root.small.create_best.cols.tfill[:]
    if table_size:
        values = f.root.small.create_best.cols.fsize[:] / nrows
    if query:
        values = corrected_sizes / \
            f.root.small.search_best.inkernel.int.cols.time1[:]
    if query_cache:
        values = corrected_sizes / \
            f.root.small.search_best.inkernel.int.cols.time2[:]

    f.close()
    return nrows, values


def show_plot(plots, yaxis, legends, gtitle):
    xlabel('Number of rows')
    ylabel(yaxis)
    xlim(10 ** 3, 10 ** 8)
    title(gtitle)
    grid(True)

#     legends = [f[f.find('-'):f.index('.out')] for f in filenames]
#     legends = [l.replace('-', ' ') for l in legends]
    if table_size:
        legend([p[0] for p in plots], legends, loc="upper right")
    else:
        legend([p[0] for p in plots], legends, loc="upper left")

    #subplots_adjust(bottom=0.2, top=None, wspace=0.2, hspace=0.2)
    if outfile:
        savefig(outfile)
    else:
        show()

if __name__ == '__main__':

    import sys
    import getopt

    usage = """usage: %s [-o file] [-t title] [--insert] [--table-size] [--query] [--query-cache] [--MB-units] files
 -o filename for output (only .png and .jpg extensions supported)
 -t title of the plot
 --insert -- Insert time for table
 --table-size -- Size of table
 --query -- Time for querying the integer column
 --query-cache -- Time for querying the integer (cached)
 --MB-units -- Express speed in MB/s instead of MRows/s
 \n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'o:t:',
                                    ['insert',
                                     'table-size',
                                     'query',
                                     'query-cache',
                                     'MB-units',
                                     ])
    except:
        sys.stderr.write(usage)
        sys.exit(0)

    progname = sys.argv[0]
    args = sys.argv[1:]

    # if we pass too few parameters, abort
    if len(pargs) < 1:
        sys.stderr.write(usage)
        sys.exit(0)

    # default options
    outfile = None
    insert = 0
    table_size = 0
    query = 0
    query_cache = 0
    mb_units = 0
    yaxis = "No axis name"
    tit = None
    gtitle = "Please set a title!"

    # Get the options
    for option in opts:
        if option[0] == '-o':
            outfile = option[1]
        elif option[0] == '-t':
            tit = option[1]
        elif option[0] == '--insert':
            insert = 1
            yaxis = "MRows/s"
            gtitle = "Writing with small (16 bytes) record size"
        elif option[0] == '--table-size':
            table_size = 1
            yaxis = "Bytes/row"
            gtitle = ("Disk space taken by a record (original record size: "
                      "16 bytes)")
        elif option[0] == '--query':
            query = 1
            yaxis = "MRows/s"
            gtitle = ("Selecting with small (16 bytes) record size (file not "
                      "in cache)")
        elif option[0] == '--query-cache':
            query_cache = 1
            yaxis = "MRows/s"
            gtitle = ("Selecting with small (16 bytes) record size (file in "
                      "cache)")
        elif option[0] == '--MB-units':
            mb_units = 1

    filenames = pargs

    if mb_units and yaxis == "MRows/s":
        yaxis = "MB/s"

    if tit:
        gtitle = tit

    plots = []
    legends = []
    for filename in filenames:
        plegend = filename[filename.find('cl-') + 3:filename.index('.h5')]
        plegend = plegend.replace('-', ' ')
        xval, yval = get_values(filename, '')
        print(f"Values for {filename} --> {xval}, {yval}")
        #plots.append(loglog(xval, yval, linewidth=5))
        plots.append(semilogx(xval, yval, linewidth=4))
        legends.append(plegend)
    if 0:  # Per a introduir dades simulades si es vol...
        xval = [1000, 10_000, 100_000, 1_000_000, 10_000_000,
                100_000_000, 1_000_000_000]
#         yval = [0.003, 0.005, 0.02, 0.06, 1.2,
#                 40, 210]
        yval = [0.0009, 0.0011, 0.0022, 0.005, 0.02,
                0.2, 5.6]
        plots.append(loglog(xval, yval, linewidth=5))
        legends.append("PyTables Std")
    show_plot(plots, yaxis, legends, gtitle)
