from pylab import *

def get_values(filename):
    f = open(filename)
    sizes = []
    values = []
    isize = None
    for line in f:
        if line.startswith('range'):
            tmp = line.split(':')[1]
            tmp = tmp.strip()
            tmp = tmp[1:-1]
            lower, upper = int(tmp.split(',')[0]), int(tmp.split(',')[1])
            isize = upper - lower
            #print "isize-->", isize
        if isize is None or isize == 0:
            continue
        if insert and line.startswith('Insert time'):
            tmp = line.split(':')[1]
            #itime = float(tmp[:tmp.index(',')])
            itime = float(tmp)
            sizes.append(isize)
            values.append(itime)
        elif line.startswith('Index time'):
            tmp = line.split(':')[1]
            #xtime = float(tmp[:tmp.index(',')])
            xtime = float(tmp)
            txtime += xtime
            if create_index and create_index in line:
                sizes.append(isize)
                values.append(xtime)
            elif create_total and txtime > xtime:
                sizes.append(isize)
                values.append(txtime)
        elif table_size and line.startswith('Table size'):
            tsize = float(line.split(':')[1])
            sizes.append(isize)
            values.append(tsize)
        elif indexes_size and line.startswith('Indexes size'):
            xsize = float(line.split(':')[1])
            sizes.append(isize)
            values.append(xsize)
        elif total_size and line.startswith('Full size'):
            fsize = float(line.split(':')[1])
            sizes.append(isize)
            values.append(fsize)
        elif (query or query_cache) and line.startswith('[NOREP]'):
            tmp = line.split(':')[1]
            if colname in line:
                if query and '1st' in line:
                    qtime = float(tmp)
                    sizes.append(isize)
                    values.append(qtime)
                elif query_cache and 'cache' in line:
                    qtime = float(tmp[:tmp.index('+-')])
                    sizes.append(isize)
                    values.append(qtime)

    f.close()
    return sizes, values

def show_plot(plots, yaxis, legends, gtitle):
    xlabel('Number of hits')
    ylabel(yaxis)
    title(gtitle)
    #ylim(0, 400)
    grid(True)

#     legends = [f[f.find('-'):f.index('.out')] for f in filenames]
#     legends = [l.replace('-', ' ') for l in legends]
    legend([p[0] for p in plots], legends, loc = "upper left")


    #subplots_adjust(bottom=0.2, top=None, wspace=0.2, hspace=0.2)
    if outfile:
        savefig(outfile)
    else:
        show()

if __name__ == '__main__':

    import sys, getopt

    usage = """usage: %s [-o file] [-t title] [--insert] [--create-index] [--create-total] [--table-size] [--indexes-size] [--total-size] [--query=colname] [--query-cache=colname] files
 -o filename for output (only .png and .jpg extensions supported)
 -t title of the plot
 --insert -- Insert time for table
 --create-index=colname -- Index time for column
 --create-total -- Total time for creation of table + indexes
 --table-size -- Size of table
 --indexes-size -- Size of all indexes
 --total-size -- Total size of table + indexes
 --query=colname -- Time for querying the specified column
 --query-cache=colname -- Time for querying the specified column (cached)
 \n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'o:t:',
                                    ['insert',
                                     'create-index=',
                                     'create-total',
                                     'table-size',
                                     'indexes-size',
                                     'total-size',
                                     'query=',
                                     'query-cache=',
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
    create_index = None
    create_total = 0
    table_size = 0
    indexes_size = 0
    total_size = 0
    query = 0
    query_cache = 0
    colname = None
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
            yaxis = "Time (s)"
            gtitle = "Insert time for table"
        elif option[0] == '--create-index':
            create_index = option[1]
            yaxis = "Time (s)"
            gtitle = "Create index time for column " + create_index
        elif option[0] == '--create-total':
            create_total = 1
            yaxis = "Time (s)"
            gtitle = "Create time for table + indexes"
        elif option[0] == '--table-size':
            table_size = 1
            yaxis = "Size (MB)"
            gtitle = "Table size"
        elif option[0] == '--indexes-size':
            indexes_size = 1
            yaxis = "Size (MB)"
            gtitle = "Indexes size"
        elif option[0] == '--total-size':
            total_size = 1
            yaxis = "Size (MB)"
            gtitle = "Total size (table + indexes)"
        elif option[0] == '--query':
            query = 1
            colname = option[1]
            yaxis = "Time (s)"
            if colname == 'col1':
                gtitle = "Query time for int32 (not indexed)"
            elif colname == 'col3':
                gtitle = "Query time for float64 (not indexed)"
            elif colname == 'col2':
                gtitle = "Query time for int32 (index not in cache)"
            elif colname == 'col4':
                gtitle = "Query time for float64 (index not in cache)"
        elif option[0] == '--query-cache':
            query_cache = 1
            colname = option[1]
            yaxis = "Time (s)"
            if colname == 'col1':
                gtitle = "Query time for int32 (not indexed, in cache)"
            elif colname == 'col3':
                gtitle = "Query time for float64 (not indexed, in cache)"
            elif colname == 'col2':
                gtitle = "Query time for int32 (index in cache)"
            elif colname == 'col4':
                gtitle = "Query time for float64 (index in cache)"

    filenames = pargs

    if tit:
        gtitle = tit

    plots = []
    legends = []
    for filename in filenames:
        plegend = filename[filename.find('-'):filename.index('.out')]
        plegend = plegend.replace('-', ' ')
        xval, yval = get_values(filename)
        print "Values for %s --> %s, %s" % (filename, xval, yval)
        plots.append(loglog(xval, yval, linewidth=5))
        #plots.append(semilogx(xval, yval, linewidth=5))
        legends.append(plegend)
    if 0:  # Per a introduir dades simulades si es vol...
        xval = [1000, 10000, 100000, 1000000, 10000000,
                100000000, 1000000000]
#         yval = [0.003, 0.005, 0.02, 0.06, 1.2,
#                 40, 210]
        yval = [0.0009, 0.0011, 0.0022, 0.005, 0.02,
                0.2, 5.6]
        plots.append(loglog(xval, yval, linewidth=5))
        legends.append("PyTables Std")
    show_plot(plots, yaxis, legends, gtitle)
