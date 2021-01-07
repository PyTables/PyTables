#!/usr/bin/env python
# a stacked bar plot with errorbars

from pathlib import Path
from pylab import *

checks = ['open_close', 'only_open',
          'full_browse', 'partial_browse',
          'full_browse_attrs', 'partial_browse_attrs',
          'open_group', 'open_leaf',
          'total']
width = 0.15       # the width of the bars: can also be len(x) sequence
colors = ['r', 'm', 'g', 'y', 'b']
ind = arange(len(checks))    # the x locations for the groups


def get_values(filename):
    values = []
    for line in Path(filename).read_text().splitlines():
        if show_memory:
            if line.startswith('VmData:'):
                values.append(float(line.split()[1]) / 1024)
        else:
            if line.startswith('WallClock time:'):
                values.append(float(line.split(':')[1]))
    return values


def plot_bar(values, n):
    global ind
    if not gtotal:
        # Remove the grand totals
        values.pop()
        if n == 0:
            checks.pop()
            ind = arange(len(checks))
    p = bar(ind + width * n, values, width, color=colors[n])
    return p


def show_plot(bars, filenames, tit):
    if show_memory:
        ylabel('Memory (MB)')
    else:
        ylabel('Time (s)')
    title(tit)
    n = len(filenames)
    xticks(ind + width * n / 2, checks, rotation=45,
           horizontalalignment='right', fontsize=8)
    if not gtotal:
        #loc = 'center right'
        loc = 'upper left'
    else:
        loc = 'center left'

    legends = [f[:f.index('_')] for f in filenames]
    legends = [l.replace('-', ' ') for l in legends]
    legend([p[0] for p in bars], legends, loc=loc)

    subplots_adjust(bottom=0.2, top=None, wspace=0.2, hspace=0.2)
    if outfile:
        savefig(outfile)
    else:
        show()

if __name__ == '__main__':

    import sys
    import getopt

    usage = """usage: %s [-g] [-m] [-o file] [-t title] files
            -g grand total
            -m show memory instead of time
            -o filename for output (only .png and .jpg extensions supported)
            -t title of the plot
            \n""" % sys.argv[0]

    try:
        opts, pargs = getopt.getopt(sys.argv[1:], 'gmo:t:')
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
    tit = "Comparison of differents PyTables versions"
    gtotal = 0
    show_memory = 0
    outfile = None

    # Get the options
    for option in opts:
        if option[0] == '-g':
            gtotal = 1
        elif option[0] == '-m':
            show_memory = 1
        elif option[0] == '-o':
            outfile = option[1]
        elif option[0] == '-t':
            tit = option[1]

    filenames = pargs
    bars = []
    n = 0
    for filename in filenames:
        values = get_values(filename)
        print("Values-->", values)
        bars.append(plot_bar(values, n))
        n += 1
    show_plot(bars, filenames, tit)
