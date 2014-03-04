# -*- coding: utf-8 -*-

########################################################################
#
# License: BSD
# Created: April 9, 2013
# Author:  Anthony Scopatz - scopatz@gmail.com
#
# $Id$
#
########################################################################

"""This utility helps you migrate from PyTables 2.x APIs to 3.x APIs, which
are PEP 8 compliant.

"""
import os
import re
import sys
import argparse

from tables._past import old2newnames, new2oldnames

# Note that it is tempting to use the ast module here, but then this
# breaks transforming cython files.  So instead we are going to do the
# dumb thing with replace.


def make_subs(ns):
    names = new2oldnames if ns.reverse else old2newnames
    s = '(?<=\W)({0})(?=\W)'.format('|'.join(names.keys()))
    if ns.ignore_previous:
        s += '(?!\s*?=\s*?previous_api(_property)?\()'
        s += '(?!\* to \*\w+\*)'
        s += '(?!\* parameter has been renamed into \*\w+\*\.)'
        s += '(?! is pending deprecation, import \w+ instead\.)'
    subs = re.compile(s, flags=re.MULTILINE)

    def repl(m):
        return names.get(m.group(1), m.group(0))
    return subs, repl


def main():
    desc = ('PyTables 2.x -> 3.x API transition tool\n\n'
            'This tool displays to standard out, so it is \n'
            'common to pipe this to another file:\n\n'
            '$ pt2to3 oldfile.py > newfile.py')
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-r', '--reverse', action='store_true', default=False,
                        dest='reverse',
                        help="reverts changes, going from 3.x -> 2.x.")
    parser.add_argument('-p', '--no-ignore-previous', action='store_false',
                        default=True, dest='ignore_previous',
                        help="ignores previous_api() calls.")
    parser.add_argument('-o', default=None, dest='output',
                        help="output file to write to.")
    parser.add_argument('-i', '--inplace', action='store_true', default=False,
                        dest='inplace', help="overwrites the file in-place.")
    parser.add_argument('filename', help='path to input file.')
    ns = parser.parse_args()

    if not os.path.isfile(ns.filename):
        sys.exit('file {0!r} not found'.format(ns.filename))
    with open(ns.filename, 'r') as f:
        src = f.read()

    subs, repl = make_subs(ns)
    targ = subs.sub(repl, src)

    ns.output = ns.filename if ns.inplace else ns.output
    if ns.output is None:
        sys.stdout.write(targ)
    else:
        with open(ns.output, 'w') as f:
            f.write(targ)

if __name__ == '__main__':
    main()
