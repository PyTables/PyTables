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
    s = '\W({0})\W'.format('|'.join(names.keys()))
    if ns.ignore_previous:
        s += '(?!\s*?=\s*?previous_api\()'
    subs = re.compile(s, flags=re.MULTILINE)
    def repl(m):
        return names.get(m.group(1), m.group(0))
    return subs, repl


def main():
    parser = argparse.ArgumentParser(description='PyTables 2.x -> 3.x API transition tool')
    parser.add_argument('-r', '--reverse', action='store_true', default=False, 
                        dest='reverse', help="reverts changes, going from 3.x -> 2.x.")
    parser.add_argument('-p', '--no-ignore-previous', action='store_false', 
                default=True, dest='ignore_previous', help="i previous_api() calls.")
    parser.add_argument('filename', help='path to input file.')
    ns = parser.parse_args()

    if not os.path.isfile(ns.filename):
        sys.exit('file {0!r} not found'.format(ns.filename))
    with open(ns.filename, 'r') as f:
        src = f.read()

    subs, repl = make_subs(ns)
    targ = subs.sub(repl, src)
    print targ

if __name__ == '__main__':
    main()
