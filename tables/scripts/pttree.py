# -*- coding: utf-8 -*-

########################################################################
#
# License: BSD
# Created: November 8, 2014
# Author:  Alistair Muldal - alimuldal@gmail.com
#
# $Id$
#
########################################################################

"""This utility prints the contents of an HDF5 file as a tree.

Pass the flag -h to this for help on usage.

"""

import tables
import numpy as np
import os
import argparse

def _get_parser():
    parser = argparse.ArgumentParser(
        description='''
        `pttree` is designed to give a quick overview of the contents of a
        PyTables HDF5 file by printing a depth-indented list of nodes, similar
        to the output of the Unix `tree` utility for viewing directory
        structures. It can also display the size, shape and compression states
        of individual nodes, as well as summary information for the whole file.
        For a more verbose output including metadata, see `ptdump`.
        ''')


    parser.add_argument(
        '-L', '--max-level', type=int, dest='max_depth',
        help='maximum display depth of tree (-1 = no limit)',
    )
    parser.add_argument(
        '--print-size', action='store_true', dest='print_size',
        help='print size of each node',
    )
    parser.add_argument(
        '--no-print-size', action='store_false', dest='print_size',
    )
    parser.add_argument(
        '--print-shape', action='store_true', dest='print_shape',
        help='print shape of each node',
    )
    parser.add_argument(
        '--no-print-shape', action='store_false', dest='print_shape',
    )
    parser.add_argument(
        '--print-compression', action='store_true', dest='print_compression',
        help='print compression library(level) for each compressed node',
    )
    parser.add_argument(
        '--no-print-compression', action='store_false',
        dest='print_compression',
    )

    parser.add_argument('src', metavar='filename[:nodepath]',
                        help='path to the root of the tree structure')

    parser.set_defaults(max_depth=-1, print_size=True, print_shape=False,
                        print_compression=False)

    return parser


def main():

    parser = _get_parser()
    args = parser.parse_args()

    # Catch the files passed as the last arguments
    src = args.__dict__.pop('src').split(':')
    if len(src) == 1:
        filename, nodename = src[0], "/"
    else:
        filename, nodename = src
        if nodename == "":
            # case where filename == "filename:" instead of "filename:/"
            nodename = "/"

    with tables.open_file(filename, 'r') as f:
        tree_str = get_tree_str(f, nodename, **args.__dict__)
        print tree_str

    pass

def get_tree_str(f, where='/', max_depth=-1, print_class=True,
                 print_size=True, print_shape=False, print_compression=False,
                 print_total=True):

    root_node = f.get_node(where)
    root_node._g_check_open()

    start_depth = root_node._v_depth

    tree_nodes = {}

    total_in_mem = 0
    total_on_disk = 0
    total_items = 0

    if max_depth < 0:
        max_depth = os.sys.maxint

    for node in f.walk_nodes(root_node):

        pathname = node._v_pathname
        parent_pathname = node._v_parent._v_pathname
        name  = node._v_name
        if print_class:
            name += " (%s)" % node.__class__.__name__
        labels = []

        depth = node._v_depth - start_depth

        if depth > max_depth:
            # this is pretty dumb, but I don't really know of a way to stop
            # walk_nodes at a particular depth
            continue

        elif depth == max_depth and isinstance(node, tables.group.Group):

            # we measure the size of all of the children of this branch
            n_items, in_mem, on_disk = get_branch_size(f, node)
            ratio = float(on_disk) / in_mem
            if print_size:
                sizestr = ', total size=(%s/%s/%.2f)' % (
                    b2h(in_mem), b2h(on_disk), ratio)
            else:
                sizestr = ''
            extra_itemstr = '... %i items%s' % (n_items, sizestr)
            labels.append(extra_itemstr)

            total_items += n_items
            total_on_disk += on_disk
            total_in_mem += in_mem

            pass

        else:

            # node labels
            if isinstance(node, tables.link.Link):
                labels.append('target=%s' % node.target)

            elif isinstance(node, (tables.array.Array, tables.table.Table)):

                on_disk = node.size_on_disk
                in_mem = node.size_in_memory
                ratio = float(on_disk) / in_mem
                if print_size:
                    labels.append('size=(%s/%s/%.2f)' % (
                                  b2h(in_mem), b2h(on_disk), ratio))
                if print_shape:
                    labels.append('shape=%s' % node.shape)
                if print_compression:
                    lib = node.filters.complib
                    level = node.filters.complevel
                    if level:
                        compstr = '%s(%i)' % (lib, level)
                    else:
                        compstr = 'None'
                    labels.append('compression=%s' % compstr)

                total_items += 1
                total_on_disk += on_disk
                total_in_mem += in_mem

        new_tree_node = PrettyTree(name, labels=labels)
        tree_nodes.update({pathname:new_tree_node})

        # exclude root node (otherwise we get infinite recursions)
        if pathname != '/' and parent_pathname in tree_nodes:
            tree_nodes[parent_pathname].add_child(new_tree_node)

    out_str = '\n' + '-' * 60 + '\n' * 2
    out_str += str(tree_nodes[root_node._v_pathname]) + '\n' * 2

    if print_total:
        avg_ratio = float(total_on_disk) / total_in_mem
        fsize = os.stat(f.filename).st_size

        out_str += '-' * 60 + '\n'
        out_str += 'Total stored items:     %i\n' % total_items
        out_str += 'Total data size:        %s in memory, %s on disk\n' % (
                    b2h(total_in_mem), b2h(total_on_disk))
        out_str += 'Mean compression ratio: %.2f\n' % avg_ratio
        out_str += 'HDF5 file size:         %s\n' % b2h(fsize)
        out_str += '-' * 60 + '\n'

    return out_str


class PrettyTree(object):
    """

    A pretty ASCII representation of a recursive tree structure. Each node can
    have multiple labels, given as a list of strings.

    Example:
    --------

        A = PrettyTree('A', labels=['wow'])
        B = PrettyTree('B', labels=['such tree'])
        C = PrettyTree('C', children=[A, B])
        D = PrettyTree('D', labels=['so recursive'])
        root = PrettyTree('root', labels=['many nodes'], children=[C, D])
        print root

    Credit to Andrew Cooke's blog:
    <http://www.acooke.org/cute/ASCIIDispl0.html>

    """

    def __init__(self, name, children=None, labels=None):

        # NB: do NOT assign default list/dict arguments in the function
        # declaration itself - these objects are shared between ALL instances
        # of PrettyTree, and by assigning to them it's easy to get into
        # infinite recursions, e.g. when 'self in self.children == True'
        if children is None:
            children = []
        if labels is None:
            labels = []

        self.name = name
        self.children = children
        self.labels = labels

    def add_child(self, child):
        # some basic checks to help to avoid infinite recursion
        assert child is not self
        assert child not in self.children
        assert self not in child.children
        self.children.append(child)

    def tree_lines(self):
        yield self.name
        for label in self.labels:
            yield '   ' + label
        last = self.children[-1] if self.children else None
        for child in self.children:
            prefix = '`--' if child is last else '+--'
            for line in child.tree_lines():
                yield prefix + line
                prefix = '   ' if child is last else '|  '

    def __str__(self):
        return "\n".join(self.tree_lines())

    def __repr__(self):
        return '<%s at %s>' % (self.__class__.__name__, hex(id(self)))


def b2h(nbytes, use_si_units=False):

    if use_si_units:
        prefixes = 'TB', 'GB', 'MB', 'kB', 'B'
        values = 1E12, 1E9, 1E6, 1E3, 1
    else:
        prefixes = 'TiB', 'GiB', 'MiB', 'KiB', 'B'
        values = 2 ** 40, 2 ** 30, 2 ** 20, 2 ** 10, 1

    for (prefix, value) in zip(prefixes, values):
        scaled = float(nbytes) / value
        if scaled >= 1:
            break

    return "%.1f%s" % (scaled, prefix)


def get_branch_size(f, where):

    total_mem = 0.
    total_disk = 0.
    total_items = 0

    for node in f.walk_nodes(where):

        # don't dereference links, or we'll count the same arrays multiple
        # times
        if not isinstance(node, tables.link.Link):
            try:
                in_mem = node.size_in_memory
                on_disk = node.size_on_disk
            except AttributeError:
                continue

            total_mem += in_mem
            total_disk += on_disk
            total_items += 1

    return total_items, total_mem, total_disk


def make_test_file(prefix='/tmp'):
    f = tables.open_file(os.path.join(prefix, 'test_pttree.hdf5'), 'w')

    g1 = f.create_group('/', 'group1')
    g1a = f.create_group(g1, 'group1a')
    g1b = f.create_group(g1, 'group1b')

    filters = tables.Filters(complevel=5, complib='bzip2')

    for gg in g1a, g1b:
        f.create_carray(gg, 'zeros128b', obj=np.zeros(32, dtype=np.float64),
                        filters=filters)
        f.create_carray(gg, 'random128b', obj=np.random.rand(32),
                        filters=filters)

    g2 = f.create_group('/', 'group2')

    softlink = f.create_soft_link(g2, 'softlink_g1_z128',
                                  '/group1/group1a/zeros128b')
    hardlink = f.create_hard_link(g2, 'hardlink_g1a_z128',
                                  '/group1/group1a/zeros128b')

    return f