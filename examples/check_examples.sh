#!/bin/sh

set -e

PYTHON=python
# Small script to check the example repository quickly
$PYTHON add-column.py
$PYTHON array1.py
$PYTHON array2.py
$PYTHON array3.py
$PYTHON array4.py
$PYTHON attributes1.py
$PYTHON carray1.py
$PYTHON earray1.py
$PYTHON earray2.py
#$PYTHON enum.py       # This should always fail
$PYTHON filenodes1.py
$PYTHON index.py
$PYTHON inmemory.py
$PYTHON links.py
$PYTHON multiprocess_access_benchmarks.py
$PYTHON multiprocess_access_queues.py
$PYTHON nested1.py
#$PYTHON nested-iter.py    # Run this after "tutorial1-1.py"
$PYTHON nested-tut.py
$PYTHON objecttree.py
$PYTHON particles.py
$PYTHON read_array_out_arg.py
$PYTHON split.py
$PYTHON table1.py
$PYTHON table2.py
$PYTHON table3.py
$PYTHON table-tree.py
$PYTHON tutorial1-1.py
$PYTHON tutorial1-2.py
#$PYTHON tutorial2.py   # This should always fail at the beginning
$PYTHON tutorial3-1.py
$PYTHON tutorial3-2.py
$PYTHON undo-redo.py
$PYTHON vlarray1.py
$PYTHON vlarray2.py
$PYTHON vlarray3.py
$PYTHON vlarray4.py


$PYTHON nested-iter.py
