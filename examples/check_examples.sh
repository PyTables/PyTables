#!/bin/sh
#
# Small script to check the example repository quickly

# CONFIGURATION - interpreter to use
PYTHON=python

# exit on non-zero return status
set -e

for script in \
    add-column.py \
    array1.py \
    array2.py \
    array3.py \
    array4.py \
    attributes1.py \
    carray1.py \
    earray1.py \
    earray2.py \
    index.py \
    inmemory.py \
    links.py \
    nested1.py \
    nested-tut.py \
    particles.py \
    read_array_out_arg.py \
    split.py \
    table1.py \
    table2.py \
    table3.py \
    tutorial1-1.py \
    tutorial1-2.py \
    tutorial3-1.py \
    tutorial3-2.py \
    undo-redo.py \
    vlarray1.py \
    vlarray2.py \
    vlarray3.py \
    vlarray4.py
do
    $PYTHON "$script"
done

#TO DEBUG:
#--------- python2.7 works
#--------- python3.4 DON'T WORK
# filenodes1.py
# multiprocess_access_queues.py
# multiprocess_access_benchmarks.py
# objecttree.py
# table-tree.py

#--------- python2.7 DON'T WORK
#--------- python3.4 DON'T WORK
# enum.py           # This should always fail
# nested-iter.py    # Run this after "tutorial1-1.py" (file missing)
# tutorial2.py      # This should always fail at the beginning

