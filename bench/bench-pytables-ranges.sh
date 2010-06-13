#!/bin/sh

#export LD_LIBRARY_PATH=$HOME/computacio/hdf5-1.8.2/hdf5/lib
export PYTHONPATH=..:$PYTHONPATH

bench="python -O -u indexed_search.py"
flags="-T -m -v "
#sizes="1g 500m 200m 100m 50m 20m 10m 5m 2m 1m"
sizes="1g"
#sizes="1m"
#working_dir="data.nobackup"
working_dir="/scratch3/faltet"

#for comprlvl in '-z0' '-z1 -lzlib' '-z1 -llzo' ; do
for comprlvl in '-z1 -llzo' ; do
  #for optlvl in '-tfull -O9' '-tmedium -O6' '-tlight -O6' '-tultralight -O3'; do
  #for optlvl in '-tultralight -O3' '-tlight -O6' '-tmedium -O6' '-tfull -O9'; do
  for optlvl in '-tultralight -O3'; do
    rm -f $working_dir/*
    #for mode in -c '-Q8 -i -x -s float' ; do
    for mode in '-c -s float' '-Q8 -I -s float' '-Q8 -S -s float'; do
      for size in $sizes ; do
        $bench $flags $mode -n $size $optlvl $comprlvl -d $working_dir
      done
    done
  done
done
