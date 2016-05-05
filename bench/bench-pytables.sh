#!/bin/sh

export LD_LIBRARY_PATH=$HOME/computacio/hdf5-1.8.1/hdf5/lib
#export PYTHONPATH=..${PYTHONPATH:+:$PYTHONPATH}

bench="python2.7 -O -u indexed_search.py"
flags="-T -m -v -d data.nobackup"
#sizes="1m 2m 5m 10m 20m 50m 100m 200m 500m 1g"
sizes="2g 1g 500m 200m 100m 50m 20m 10m 5m 2m 1m 500k 200k 100k 50k 20k 10k 5k 2k 1k"
#sizes="1m 100k"

#for optimlvl in 0 1 2 3 4 5 6 7 8 9 ; do
for idxtype in ultralight light medium full; do
#for idxtype in medium full; do
  for optimlvl in 0 3 6 9; do
    for compropt in '' '-z1 -lzlib' '-z1 -llzo' ; do
    #for compropt in '-z1 -llzo' ; do
      rm -rf data.nobackup/*  # Atencio: esta correctament posat?
      #for mode in -c '-i -s float' ; do
      for mode in -c '-i' ; do
        for size in $sizes ; do
          $bench $flags $mode -n $size -O $optimlvl -t $idxtype $compropt
        done
      done
    done
  done
done
rm -rf data.nobackup
