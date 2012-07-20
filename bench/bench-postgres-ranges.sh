#!/bin/sh

export PYTHONPATH=..${PYTHONPATH:+:$PYTHONPATH}

pyopt="-O -u"
#qlvl="-Q8 -x"
#qlvl="-Q8"
qlvl="-Q7"
#size="500m"
size="1g"

#python $pyopt indexed_search.py -P -c -n $size -m -v
python $pyopt indexed_search.py -P -i -n $size -m -v -sfloat $qlvl

