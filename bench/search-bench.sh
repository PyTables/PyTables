#!/bin/sh
python="python2.5 -O"

writedata () {
  nrows=$1
  bfile=$2
  heavy=$3
  psyco=$4
  if [ "$shuffle" = "1" ]; then
      shufflef="-S"
  else
      shufflef=""
  fi
  cmd="${python} search-bench.py -b ${bfile} ${heavy} ${psyco} -l ${libcomp} -c ${complevel} ${shufflef} -w -n ${nrows} -x data.nobackup/bench-${libcomp}-${nrows}k.h5"
  echo ${cmd}
  ${cmd}
}

readdata () {
  nrows=$1
  bfile=$2
  heavy=$3
  psyco=$4
  smode=$5

  if [ "$smode" = "indexed" ]; then
      repeats=100
  else
      repeats=2
  fi
  if [ "$heavy" = "-h" -a "$smode" = "standard" ]; then
      # For heavy mode don't do a standard search
      echo "Skipping the standard search for heavy mode"
  else
      cmd="${python} search-bench.py -b ${bfile} ${heavy} ${psyco} -m ${smode} -r -k ${repeats} data.nobackup/bench-${libcomp}-${nrows}k.h5"
      echo ${cmd}
      ${cmd}
  fi
  if [ "$smode" = "standard" -a "1" = "0" ]; then
      # Finally, after the final search, delete the source (if desired)
      rm -f data.nobackup/bench-${libcomp}-${nrows}k.h5
  fi
  return
}

overwrite=0
if [ $# > 1 ]; then
    if [ "$1" = "-o" ]; then
	overwrite=1
    fi
fi
if [ $# > 2 ]; then
    psyco=$2
fi
# The next can be regarded as parameters
libcomp="lzo"
complevel=1
shuffle=1

# The name of the data bench file
bfile="dbench-cl-${libcomp}-c${complevel}-S${shuffle}.h5"

# Move out a possible previous benchmark file
bn=`basename $bfile ".h5"`
mv -f ${bn}-bck2.h5 ${bn}-bck3.h5
mv -f ${bn}-bck.h5 ${bn}-bck2.h5
if [ "$overwrite" = "1" ]; then
    echo "moving ${bn}.h5 to ${bn}-bck.h5"
    mv -f ${bn}.h5 ${bn}-bck.h5
else
    echo "copying ${bn}.h5 to ${bn}-bck.h5"
    cp -f ${bn}.h5 ${bn}-bck.h5
fi

# Configuration for testing
nrowslist="1 2"
nrowslistheavy="5 10"
# This config takes 10 minutes to complete (psyco, zlib)
#nrowslist="1 2 5 10 20 50 100 200 500 1000"
#nrowslistheavy="2000 5000 10000"
#nrowslist=""
#nrowslistheavy="1 2 5 10 20 50 100 200 500 1000 2000 5000 10000 20000 50000 100000"

# Normal test
#nrowslist="1 2 5 10 20 50 100 200 500 1000 2000 5000 10000"
#nrowslistheavy="20000 50000 100000 200000 500000 1000000"
# Big test
#nrowslist="1 2 5 10 20 50 100 200 500 1000 2000 5000 10000"
#nrowslistheavy="20000 50000 100000 200000 500000 1000000 2000000 5000000"

for heavy in "" -h; do
    # Write data files (light mode)
    if [ "$heavy" = "-h" ]; then
	echo
	echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
	echo "Entering heavy mode..."
	echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
	echo
	nrowslist=$nrowslistheavy
    fi
    # Write data file
    for nrows in $nrowslist; do
	echo "*************************************************************"
	echo "Writing for nrows=$nrows Krows, psyco=$psyco, heavy='${heavy}'"
	echo "*************************************************************"
	writedata ${nrows} ${bfile} "${heavy}" "${psyco}"
    done
    # Read data files
    #for smode in indexed inkernel standard; do
    for smode in inkernel standard; do
#    for smode in indexed; do
	${python} cacheout.py
	for nrows in $nrowslist; do
	    echo "***********************************************************"
	    echo "Searching for nrows=$nrows Krows, $smode, psyco=$psyco, heavy='${heavy}'"
	    echo "***********************************************************"
	    readdata ${nrows} ${bfile} "${heavy}" "${psyco}" "${smode}"
	done
    done
done

echo "New data available on: $bfile"
exit 0
