#!/bin/sh
# I don't know why, but the /usr/bin/python2.3 from Debian is a 30% slower
# than my own compiled version! 2004-08-18
python="/usr/local/bin/python2.3 -O"

writedata () {
  nrows=$1
  bfile=$2
  worst=$3
  psyco=$4
  if [ "$shuffle" = "1" ]; then
      shufflef="-S"
  else
      shufflef=""
  fi
  cmd="${python} search-bench.py -R ${worst} -b ${bfile} -h ${psyco} -l ${libcomp} -c ${complevel} ${shufflef} -w -n ${nrows} data.nobackup/bench-${libcomp}-${nrows}k.h5"
  echo ${cmd}
  ${cmd}
}

readdata () {
  nrows=$1
  bfile=$2
  worst=$3
  psyco=$4
  smode=$5

  if [ "$smode" = "indexed" ]; then
      #repeats=100
      repeats=20
  else
      repeats=2
  fi
  cmd="${python} search-bench.py -R ${worst} -h -b ${bfile} ${psyco} -m ${smode} -r -k ${repeats} data.nobackup/bench-${libcomp}-${nrows}k.h5"
  echo ${cmd}
  ${cmd}
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

# Configuration for testing
#nrowslist="50000"
#nrowslistworst="50000"

# Normal test
#nrowslist="1 2 5 10 20 50 100 200 500 1000 2000 5000 10000 20000"
#nrowslistworst="1 2 5 10 20 50 100 200 500 1000 2000 5000 10000 20000"
nrowslist="1 2 5 10 20 50 100 200 500 1000"
nrowslistworst="1 2 5 10 20 50 100 200 500 1000"
#nrowslist="1 2 5 10"
#nrowslistworst="1 2 5 10"

# The next can be regarded as parameters
shuffle=1

for libcomp in none zlib lzo; do
#for libcomp in none lzo; do
    if [ "$libcomp" = "none" ]; then
	complevel=0
    else
	complevel=1
    fi
    # The name of the data bench file
    bfile="worst-dbench-cl-${libcomp}-c${complevel}-S${shuffle}.h5"
    
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
    for worst in "" -t; do
    #for worst in ""; do
        # Write data files
	if [ "$worst" = "-t" ]; then
	    echo
	    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
	    echo "Entering worst case..."
	    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
	    echo
	    nrowslist=$nrowslistworst
	fi
        # Write data file
	for nrows in $nrowslist; do
	    echo "*************************************************************"
	    echo "Writing for nrows=$nrows Krows, psyco=$psyco, worst='${worst}'"
	    echo "*************************************************************"
	    writedata ${nrows} ${bfile} "${worst}" "${psyco}"
	done
        # Read data files
	for smode in indexed inkernel standard; do
	    ${python} cacheout.py
	    for nrows in $nrowslist; do
		echo "***********************************************************"
		echo "Searching for nrows=$nrows Krows, $smode, psyco=$psyco, worst='${worst}'"
		echo "***********************************************************"
		readdata ${nrows} ${bfile} "${worst}" "${psyco}" "${smode}"
	    done
	done
        # Finally, after the final search, delete the source (if desired)
# 	for nrows in $nrowslist; do
# 	    rm -f data.nobackup/bench-${libcomp}-${nrows}k.h5
# 	done
    done
    echo "New data available on: $bfile"
done

exit 0
