#!/bin/sh
# I don't know why, but the /usr/bin/python2.3 from Debian is a 30% slower
# than my own compiled version! 2004-08-18
python="/usr/local/bin/python2.3 -O"

writedata () {
  nrows=$1
  bfile=$2
  smode=$3
  psyco=$4
  cmd="${python} sqlite-search-bench.py -R -h -b ${bfile} ${psyco} -m ${smode} -w -n ${nrows} data.nobackup/sqlite-${nrows}k.h5"
  echo ${cmd}
  ${cmd}
}

readdata () {
  nrows=$1
  bfile=$2
  smode=$3
  psyco=$4

  if [ "$smode" = "indexed" ]; then
      #repeats=100
      repeats=20
  else
      repeats=2
  fi
  cmd="${python} sqlite-search-bench.py -R -h -b ${bfile} ${psyco} -n ${nrows} -m ${smode} -r -k ${repeats} data.nobackup/sqlite-${nrows}k.h5"
  echo ${cmd}
  ${cmd}
  # Finally, delete the source (if desired)
  if [ "$smode" = "indexed" ]; then
      echo "Deleting data file data.nobackup/sqlite-${nrows}k.h5"
#      rm -f data.nobackup/sqlite-${nrows}k.h5
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

# The name of the data bench file
bfile="sqlite-dbench.h5"
  
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
nrowsliststd="1 2"
nrowslistidx="1 2"
#nrowsliststd="1 2 5 10 20 50 100 200 500 1000 2000 5000 10000 20000 50000"
#nrowsliststd="1 2 5 10 20"
#nrowslistidx="1 2 5 10 20"
# nrowsliststd="1 2 5 10 20 50 100 200 500 1000 2000 5000 10000"
# nrowslistidx="1 2 5 10 20 50 100 200 500 1000 2000 5000 10000"
#nrowsliststd="1 2 5 10 20 50 100 200 500 1000 2000 5000 10000 20000 50000 100000"
#nrowslistidx="1 2 5 10 20 50 100 200 500 1000 2000 5000 10000 20000 50000 100000"

for smode in standard indexed; do
#for smode in indexed; do
    echo
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo "Entering ${smode} mode..."
    echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
    echo
    if [ "$smode" = "standard" ]; then
	nrowslist=$nrowsliststd
    else
	nrowslist=$nrowslistidx
    fi
    # Write data files
    for nrows in $nrowslist; do
	echo "*************************************************************"
	echo "Writing for nrows=$nrows Krows, $smode, psyco=$psyco"
	echo "*************************************************************"
	writedata ${nrows} ${bfile} "${smode}" "${psyco}"
    done
    # Read data files
    ${python} cacheout.py
    for nrows in $nrowslist; do
	echo "***********************************************************"
	echo "Searching for nrows=$nrows Krows, $smode, psyco=$psyco"
	echo "***********************************************************"
	readdata ${nrows} ${bfile} "${smode}" "${psyco}"
    done
done

echo "New data available on: $bfile"
exit 0
