#set term post color
set term post eps color
set xlabel "Number of rows"
set ylabel "Speed (Krow/s)"

set linestyle 1 lw 7
set linestyle 2 lw 7
set linestyle 3 lw 7
set linestyle 4 lw 7
set logscale x

# For small record size
set output "read-small-lzo-zlib-ucl-comparison.eps"
set tit "Selecting with small record size (16 bytes)"
pl [1000:] [0:1000] "small-nc.out" u ($1):($10) t "No compression" w linesp ls 1, \
 "small-zlib.out" u ($1):($10) t "ZLIB" w linesp ls 2, \
 "small-lzo.out" u ($1):($10) t "LZO" w linesp ls 3, \
 "small-ucl.out" u ($1):($10) t "UCL" w linesp ls 4

# For small record size
set output "write-small-lzo-zlib-ucl-comparison.eps"
set tit "Writing with small record size (16 bytes)"
pl [1000:] [0:500] "small-nc.out" u ($1):($5) tit "No compression" w linesp ls 1, \
 "small-zlib.out" u ($1):($5) tit "ZLIB" w linesp ls 2, \
 "small-lzo.out" u ($1):($5) tit "LZO" w linesp ls 3, \
 "small-ucl.out" u ($1):($5) tit "UCL" w linesp ls 4

