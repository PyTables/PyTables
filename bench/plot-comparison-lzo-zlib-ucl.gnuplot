#set term post color
set term post eps color
set xlabel "log(nrows)"
set ylabel "Krow/s"

set linestyle 1 lw 7
set linestyle 2 lw 7
set linestyle 3 lw 7
set linestyle 4 lw 7

# For small record size
set output "read-small-lzo-zlib-ucl-comparison.eps"
set tit "Selecting with small record size (16 bytes)"
pl [] [0:1000] "small-nc.out" u (log10($1)):($10) t "PyTables & no compression" w linesp ls 1, \
 "small-zlib-nc.out" u (log10($1)):($10) t "PyTables & ZLIB" w linesp ls 2, \
 "small-lzo-nc.out" u (log10($1)):($10) t "PyTables & LZO" w linesp ls 3, \
 "small-ucl-nc.out" u (log10($1)):($10) t "PyTables & UCL" w linesp ls 4


# For small record size
set output "write-small-lzo-zlib-ucl-comparison.eps"
set tit "Writing with small record size (16 bytes)"
pl [] [0:1000] "small-nc.out" u (log10($1)):($5) tit "PyTables & no compression" w linesp ls 1, \
 "small-zlib-nc.out" u (log10($1)):($5) tit "PyTables & ZLIB" w linesp ls 2, \
 "small-lzo-nc.out" u (log10($1)):($5) tit "PyTables & LZO" w linesp ls 3, \
 "small-ucl-nc.out" u (log10($1)):($5) tit "PyTables & UCL" w linesp ls 4

