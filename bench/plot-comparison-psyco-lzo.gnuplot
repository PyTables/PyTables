#set term post color
set term post eps color
set xlabel "Number of rows"
set ylabel "Speed (Krow/s)"

set linestyle 1 lw 7
set linestyle 2 lw 7
set linestyle 3 lw 7
set linestyle 4 lw 7

# For small record size
set output "read-small-psyco-lzo-comparison.eps"
set tit "Selecting with small record size (16 bytes)"
set logscale x
pl [1000:] [0:1200] "small-psyco-lzo.out" u ($1):($10) t "Psyco & compression (LZO)" w linesp ls 2, \
 "small-psyco-nc.out" u ($1):($10) tit "Psyco & no compresion" w linesp ls 3, \
 "small-lzo.out" u ($1):($10) t "No Psyco & compression (LZO)" w linesp ls 1, \
 "small-nc.out" u ($1):($10) tit "No Psyco & no compression" w linesp ls 4

# For small record size
set output "write-small-psyco-lzo-comparison.eps"
set tit "Writing with small record size (16 bytes)"
set logscale x
pl [1000:] [0:1000] "small-psyco-lzo.out" u ($1):($5) t "Psyco & compression (LZO)" w linesp ls 2, \
 "small-psyco-nc.out" u ($1):($5) tit "Psyco & no compresion" w linesp ls 3, \
 "small-lzo.out" u ($1):($5) t "No Psyco & compression (LZO)" w linesp ls 1, \
 "small-nc.out" u ($1):($5) tit "No Psyco & no compression" w linesp ls 4

