# Files to be generated
genpdf = objecttree.pdf rootUEP1.pdf rootUEP2.pdf
# Get all .pdf files except "usersguide.pdf"
pdfs := $(filter-out usersguide.pdf, $(wildcard *.pdf))
# Add files that have to be derived from other formats
pdfs += $(genpdf)
# The generated PNGs
genpng := $(patsubst %.pdf,%.png,$(pdfs))
# The generated latex stuff
gentex := $(filter-out usersguide.xml usersguide.bib, $(wildcard usersguide*.*))
# tbook junk to be deleted
tbjunk = makebib makeclean makeepss makepage makepdfs makewebs makeeqns makeidx \
	*image-comment.txt *image-gallery-raw.tex natbib.cfg

.PHONY:		all clean

all:		usersguide.pdf usersguide.html

clean:
	rm -f $(tbjunk)
	rm -f $(genpng)
	rm -f $(gentex)
	rm -f $(genpdf)
	rm -f *.html

usersguide.tex:	usersguide.xml
	tbtolatex "$*"

usersguide.pdf:	usersguide.tex usersguide.bib $(pdfs)
	pdflatex "$<"
	bibtex usersguide
	pdflatex "$<"
	pdflatex "$<"
	pdflatex "$<"

usersguide.html:	 usersguide.xml $(genpng)
	tbtodocbk usersguide
	python strip-dbmanual.py usersguide-db.xml
	docbook2html usersguide-db.xml

%.pdf:		%.dia
	dia --export=tmp.eps --export-to-format=eps "$<"
	epstopdf tmp.eps -outfile="$@"
	rm tmp.eps

%.pdf:		%.fig
	fig2dev -L eps "$<" tmp.eps
	epstopdf tmp.eps -outfile="$@"
	rm tmp.eps

%.png:	        %.pdf
	convert "$<" "$@"

