GENEPS = logo3-ombra.eps objecttree-h5.eps objecttree.eps \
	tutorial1-tableview.eps tutorial1-general.eps tutorial2-tableview.eps \
	rootUEP1.eps rootUEP2.eps
GENTEX = usersguide.tex usersguide.log usersguide.aux usersguide.toc \
	natbib.cfg usersguide.bbl usersguide.blg usersguide.brf \
	usersguide.out usersguide.pdf
TBJUNK = makebib makeclean makeepss makepage makepdfs makewebs makeeqns makeidx


# For TBook to find the 'saxon' script.
PATH := $(PATH):.



.PHONY:		all clean

all:		usersguide.pdf usersguide.html

clean:
	-sh makeclean
	rm -f $(TBJUNK)
	rm -f $(GENEPS)
	rm -f $(GENTEX)
	rm -f usersguide*.html


# Relying on the TeX file ensures creation of TBook's 'make*' scripts.
tbook-scripts:	usersguide.tex


usersguide.tex:	usersguide.xml
	tbtolatex "$*"

usersguide-eps2pdf:	tbook-scripts $(GENEPS)
	sh makepdfs

usersguide.pdf:	usersguide.tex usersguide.bib usersguide-eps2pdf
	pdflatex "$<"
	bibtex usersguide
	pdflatex "$<"
	pdflatex "$<"
	pdflatex "$<"


usersguide-eps2png:	tbook-scripts $(GENEPS)
	sh makewebs

usersguide.html:	usersguide-eps2png
	tbtohtml -t usersguide split-level=chapter \
		css-mode=very-careful css-file=html.css


%.eps:		%.dia
	dia --export="$@" --export-to-format=eps "$<"

%.eps:		%.fig
	fig2dev -L eps "$<" "$@"

%.eps:		%.jpg
	convert "$<" "$@"

%.pdf:		%.dvi
	dvipdf "$<" "$@"
