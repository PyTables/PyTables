# Get all .svg files
svgs := $(wildcard *.svg)

# Images with no SVG version. They MUST NOT be cleaned
no_svg = logo3-ombra.png objecttree-h5.png tutorial1-1-tableview.png tutorial1-2-tableview.png tutorial1-general.png tutorial1-general.png

# The generated PNGs
genpng := $(patsubst %.svg,%.png,$(svgs))
allpng := $(genpng) $(no_svg)

XML_CATALOG_FILES=./custom_layer/pytables_catalog.xml
XML_DEBUG_CATALOG=1
export XML_CATALOG_FILES XML_DEBUG_CATALOG

.PHONY:		all clean

all:		usersguide.pdf usersguide.html

clean:
	rm -f $(genpng)
	rm -f *.html
	rm -f usersguide.fo
	rm -f usersguide.pdf

usersguide.pdf: usersguide.xml usersguide_bib.xml
	xsltproc -o usersguide.fo pytables_fo.xsl usersguide.xml
	fop -d -c fop_conf.xml usersguide.fo usersguide.pdf


usersguide.html:	 usersguide.xml $(allpng)
	xsltproc pytables_html.xsl usersguide.xml

%.png:	        %.svg
	inkscape "$<" --export-png="$@"

