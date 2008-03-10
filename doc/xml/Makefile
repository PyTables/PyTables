# Required SVG files
svgs_current := $(wildcard *.svg)
svgs_images := $(wildcard images/*.svg)
svgs := $(svgs_current) $(svgs_images)

# Generated PNG (intermediate) files
genpng := $(patsubst %.svg,%.png,$(svgs))

# PNG files with no SVG version. They MUST NOT be cleaned
no_svg = objecttree-h5.png tutorial1-1-tableview.png tutorial1-2-tableview.png tutorial1-general.png tutorial1-general.png

# Required PNG files
allpng := $(genpng) $(no_svg)

XML_CATALOG_FILES=./custom_layer/pytables_catalog.xml
XML_DEBUG_CATALOG=1
export XML_CATALOG_FILES XML_DEBUG_CATALOG

# The intermediate %.png files will not be automatically deleted by make after
# being used by xsltproc
.PRECIOUS: %.png
.PHONY:		all clean

all:		usersguide.pdf %.html

clean:
	rm -f usersguide.fo
	rm -f usersguide.pdf
	rm -f $(genpng)
	rm -f *.html

usersguide.pdf: usersguide.xml usersguide_bib.xml
	xsltproc --xinclude -o usersguide.fo pytables_fo.xsl usersguide.xml
	fop -d -c fop_conf.xml usersguide.fo usersguide.pdf


%.png: %.svg
	inkscape "$<" --export-png="$@"

%.html:	usersguide.xml $(allpng)
	xsltproc --xinclude pytables_html.xsl usersguide.xml


