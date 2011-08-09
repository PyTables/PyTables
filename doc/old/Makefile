SRCDIR = xml
HTMLDIR = html


.PHONY:		dist clean

dist:		usersguide.pdf html
	$(MAKE) -C $(SRCDIR) clean

clean:
	rm -f usersguide.pdf
	rm -rf $(HTMLDIR)
	$(MAKE) -C $(SRCDIR) clean


usersguide.pdf: $(SRCDIR)/usersguide.xml
	$(MAKE) -C $(SRCDIR) usersguide.pdf
	cp  $(SRCDIR)/usersguide.pdf  .

# The copy of PNG files should be done in the %.html target of
# xml/Makefile IF the %.png target was not .PRECIOUS
html: $(SRCDIR)/usersguide.xml
	$(MAKE) -C $(SRCDIR) %.html
	mkdir -p $(HTMLDIR)/images
	cp  $(SRCDIR)/pytables_style.css $(HTMLDIR)
	cp  $(SRCDIR)/*.html $(HTMLDIR)
	cp  $(SRCDIR)/*.png $(HTMLDIR)
	cp  $(SRCDIR)/images/*.png $(HTMLDIR)/images
