SRCDIR = xml
HTMLDIR = html


.PHONY:		dist clean

dist:		usersguide.pdf usersguide.html
	(cd $(SRCDIR) && make clean)

clean:
	-rm usersguide.pdf
	-rm -r $(HTMLDIR)
	(cd $(SRCDIR) && make clean)


usersguide.pdf: $(SRCDIR)/usersguide.xml
	(cd $(SRCDIR) && make usersguide.pdf)
	cp  $(SRCDIR)/usersguide.pdf  .

usersguide.html: $(SRCDIR)/usersguide.xml
	mkdir -p $(HTMLDIR)
	(cd $(SRCDIR) && make usersguide.html)
	cp  $(SRCDIR)/pytables_style.css $(HTMLDIR)
	cp  $(SRCDIR)/*.html $(HTMLDIR)
	cp  $(SRCDIR)/*.png $(HTMLDIR)
