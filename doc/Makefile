SRCDIR = xml
HTMLDIR = html


.PHONY:		dist clean

dist:		usersguide.pdf usersguide.html
	(cd $(SRCDIR) && make clean)

clean:
	-rm usersguide.pdf
	-rm -r $(HTMLDIR)
	(cd $(SRCDIR) && make clean)


usersguide.pdf:
	(cd $(SRCDIR) && make usersguide.pdf)
	cp  $(SRCDIR)/usersguide.pdf  .

usersguide.html:
	mkdir -p $(HTMLDIR)
	(cd $(SRCDIR) && make usersguide.html)
	cp  $(SRCDIR)/usersguide*.html $(SRCDIR)/*.{jpg,png,css}  $(HTMLDIR)
