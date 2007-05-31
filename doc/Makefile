SRCDIR = xml
HTMLDIR = html


.PHONY:		dist clean

dist:		usersguide.pdf usersguide.html
	$(MAKE) -C $(SRCDIR) clean

clean:
	-rm usersguide.pdf
	-rm -r $(HTMLDIR)
	$(MAKE) -C $(SRCDIR) clean


usersguide.pdf: $(SRCDIR)/usersguide.xml
	$(MAKE) -C $(SRCDIR) usersguide.pdf
	cp  $(SRCDIR)/usersguide.pdf  .

usersguide.html: $(SRCDIR)/usersguide.xml
	$(MAKE) -C $(SRCDIR) usersguide.html
	mkdir -p $(HTMLDIR)
	cp  $(SRCDIR)/pytables_style.css $(HTMLDIR)
	cp  $(SRCDIR)/*.html $(HTMLDIR)
	cp  $(SRCDIR)/*.png $(HTMLDIR)
