# This Makefile is only intended to prepare for distribution the PyTables
# sources exported from a repository.  For building and installing PyTables,
# please use ``setup.py`` as described in the ``README.txt`` file.

VERSION = $(shell cat VERSION)
SRCDIRS = tables src doc

GENERATED = ANNOUNCE.txt


.PHONY:		dist clean distclean

dist:		$(GENERATED)
	for srcdir in $(SRCDIRS) ; do $(MAKE) -C $$srcdir $@ ; done
	python setup.py sdist
	cp doc/usersguide-$(VERSION).pdf dist/pytablesmanual-$(VERSION).pdf
	tar -C doc -cvzf dist/pytablesmanual-$(VERSION)-html.tar.gz html
	cd dist && md5sum tables-$(VERSION).tar.gz > pytables-$(VERSION).md5 && cd -

clean:
	rm -rf MANIFEST build dist doc/build
	rm -f $(GENERATED) tables/*.so tables/numexpr/*.so
	find . '(' -name '*.py[co]' -o -name '*~' ')' -exec rm '{}' ';'
	for srcdir in $(SRCDIRS) ; do $(MAKE) -C $$srcdir $@ ; done

distclean: clean
	rm -rf doc/html
	rm -f doc/usersguide-*.pdf
	rm -f tables/_comp_*.c tables/*Extension.c tables/linkExtension.pyx

%:		%.in VERSION
	cat "$<" | sed -e 's/@VERSION@/$(VERSION)/g' > "$@"
