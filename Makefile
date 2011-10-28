# This Makefile is only intended to prepare for distribution the PyTables
# sources exported from a repository.  For building and installing PyTables,
# please use ``setup.py`` as described in the ``README.txt`` file.

VERSION = $(shell cat VERSION)
SRCDIRS = src doc
PYTHON = python
GENERATED = ANNOUNCE.txt


.PHONY:		all dist clean distclean

all:		$(GENERATED)
	for srcdir in $(SRCDIRS) ; do $(MAKE) -C $$srcdir $@ ; done

dist:		all
	$(PYTHON) setup.py sdist
	cd dist && md5sum tables-$(VERSION).tar.gz > pytables-$(VERSION).md5 && cd -
	cp RELEASE_NOTES.txt dist/RELEASE_NOTES-$(VERSION).txt
	for srcdir in $(SRCDIRS) ; do $(MAKE) -C $$srcdir $@ ; done

clean:
	rm -rf MANIFEST build dist
	rm -f $(GENERATED) tables/*.so tables/numexpr/*.so
	find . '(' -name '*.py[co]' -o -name '*~' ')' -exec rm '{}' ';'
	for srcdir in $(SRCDIRS) ; do $(MAKE) -C $$srcdir $@ ; done

distclean:	clean
	for srcdir in $(SRCDIRS) ; do $(MAKE) -C $$srcdir $@ ; done
	rm -f tables/_comp_*.c tables/*Extension.c tables/linkExtension.pyx
	#git clean -fdx

%:		%.in VERSION
	cat "$<" | sed -e 's/@VERSION@/$(VERSION)/g' > "$@"
