# This Makefile is only intended to prepare for distribution the PyTables
# sources exported from a repository.  For building and installing PyTables,
# please use ``setup.py`` as described in the ``README.txt`` file.

VERSION = $(shell cat VERSION)
SRCDIRS = src doc

GENERATED = ANNOUNCE.txt


.PHONY:		dist clean


dist:		$(GENERATED)
	for srcdir in $(SRCDIRS) ; do $(MAKE) -C $$srcdir $@ ; done

clean:
	rm -rf MANIFEST build dist
	rm -f $(GENERATED) tables/*.so tables/numexpr/*.so
	find . '(' -name '*.py[co]' -o -name '*~' ')' -exec rm '{}' ';'
	for srcdir in $(SRCDIRS) ; do $(MAKE) -C $$srcdir $@ ; done

%:		%.in VERSION
	cat "$<" | sed -e 's/@VERSION@/$(VERSION)/g' > "$@"
