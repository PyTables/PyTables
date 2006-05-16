# This Makefile is only intended to prepare for distribution the PyTables
# sources exported from a repository.  For building and installing PyTables,
# please use ``setup.py`` as described in the ``README.txt`` file.

VERSION = $(shell cat VERSION)
SRCDIRS = src doc

GENERATED = ANNOUNCE.txt RELEASE-NOTES.txt


.PHONY:		dist clean


dist:		$(GENERATED)
	for srcdir in $(SRCDIRS) ; do (cd $$srcdir && make $@) ; done

clean:
	-rm -rf build dist
	-rm $(GENERATED) tables/*.so
	find . '(' -name '*.py[co]' -o -name '*~' ')' -exec rm '{}' ';'
	for srcdir in $(SRCDIRS) ; do (cd $$srcdir && make $@) ; done

%:		%.in VERSION
	cat "$<" | sed -e 's/@VERSION@/$(VERSION)/g' > "$@"
