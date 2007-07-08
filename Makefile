# This Makefile is only intended to prepare for distribution the PyTables
# sources exported from a repository.  For building and installing PyTables,
# please use ``setup.py`` as described in the ``README.txt`` file.

VERSION = $(shell cat VERSION)
SVER = $(strip $(patsubst %pro, %, $(VERSION)))
SRCDIRS = src doc
LICENSES = personal site development
DEBFILES = debian/control debian/changelog debian/rules

GENERATED = ANNOUNCE.txt


.PHONY:		dist clean


dist:		$(GENERATED)
	for srcdir in $(SRCDIRS) ; do $(MAKE) -C $$srcdir $@ ; done
	for license in $(LICENSES) ; do \
	    cp LICENSE-$$license.txt LICENSE.txt ; \
	    for f in $(DEBFILES) ; do \
	        cat $$f.in | sed -e "s/@LICENSE@/$$license/g" > $$f ; \
                if [ $$f == "debian/rules" ] ; then \
	            chmod 755 $$f ; \
		fi ; \
	    done ; \
	    python setup.py sdist ; \
	    mv dist/tables-$(VERSION).tar.gz \
               dist/pytables-pro-$$license-$(SVER).tar.gz ; \
	done

clean:
	rm -rf LICENSE.txt MANIFEST build dist
	rm -f $(GENERATED) tables/*.so tables/numexpr/*.so
	find . '(' -name '*.py[co]' -o -name '*~' ')' -exec rm '{}' ';'
	for srcdir in $(SRCDIRS) ; do $(MAKE) -C $$srcdir $@ ; done

%:		%.in VERSION
	cat "$<" | sed -e 's/@VERSION@/$(VERSION)/g' > "$@"
