# This Makefile is only intended to prepare for distribution the PyTables
# sources exported from a repository.  For building and installing PyTables,
# please use ``setup.py`` as described in the ``README.rst`` file.

VERSION = $(shell cat VERSION)
SRCDIRS = src doc
GENERATED = ANNOUNCE.txt
PYTHON = python
PYPLATFORM = $(shell $(PYTHON) -c "from distutils.util import get_platform; print(get_platform())")
PYVER = $(shell $(PYTHON) -V 2>&1 | cut -c 8-10)
PYBUILDDIR = $(PWD)/build/lib.$(PYPLATFORM)-$(PYVER)
OPT = PYTHONPATH=$(PYBUILDDIR)


.PHONY:		all dist build check heavycheck clean distclean html

all:		$(GENERATED) build
	for srcdir in $(SRCDIRS) ; do $(MAKE) -C $$srcdir $(OPT) $@ ; done

dist:		all
	$(PYTHON) setup.py sdist
	cd dist && md5sum tables-$(VERSION).tar.gz > pytables-$(VERSION).md5 && cd -
	cp RELEASE_NOTES.txt dist/RELEASE_NOTES-$(VERSION).txt
	for srcdir in $(SRCDIRS) ; do $(MAKE) -C $$srcdir $(OPT) $@ ; done

clean:
	rm -rf MANIFEST build dist tmp tables/__pycache__
	rm -rf bench/*.h5 bench/*.prof
	rm -rf examples/*.h5 examples/raw
	rm -f $(GENERATED) tables/*.so a.out
	find . '(' -name '*.py[co]' -o -name '*~' ')' -exec rm '{}' ';'
	for srcdir in $(SRCDIRS) ; do $(MAKE) -C $$srcdir $(OPT) $@ ; done

distclean:	clean
	for srcdir in $(SRCDIRS) ; do $(MAKE) -C $$srcdir $(OPT) $@ ; done
	rm -f tables/_comp_*.c tables/*extension.c
	#git clean -fdx

html: build
	$(MAKE) -C doc $(OPT) html

%:		%.in VERSION
	cat "$<" | sed -e 's/@VERSION@/$(VERSION)/g' > "$@"

build:
	$(PYTHON) setup.py build

check: build
	cd build/lib.*-$(PYVER) && env PYTHONPATH=. $(PYTHON) tables/tests/test_all.py

heavycheck: build
	cd build/lib.*-$(PYVER) && env PYTHONPATH=. $(PYTHON) tables/tests/test_all.py --heavy
