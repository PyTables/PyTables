# This Makefile is only intended to prepare for distribution the PyTables
# sources exported from a repository.  For building and installing PyTables,
# please use ``setup.py`` as described in the ``README.rst`` file.

VERSION = $(shell grep "__version__ =" tables/_version.py | cut -f 3 -d ' ' | sed s/\"//g)
SRCDIRS = doc
GENERATED = ANNOUNCE.txt
PYTHON = python3
PYPLATFORM = $(shell $(PYTHON) -c "from sysconfig import get_platform; print(get_platform())")
PYVER = $(shell $(PYTHON) -c "import sys; print(sys.implementation.cache_tag)")
PYBUILDDIR = $(PWD)/build/lib.$(PYPLATFORM)-$(PYVER)
OPT = PYTHONPATH="$(PYBUILDDIR)"
MD5SUM = md5sum


.PHONY: default dist sdist build check heavycheck clean distclean html latex requirements

default: $(GENERATED) build

dist: sdist html latex
	cp RELEASE_NOTES.rst dist/RELEASE_NOTES-$(VERSION).rst
	cp doc/build/latex/usersguide-$(VERSION).pdf dist/pytablesmanual-$(VERSION).pdf
	tar cvzf dist/pytablesmanual-$(VERSION)-html.tar.gz doc/html
	cd dist && \
	$(MD5SUM) -b tables-$(VERSION).tar.gz RELEASE_NOTES-$(VERSION).rst \
	pytablesmanual-$(VERSION).pdf \
	pytablesmanual-$(VERSION)-html.tar.gz > pytables-$(VERSION).md5 && \
	cd -

sdist: $(GENERATED)
	# $(RM) -r MANIFEST tables/__pycache__ tables/*/__pycache__
	# $(RM) tables/_comp_*.c tables/*extension.c
	# $(RM) tables/*.so
	$(PYTHON) -m build --sdist

clean:
	$(RM) -r MANIFEST build dist tmp tables/__pycache__ doc/_build
	$(RM) bench/*.h5 bench/*.prof
	$(RM) -r examples/*.h5 examples/raw
	$(RM) -r *.egg-info
	$(RM) $(GENERATED) tables/*.so a.out
	find . '(' -name '*.py[co]' -o -name '*~' ')' -exec rm '{}' ';'
	for srcdir in $(SRCDIRS) ; do $(MAKE) $(OPT) -C $$srcdir $@ ; done

distclean: clean
	$(RM) tables/_comp_*.c tables/*extension.c
	$(RM) doc/usersguide-*.pdf
	$(RM) -r doc/html
	$(RM) -r .pytest_cache .mypy_cache
	# git clean -fdx

html: build
	$(MAKE) $(OPT) -C doc html
	$(RM) -r doc/html
	cp -R doc/build/html doc/html

latex: build
	$(MAKE) $(OPT) -C doc latexpdf
	$(RM) doc/usersguide-*.pdf
	cp doc/build/latex/usersguide-$(VERSION).pdf doc

ANNOUNCE.txt: ANNOUNCE.txt.in tables/__init__.py
	cat "$<" | sed -e 's/@VERSION@/$(VERSION)/g' > "$@"

build:
	$(PYTHON) setup.py build

check: build
	# cd build/lib.* && env PYTHONPATH=. $(PYTHON) -m pytest --doctest-only --pyargs tables -k "not AttributeSet"
	cd build/lib.* && env PYTHONPATH=. $(PYTHON) tables/tests/test_all.py

heavycheck: build
	cd build/lib.* && env PYTHONPATH=. $(PYTHON) tables/tests/test_all.py --heavy

requirements: \
	requirements.txt \
	requirements-docs.txt \
	.github/workflows/requirements/build-requirements.txt \
	.github/workflows/requirements/wheels-requirements.txt

%.txt: %.in
	pip-compile -U --allow-unsafe --generate-hashes --strip-extras $<
