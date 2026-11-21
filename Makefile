# This Makefile is only intended to prepare for distribution the PyTables
# sources exported from a repository.  For building and installing PyTables,
# please use ``setup.py`` as described in the ``README.rst`` file.

PYTHON=python3
MD5SUM=md5sum
TARGET=tables
VERSION=$(shell grep "__version__ =" $(TARGET)/_version.py | cut -f 3 -d ' ' | sed s/\"//g)
GENERATED=ANNOUNCE.txt
CYTHONIZED=tables/_comp_*.c tables/*extension.c

.PHONY: default help dist check clean cleaner distclean \
        lint docs ext wheels \
        html latex fulldist heavycheck parallelcheck requirements clean-requirements

default: help

help:
	@echo "Usage: make <TARGET>"
	@echo "Available targets:"
	@echo "  help      - print this help message"
	@echo "  dist      - generate the distribution packages (source and wheel)"
	@echo "  check     - run standard test suite"
	@echo "  clean     - clean build artifacts"
	@echo "  cleaner   - clean cache files and working directories of al tools"
	@echo "  distclean - clean all the generated files"
	@echo "  lint      - perform check with code linter (flake8, black)"
	@echo "  docs      - generate the sphinx documentation"
	@echo "  api       - update the API source files in the documentation"
	@echo "  ext       - build Python extensions in-place"
	@echo "  wheels    - build Python wheels"
	@echo "  html      - build HTML documentation"
	@echo "  latex     - build PDF/LaTeX documentation"
	@echo "  fulldist  - build the full distribution for a release (sdizt, doc, ANNOUNCE, etc.)"
	@echo "  heavycheck         - run the full test suite"
	@echo "  parallelcheck      - run the parallel test suite"
	@echo "  requirements       - run the parallel test suite"
	@echo "  clean-requirements - run the parallel test suite"

dist: clean
	$(PYTHON) -m build --sdist
	$(PYTHON) -m twine check dist/*.tar.gz
	# Cython output is tied to the NumPy of the isolated build env,
	# so don't let the in-tree build reuse it.
	$(RM) $(CYTHONIZED)

check: ext
	env PYTHONPATH=. $(PYTHON) -m pytest --doctest-only --pyargs $(TARGET) -k "not AttributeSet"
	env PYTHONPATH=. $(PYTHON) -Werror tests/test_all.py

clean:
	$(RM) MANIFEST ANNOUNCE.txt
	$(RM) -r *.*-info build
	find . -name __pycache__ -type d -exec $(RM) -r {} +
	# $(RM) -r __pycache__ */__pycache__ */*/__pycache__ */*/*/__pycache__
	$(RM) $(TARGET)/*.c $(TARGET)/*.cpp $(TARGET)/*.so $(TARGET)/*.o
	if [ -f doc/Makefile ] ; then $(MAKE) -C doc clean; fi
	$(RM) -r doc/build
	find . '(' -name '*.py[co]' -o -name '*~' ')' -not -path './.pixi/*' -exec rm '{}' ';'
	$(RM) bench/*.h5 bench/*.prof
	$(RM) -r examples/*.h5 examples/raw
	$(RM) -r tmp a.out

cleaner: clean
	$(RM) -r .coverage htmlcov
	$(RM) -r .pytest_cache
	$(RM) -r .tox
	$(RM) -r .mypy_cache
	$(RM) -r .ruff_cache
	$(RM) -r .ipynb_checkpoints
	$(RM) -r .hypothesis

distclean: cleaner
	$(RM) -r dist

lint:
	$(PYTHON) -m flake8 --count --statistics $(TARGET) tests examples
	$(PYTHON) -m pydocstyle --count $(TARGET)
	$(PYTHON) -m isort --check $(TARGET) tests examples
	$(PYTHON) -m black --check $(TARGET) tests examples
	# $(PYTHON) -m mypy --check-untyped-defs --ignore-missing-imports $(TARGET) tests examples
	ruff check $(TARGET) tests examples
	# ruff format --check $(TARGET) tests examples
	codespell
	# cython-lint
	$(PYTHON) ci/check_requirements_sync.py

docs: ext html latex
	# $(MAKE) PYTHONPATH=${PWD} -C doc html
	# $(MAKE) PYTHONPATH=${PWD} -C doc doctest
	$(MAKE) PYTHONPATH=${PWD} -C doc linkcheck
	# $(MAKE) PYTHONPATH=${PWD} -C doc spelling

ext:
	$(PYTHON) setup.py build_ext --inplace

wheels:
	# Requires docker
	python3 -m cibuildwheel --platform auto

html: ext
	$(MAKE) PYTHONPATH=${PWD} -C doc html

latex: ext
	$(MAKE) PYTHONPATH=${PWD} -C doc latexpdf

fulldist: clean dist $(GENERATED) html latex
	cp ANNOUNCE.txt dist/ANNOUNCE-$(VERSION).txt
	cp RELEASE_NOTES.rst dist/RELEASE_NOTES-$(VERSION).rst
	cp doc/build/latex/usersguide-$(VERSION).pdf dist/pytablesmanual-$(VERSION).pdf
	tar cvzf dist/pytablesmanual-$(VERSION)-html.tar.gz doc/build/html
	cd dist && \
	$(MD5SUM) -b tables-$(VERSION).tar.gz RELEASE_NOTES-$(VERSION).rst \
	pytablesmanual-$(VERSION).pdf \
	pytablesmanual-$(VERSION)-html.tar.gz > pytables-$(VERSION).md5 && \
	cd -

heavycheck: ext
	env PYTHONPATH=. $(PYTHON) tests/test_all.py --heavy

parallelcheck: ext
	env PYTHONPATH=. $(PYTHON) tests/run_ft.py --max-tests=400

requirements: \
	requirements.txt \
	requirements-docs.txt \
	.github/workflows/requirements/build-requirements.txt \
	.github/workflows/requirements/wheels-requirements.txt

clean-requirements:
	$(RM) \
		requirements.txt \
		requirements-docs.txt \
		.github/workflows/requirements/build-requirements.txt \
		.github/workflows/requirements/wheels-requirements.txt

ANNOUNCE.txt: ANNOUNCE.txt.in $(TARGET)/__init__.py
	cat "$<" | sed -e 's/@VERSION@/$(VERSION)/g' > "$@"

%.txt: %.in
	pip-compile -U --allow-unsafe --generate-hashes --strip-extras $<
	# uv pip compile --python 3.11 -U --allow-unsafe --generate-hashes --strip-extras -o $@ $<
