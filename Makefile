# This Makefile is only intended to prepare for distribution the PyTables
# sources exported from a repository.  For building and installing PyTables,
# please use ``setup.py`` as described in the ``README.txt`` file.

SRCDIRS = src doc

.PHONY:		dist clean

dist clean:
	for srcdir in $(SRCDIRS) ; do (cd $$srcdir && make $@) ; done
