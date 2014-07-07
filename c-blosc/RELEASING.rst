================
Releasing Blosc
================

:Author: Francesc Alted
:Contact: francesc@blosc.io
:Date: 2014-01-15


Preliminaries
-------------

- Make sure that ``RELEASE_NOTES.rst`` and ``ANNOUNCE.rst`` are up to
  date with the latest news in the release.

- Check that *VERSION* symbols in blosc/blosc.h contains the correct info.

Testing
-------

Create a new build/ directory, change into it and issue::

  $ cmake ..
  $ make
  $ make test

To actually test Blosc the hard way, look at the end of:

http://blosc.org/trac/wiki/SyntheticBenchmarks

where instructions on how to intensively test (and benchmark) Blosc
are given.


Tagging
-------

- Create a tag ``X.Y.Z`` from ``master``.  Use the next message::

    $ git tag -a vX.Y.Z -m "Tagging version X.Y.Z"

- Push the tag to the github repo::

    $ git push --tags


Announcing
----------

- Update the release notes in the github wiki:

https://github.com/Blosc/c-blosc/wiki/Release-notes

- Send an announcement to the blosc, pytables, carray and
  comp.compression lists.  Use the ``ANNOUNCE.rst`` file as skeleton
  (possibly as the definitive version).

Post-release actions
--------------------

- Edit *VERSION* symbols in blosc/blosc.h in master to increment the
  version to the next minor one (i.e. X.Y.Z --> X.Y.(Z+1).dev).

- Create new headers for adding new features in ``RELEASE_NOTES.rst``
  and empty the release-specific information in ``ANNOUNCE.rst`` and
  add this place-holder instead:

  #XXX version-specific blurb XXX#


That's all folks!


.. Local Variables:
.. mode: rst
.. coding: utf-8
.. fill-column: 70
.. End:
