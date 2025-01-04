========================================
 Release notes for PyTables 3.10 series
========================================

:Author: PyTables Developers
:Contact: pytables-dev@googlegroups.com

.. py:currentmodule:: tables

Changes from 3.10.2 to 3.10.3
=============================

* TBW


Changes from 3.10.1 to 3.10.2
=============================

Improvements
------------

- Wheels for Python v3.13 are now provided (:issue:`1217`).
- Convert HDF5-Blosc sources into Git submodule (:issue:`1197`).
- Complete code re-formatting and improvement of variables naming.
  Now the entire codebase is fully PEP8 compliant and regularly checked in
  CI with `black`_, `flake8`_ and `isort`_ (:issue:`867`).
- The automatic build of the documentation has been re-worked.
- Use `numpy.typing.DTypeLike` over `np.dtype` for parameters.
  Thanks to Joel T. Collins
- Accept IsDescription, dict, `numpy.dtype` as table descriptions.
  Thanks to Joel T. Collins
- Allow multi-dimension `chunkshape` when creating arrays.
  Thanks to Joel T. Collins
- Improve the way `setup.py` finds runtime libs (see :issue:`1219` and
  :issue:`1223`). Thanks to Jason Bacon.

Bugfixes
--------

- Fixed blosc2 search paths.
- Fixed the copy of tables with `createparents=True` (:issue:`1208`).
- Fixed links in `README.rst` (:issue:`1218`). Thanks to David Poznik.
- Fixed the function for writing cpu info to cache file (:issue:`#1222`).
  Thanks to Adrian Altenhoff.
- Fixed an incorrect access to the obsolete `sys.maxint` in `pttree`
  (:issue:`1224`).

.. _black: https://github.com/psf/black
.. _flake8: https://flake8.pycqa.org
.. _isort: https://pycqa.github.io/isort


Changes from 3.10.0 to 3.10.1
=============================

Bugfixes
--------

- Fix version constraints for the ``numpy`` runtime requirements
  (:issue:`1204`).
  For a mistake it didn't allow to use PyTables with ``numpy`` 2.x.
- Fix compatibility with PyPy (:issue:`1205`), Thanks to Michał Górny.


Improvements
------------

- Enforce `numpy >= 2` as build constraint (see discussion in :issue:`1200`).
- Use tuple of plain ints for chunk info coordinates.
  Thanks to Ivan Vilata-i-Balaguer.
- Enable `faulthandler` in `tables.tests.test_all`. Thanks to Eric Larson.


Changes from 3.9.2 to 3.10.0
============================

New features
------------

- New direct chunking API which allows access to raw chunk data skipping the
  HDF5 filter pipeline (cutting overhead, see "Optimization tips" in User's
  Guide), as well as querying chunk information (:PR:`1187`).  Thanks to Ivan
  Vilata and Francesc Alted.  This development was funded by a NumFOCUS grant.

Improvements
------------

- This release is finally compatible with NumPy 2, with wheels being built
  against it so that they are still binary-compatible with NumPy 1
  installations (:PR:`1176`, :PR:`1183`, :PR:`1184`, :PR:`1192`, :PR:`1195`,
  :issue:`1160`, :issue:`1172`, :issue:`1185`).  NumPy >= 1.20 is required
  now.  Thanks to Antonio Valentino, Maximilian Linhoff and Eric Larson.
- Fix compatibility with Python 3.13 (:issue:`1166`), Python >= 3.10 required.
  Cython 3.0.10 is required for building.  Thanks to Antonio Valentino.
- Add type hints to `atom.py` (:PR:`1079`).  This also narrows some types,
  only allowing bytes to be stored in `VLStringAtom` and only str in
  `VLUnicodeAtom`.  Thanks to Nils Carlson.
- Add type hints to (hopefully) the complete PyTables API (:PR:`1119`,
  :PR:`1120`, :PR:`1121`, :PR:`1123`, :PR:`1124`, :PR:`1125`, :PR:`1125`,
  :PR:`1126`, :PR:`1128`, :PR:`1129`, :PR:`1130`, :PR:`1131`, :PR:`1132`,
  :PR:`1133`, :PR:`1135`, :PR:`1136`, :PR:`1137`, :PR:`1138`, :PR:`1139`,
  :PR:`1140`, :PR:`1141`, :PR:`1142`, :PR:`1143`, :PR:`1145`, :PR:`1146`,
  :PR:`1147`, :PR:`1148`, :PR:`1150`, :PR:`1151`, :PR:`1152`).  Thanks to Ko
  Stehner.
- Reduce impact of CPU information gathering by caching in local file
  (:PR:`1091`, :PR:`1118`, :issue:`1081`).  Thanks to Antti Mäkinen and
  Maximilian Linhoff.

Bugfixes
--------

- Fix Windows AMD64 build issues with Bzip2 and C-Blosc2 libraries
  (:issue:`1188`).  Thanks to Antonio Valentino and Eric Larson.
- Fix typos and may other language errors in docstrings (:PR:`1122`).  Thanks
  to Ko Stehner.
- Fix Blosc2 filter not setting `dparams.schunk` on decompression (:PR:`1110`
  and :issue:`1109`).  Thanks to Tom Birch.
- Fix using B2ND optimizations when Blosc2 is not the only enabled filter;
  move Fletcher32 compression to end of pipeline when enabled (:PR:`1191` and
  :issue:`1162`).  Thanks to Ivan Vilata and Alex Laslavic.
- Fix broken internal passing of `createparents` argument in `Leaf.copy`
  (:PR:`1127` and :issue:`1125`).  Thanks to Ko Stehner.
- Re-enable relative paths in `ExternalLink` class (:PR:`1095`).  Thanks to
  erikdl-zeiss.
- Fix using prefix in heavy tests methods of `test_queries` (:PR:`1169`).
  Thanks to Miro Hrončok.
- Fix `TypeError` when computing Blosc2 search paths with missing library
  (:PR:`1188` and :issue:`1100`).  Thanks to martinowitsch, Padraic Calpin and
  Eric Larson.
- Avoid overflow `RuntimeWarning` on NumPy `expectedrows` value (:PR:`1010`).
  Thanks to wony-zheng and Ivan Vilata.

Other changes
-------------

- Add wheels for macOS ARM64 (Apple Silicon), set `MACOSX_DEPLOYMENT_TARGET`
  in Docker (:PR:`1050` and :issue:`1165`).  Thanks to Clemens Brunner,
  Antonio Valentino, Maximilian Linhoff and Eric Larson.
- Avoid illegal hardware instruction under macOS on M1/M2 with Rosetta and
  AMD64 wheels (:PR:`1195` and :issue:`1186`).  Thanks to Antonio Valentino
  and Jon Peirce.
- Produce nightly wheels (with HDF5 1.14.4), also uploaded to Scientific
  Python Anaconda repo.  Wheels are also produced for PR workflows.  Thanks to
  Antonio Valentino and Eric Larson (:PR:`1175`).
- Wheels are no longer linked with the LZO library to avoid licensing issues
  (:PR:`1195`).  Thanks to Antonio Valentino.
- Hash-pin dependencies on wheel workflows to increase build procedure
  security, with support for Dependabot and Renovatebot updates (:PR:`1085`
  and :issue:`1015`).  Thanks to Joyce Brum and Diogo Teles Sant'Anna.
- Hash-pin GitHub action versions in wheels workflow.  Thanks to Antonio
  Valentino.
- Update ReadTheDocs configuration to version 2 (:PR:`1092`).  Thanks to
  Maximilian Linhoff.
- Assorted fixes to b2nd benchmark, with new results.  Thanks to Ivan Vilata.
- Point users to example code to handle "Selection lists cannot have repeated
  values" exception (:PR:`1161` and :issue:`1149`).  Thanks to Joshua Albert.
- Remove unused `getLibrary` C code.  Thanks to Antonio Valentino.
- Update included C-Blosc to 1.21.6 (:PR:`1193`).  Thanks to Ivan Vilata.
- Update included HDF5-Blosc filter to 1.0.1 (:PR:`1194`).  Thanks to Ivan
  Vilata.

Thanks
------

In alphabetical order:

- Alex Laslavic
- Antonio Valentino
- Antti Mäkinen
- Clemens Brunner
- Diogo Teles Sant'Anna
- Eric Larson
- erikdl-zeiss
- Francesc Alted
- Ivan Vilata
- Jon Peirce
- Joshua Albert
- Joyce Brum
- Ko Stehner
- martinowitsch
- Maximilian Linhoff
- Miro Hrončok
- Nils Carlson
- Padraic Calpin
- Tom Birch
- wony-zheng
