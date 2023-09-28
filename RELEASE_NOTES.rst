=======================================
 Release notes for PyTables 3.9 series
=======================================

:Author: PyTables Developers
:Contact: pytables-dev@googlegroups.com

.. py:currentmodule:: tables


Changes from 3.8.0 to 3.9.0
===========================

New features
------------

- Apply optimized slice read to Blosc2-compressed `CArray` and `EArray`, with
  Blosc2 NDim 2-level partitioning for multidimensional arrays (:PR:`1056`).
  See "Multidimensional slicing and chunk/block sizes" in the User's Guide.
  Thanks to Marta Iborra and Ivan Vilata.  This development was funded by a
  NumFOCUS grant.
- Add basic API for column-level attributes as `Col._v_col_attrs` (:PR:`893`
  and :issue:`821`).  Thanks to Jonathan Wheeler, Thorben Menne, Ezequiel
  Cimadevilla Alvarez, odidev, Sander Roet, Antonio Valentino, Munehiro
  Nishida, Zbigniew Jędrzejewski-Szmek, Laurent Repiton, xmatthias, Logan
  Kilpatrick.

Other changes
-------------

- Add support for the forthcoming Python 3.12 with binary wheels and automated
  testing.
- Drop wheels and automated testing for Python 3.8; users or distributions may
  still build and test with Python 3.8 on their own (see :commit:`ae1e60e` and
  :commit:`47f5946`).
- New benchmark for ERA5 climate data.  Thanks to Óscar Guiñón.
- New "100 trillion baby" benchmark.  Thanks to Francesc Alted.
- New benchmark for querying meteorologic data.  Thanks to Francesc Alted.

Improvements
------------

- Use `H5Dchunk_iter` (when available) to speed up walking over many chunks in
  a very large table, as well as with random reads (:issue:`991`, :PR:`997`,
  :PR:`999`).  Thanks to Francesc Alted and Mark Kittisopikul.
- Improve `setup.py` (now using `pyproject.toml` as per PEP 518) and `blosc2`
  discovery mechanism.  Blosc2 may be used both via python-blosc2 or system
  c-blosc2 (:PR:`987`, :PR:`1000`, :issue:`998`, :PR:`1017`,
  :PR:`1045`). Thanks to Antonio Valentino, Ben Greiner, Iwo-KX, nega.
- Enable compatibility with Cython 3 (:PR:`1008` and :issue:`1003`).  Thanks
  to Matus Valo and Michał Górny.
- Set GitHub workflow permissions to least privileges (:PR:`1007`).  Thanks to
  Joyce Brum.
- Add `SECURITY.md` with security policy (:PR:`1012` and :issue:`1011`).
  Thanks to Joyce Brum.
- Handle py-cpuinfo missing in some platforms (:PR:`1013`).  Thanks to Sam
  James.
- Avoid NumPy >= 1.25 deprecations, use `numpy.all`, `numpy.any`,
  etc. instead.  Thanks to Antonio Valentino.
- Avoid C-related build warnings.  Thanks to Antonio Valentino.
- Update included c-blosc to v1.21.5 (fixes SSE2/AVX build issue).
- Require python-blosc2 >= 2.2.8 or c-blosc2 >= 2.10.4 (Python 3.12 support
  and assorted fixes).
- Update external libraries for CI-based wheel builds (:PR:`1018` and
  :issue:`967`):

  * hdf5 v1.14.2
  * lz4 v1.9.4
  * zlib v1.2.13

Bugfixes
--------

- Fix crash in Blosc2 optimized path with large tables (:issue:`995` and
  :PR:`996`).  Thanks to Francesc Alted.
- Fix compatibility with NumExpr v2.8.5 (:PR:`1046`).  Thanks to Antonio
  Valentino.
- Fix build errors on Windows ARM64 (:PR:`989`).  Thanks to Cristoph Gohlke.
- Fix `ptrepack` failures with external links (:issue:`938` and :PR:`990`).
  Thanks to Adrian Altenhoff.
- Replace stderr messages with Python warnings (:issue:`992` and :PR:`993`).
  Thanks to Maximilian Linhoff.
- Fixes to CI workflow and wheel building (:PR:`1009`, :PR:`1047`).  Thanks to
  Antonio Valentino.
- Fix garbled rendering of `File.get_node` docstring (:PR:`1021`).  Thanks to
  Steffen Rehberg.
- Fix open `extern "C"` block (:PR:`1026`).  Thanks to Ivan Vilata.
- Fix Cython slice indexing under Python 3.12 (:PR:`1033`).  Thanks to
  Zbigniew Jędrzejewski-Szmek.
- Fix unsafe temporary file creation in benchmark (:PR:`1053`).  Thanks to Al
  Arafat Tanin (Project Alpha-Omega).

Thanks
------

In alphabetical order:

- Adrian Altenhoff
- Al Arafat Tanin
- Antonio Valentino
- Ben Greiner
- Cristoph Gohlke
- Ezequiel Cimadevilla Alvarez
- Francesc Alted
- Ivan Vilata
- Iwo-KX
- Jonathan Wheeler
- Joyce Brum
- Laurent Repiton
- Logan Kilpatrick
- Mark Kittisopikul
- Marta Iborra
- Matus Valo
- Maximilian Linhoff
- Michał Górny
- Munehiro Nishida
- nega
- odidev
- Óscar Guiñón
- Sam James
- Sander Roet
- Seth Troisi
- Steffen Rehberg
- Thorben Menne
- xmatthias
- Zbigniew Jędrzejewski-Szmek
