=======================================
 Release notes for PyTables 3.4 series
=======================================

Changes from 3.4.3 to 3.4.4
===========================

Improvements
------------
 - Environment variable to control the use of embedded libraries.
   Thanks to avalentino.
 - Include citation in repository.
   :issue:`690`. Thanks to katrinleinweber.

Bugs fixed
----------
 - Fixed import error with numexpr 2.6.5.dev0
   :issue:`685`. Thanks to cgohlke.
 - Fixed linter warnings.
   Thanks to avalentino.
 - Fixed for re.split() is version detection.
   :issue:`687`. Thanks to mingwandroid.
 - Fixed test failures with Python 2.7 and NumPy 1.14.3
   :issue:`688` & :issue:`689`. Thanks to oleksandr-pavlyk.


Changes from 3.4.2 to 3.4.3
===========================

Improvements
------------
 - On interactive python sessions, group/attribute  `__dir__()` method
   autocompletes children that are named as valid python identifiers.
   :issue:`624` & :issue:`625` thanks to ankostis.
 - Implement `Group.__getitem__()` to have groups act as python-containers,
   so code like this works: ``hfile.root['some child']``.
   :issue:`628` thanks to ankostis.
 - Enable building with Intel compiler (icc/icpc).
   Thanks to rohit-jamuar.
 - PEP 519 support, using new `os.fspath` method.
   Thanks to mruffalo.
 - Optional disable recording of ctime (metadata creation time) when
   creating datasets that makes possible to get bitwise identical output
   from repeated runs.
   Thanks to alex-cobb.
 - Prevent from reading all rows for each coord in a VLArray when
   indexing using a list .
   Thanks to igormq.
 - Internal Blosc version updated to 1.14.3

Bugs fixed
----------
 - Fixed division by zero when using `_convert_time64()` with an empty
   nparr array.
   :issue:`653`. Thanks to alobbs.
 - Fixed deprecation warnings with numpy 1.14.
   Thanks to oleksandr-pavlyk.
 - Skip DLL check when running from a frozen app.
   :issue:`675`. Thanks to jwiggins.
 - Fixed behaviour with slices out of range.
   :issue:`651`. Thanks to jackdbd.


Changes from 3.4.1 to 3.4.2
===========================

Improvements
------------
 - setup.py detects conda env and uses installed conda (hdf5, bzip2, lzo
   and/or blosc) packages when building from source.

Bugs fixed
----------
 - Linux wheels now built against built-in blosc.
 - Fixed windows absolute paths in ptrepack, ptdump, ptree.
   :issue:`616`. Thanks to oscar6echo.


Changes from 3.4.0 to 3.4.1
===========================

Bugs fixed
----------
 - Fixed bug in ptrepack


Changes from 3.3.0 to 3.4.0
===========================

Improvements
------------
 - Support for HDF5 v1.10.x (see :issue:`582`)
 - Fix compatibility with the upcoming Python 2.7.13, 3.5.3 and 3.6 versions.
   See also :issue:`590`. Thanks to Yaroslav Halchenko
 - Internal Blosc version updated to 1.11.3
 - Gracefully handle cpuinfo failure. (PR #578)
   Thanks to Zbigniew Jędrzejewski-Szmek
 - Update internal py-cpuinfo to 3.3.0. Thanks to Gustavo Serra Scalet.

Bugs fixed
----------
 - Fix conversion of python 2 `long` type to `six.integer_types` in atom.py.
   See also :issue:`598`. Thanks to Kyle Keppler for reporting.
 - Fix important bug in bitshuffle filter in internal Blosc on big-endian
   machines. See also :issue:`583`.
 - Fix allow for long type in nextafter. (PR #587) Thanks to Yaroslav Halchenko.
 - Fix unicode bug in group and tables names. :issue:`514`
