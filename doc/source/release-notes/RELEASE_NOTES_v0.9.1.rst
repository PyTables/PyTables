What's new in PyTables 0.9.1
----------------------------

This release is mainly a maintenance version. In it, some bugs has
been fixed and a few improvements has been made. One important thing
is that chunk sizes in EArrays has been re-tuned to get much better
performance. Besides, it has been tested against the latest Python 2.4
and all unit tests seems to pass fine.

More in detail:

Improvements:

- The chunksize computation for EArrays has been re-tuned to allow the
  compression rations that were usual before 0.9 release.

- New --unpackshort and --quantize flags has been added to nctoh5
  script. --unpackshort unpack short integer variables to float
  variables using scale_factor and add_offset netCDF variable
  attributes. --quantize quantize data to improve compression using
  least_significant_digit netCDF variable attribute (not active by
  default).  See
  http://www.esrl.noaa.gov/psd/data/gridded/conventions/cdc_netcdf_standard.shtml
  for further explanation of what this attribute means. Thanks to Jeff
  Whitaker for providing this.

- Table.itersequence has received a new parameter called "sort". This
  allows to disable the sorting of the sequence in case the user wants
  so.

Backward-incompatible changes:

- Now, the AttributeSet class throw an AttributeError on __getattr__
  for nonexistent attributes in it. Formerly, the routine returned
  None, which is pretty much against convention in Python and breaks
  the built-in hasattr() function. Thanks to Robert Nemec for noting
  this and offering a patch.

- VLArray.read() has changed its behaviour. Now, it always returns a
  list, as stated in documentation, even when the number of elements
  to return is 0 or 1. This is much more consistent when representing
  the actual number of elements on a certain VLArray row.

API additions:

- A Row.getTable() has been added. It is an accessor for the associated
  Table object.

- A File.copyAttrs() has been added. It allows copying attributes from
  one leaf to other. Properly speaking, this was already there, but not
  documented :-/

Bug fixes:

- Now, the copy of hierarchies works even when there are scalar Arrays
  (i.e. Arrays which shape is ()) on it. Thanks to Robert Nemec for
  providing a patch.

- Solved a memory leak regarding the Filters instance associated with
  the File object, that was not released after closing the file. Now,
  there are no known leaks on PyTables itself.

- Improved security of nodes name checking. Closes #1074335


Enjoy data!,

-- Francesc Altet
falted@pytables.org

