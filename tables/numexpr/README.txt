This is an enhanced version of numexpr, a "Fast numerical expression
evaluator", originally conceived and implemented by David Cooke with
contributions of Tim Hochberg.

This version provides several enhancements:

- Addition of a boolean type. This allows better array copying times
for large arrays (lightweight computations are typically bound by
memory bandwidth).

- Enhanced performance for strided and unaligned data, specially for
lightweigth computations (e.g. 'a>10'). With this and the addition of
the boolean type, we can get up to 2x better times than previous
versions. Also, most of the supported computations go faster than with
numpy or numarray, even the simplest ones.

- Addition of ~, & and | operators (a la numarray.where)

- Support for both numpy and numarray (use the flag --force-numarray
in setup.py).

- Added a new benchmark for testing boolean expressions and
strided/unaligned arrays: boolean_timing.py

Things that I want to address in the future:

- Add tests on strided and unaligned data (currently only tested manually)

- Add types for int16, int64 (in 32-bit platforms), float32,
  complex64 (simple prec.)

Enjoy!

Francesc Altet
faltet@carabos.com
