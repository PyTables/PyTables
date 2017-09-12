# -*- coding: utf-8 -*-

########################################################################
#
# License: BSD
# Created: November 5, 2010
# Author:  Francesc Alted - faltet@pytables.com
#
########################################################################

"""Required versions for PyTables dependencies."""

from distutils.version import LooseVersion

#**********************************************************************
#  Keep these in sync with requirements.txt and user's guide
#**********************************************************************

# Minimum recommended versions for mandatory packages
min_numpy_version = LooseVersion('1.8.0')
min_numexpr_version = LooseVersion('2.5.2')
min_hdf5_version = LooseVersion('1.8.4')
min_h5py_version = LooseVersion('2.7')
min_blosc_version = LooseVersion("1.4.1")
min_blosc_bitshuffle_version = LooseVersion("1.8.0")
"""The minumum Blosc version where BitShuffle can be used safely."""
