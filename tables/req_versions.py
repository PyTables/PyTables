# -*- coding: utf-8 -*-

########################################################################
#
# License: BSD
# Created: November 5, 2010
# Author:  Francesc Alted - faltet@pytables.com
#
########################################################################

"""Required versions for PyTables dependencies."""

#***************************************************************
#  Keep these in sync with setup.py and user's guide and README
#***************************************************************

# Minimum recommended versions for mandatory packages
min_numpy_version = '1.4.1'
min_numexpr_version = '2.0.0'
min_cython_version = '0.13'

# The THG team has decided to fix an API inconsistency in the definition
# of the H5Z_class_t structure in version 1.8.3
min_hdf5_version = (1, 8, 4)  # necessary for allowing 1.8.10 > 1.8.5
