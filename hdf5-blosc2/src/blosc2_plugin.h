/*
 * Dynamically loaded filter plugin for HDF5 blosc2 filter.
 *
 * Header file
 * -----------
 *
 * This provides dynamically loaded HDF5 filter functionality (introduced
 * in HDF5-1.8.11, May 2013) to the blosc2 HDF5 filter.
 *
 * Usage: compile as a shared library and install either to the default
 * search location for HDF5 filter plugins (on Linux 
 * /usr/local/hdf5/lib/plugin) or to a location pointed to by the
 * HDF5_PLUGIN_PATH environment variable.
 *
 */


#ifndef PLUGIN_BLOSC2_H
#define PLUGIN_BLOSC2_H

#include "H5PLextern.h"


H5PL_type_t H5PLget_plugin_type(void);


const void* H5PLget_plugin_info(void);


#endif    // PLUGIN_BLOSC2_H


