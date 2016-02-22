import os
import sys
import warnings

from h5py import h5f, h5p

from .definitions cimport hid_t, herr_t, hsize_t, H5_HAVE_IMAGE_FILE

from .definitions cimport H5P_DEFAULT, H5Pclose, H5Pset_fapl_log, H5Pset_fapl_family, H5Pset_fapl_multi, H5Pset_fapl_split, H5Pset_sieve_buf_size

from .definitions cimport H5Fget_filesize, H5Fget_create_plist, H5Fget_vfd_handle, H5Fopen, H5Fcreate, H5Fflush, H5Fclose, H5F_ACC_RDONLY, H5F_ACC_RDWR, H5F_ACC_TRUNC

from .definitions cimport uintptr_t
from .definitions cimport set_cache_size

from .exceptions import HDF5ExtError

from .utils import check_file_access

from .utilsextension import set_blosc_max_threads, encode_filename

from .definitions cimport pt_H5Pset_fapl_direct, pt_H5Pset_fapl_windows, pt_H5Pset_file_image, pt_H5Fget_file_image

# TODO check this
from cpython.bytes cimport PyBytes_AsString, PyBytes_FromStringAndSize

cimport numpy as np

_supported_drivers = (
    "H5FD_SEC2",
    "H5FD_DIRECT",
    #"H5FD_LOG",
    "H5FD_WINDOWS",
    "H5FD_STDIO",
    "H5FD_CORE",
    #"H5FD_FAMILY",
    #"H5FD_MULTI",
    "H5FD_SPLIT",
    #"H5FD_MPIO",
    #"H5FD_MPIPOSIX",
    #"H5FD_STREAM",
)

from .definitions cimport H5_HAVE_DIRECT_DRIVER, H5_HAVE_WINDOWS_DRIVER

HAVE_DIRECT_DRIVER = bool(H5_HAVE_DIRECT_DRIVER)
HAVE_WINDOWS_DRIVER = bool(H5_HAVE_WINDOWS_DRIVER)


cdef class File:
  def _g_new(self, name, pymode, **params):
    cdef herr_t err = 0
    cdef hid_t meta_plist_id = H5P_DEFAULT, raw_plist_id = H5P_DEFAULT
    cdef size_t img_buf_len = 0, user_block_size = 0
    cdef void *img_buf_p = NULL
    cdef bytes encname
    #cdef bytes logfile_name

    # Check if we can handle the driver
    driver = params["DRIVER"]
    if driver is not None and driver not in _supported_drivers:
      raise ValueError("Invalid or not supported driver: '%s'" % driver)
    if driver == "H5FD_SPLIT":
      meta_ext = params.get("DRIVER_SPLIT_META_EXT", "-m.h5")
      raw_ext = params.get("DRIVER_SPLIT_RAW_EXT", "-r.h5")
      meta_name = meta_ext % name if "%s" in meta_ext else name + meta_ext
      raw_name = raw_ext % name if "%s" in raw_ext else name + raw_ext
      enc_meta_ext = encode_filename(meta_ext)
      enc_raw_ext = encode_filename(raw_ext)

    # Create a new file using default properties
    self.name = name

    # Encode the filename in case it is unicode
    encname = encode_filename(name)

    # These fields can be seen from Python.
    self._v_new = None  # this will be computed later
    # """Is this file going to be created from scratch?"""

    self._isPTFile = True  # assume a PyTables file by default
    # """Does this HDF5 file have a PyTables format?"""

    # defaults access and creation property lists
    access_plist = h5p.DEFAULT
    create_plist = h5p.DEFAULT

    assert pymode in ('r', 'r+', 'a', 'w'), ("an invalid mode string ``%s`` "
           "passed the ``check_file_access()`` test; "
           "please report this to the authors" % pymode)

    image = params.get('DRIVER_CORE_IMAGE')
    if image:
      if driver != "H5FD_CORE":
        warnings.warn("The DRIVER_CORE_IMAGE parameter will be ignored by "
                      "the '%s' driver" % driver)
      elif not isinstance(image, bytes):
        raise TypeError("The DRIVER_CORE_IMAGE must be a string of bytes")
      elif not H5_HAVE_IMAGE_FILE:
        raise RuntimeError("Support for image files is only availabe in "
                           "HDF5 >= 1.8.9")

    # After the following check we can be quite sure
    # that the file or directory exists and permissions are right.
    if driver == "H5FD_SPLIT":
      for n in meta_name, raw_name:
        check_file_access(n, pymode)
    else:
      backing_store = params.get("DRIVER_CORE_BACKING_STORE", 1)
      if driver != "H5FD_CORE" or backing_store:
        check_file_access(name, pymode)

    # Should a new file be created?
    if image:
      exists = True
    elif driver == "H5FD_SPLIT":
      exists = os.path.exists(meta_name) and os.path.exists(raw_name)
    else:
      exists = os.path.exists(name)
    self._v_new = not (pymode in ('r', 'r+') or (pymode == 'a' and exists))

    user_block_size = params.get("USER_BLOCK_SIZE", 0)
    if user_block_size and not self._v_new:
        warnings.warn("The HDF5 file already esists: the USER_BLOCK_SIZE "
                      "will be ignored")
    elif user_block_size:
      is_pow_of_2 = ((user_block_size & (user_block_size - 1)) == 0)
      if user_block_size < 512 or not is_pow_of_2:
        raise ValueError("The USER_BLOCK_SIZE must be zero or a power of 2"
            "greater than 512")

      # File creation property list
      create_plist = h5p.create(h5p.FILE_CREATE)
      create_plist.set_userblock(user_block_size)

    # File access property list
    access_plist = h5p.create(h5p.FILE_ACCESS)

    # Set parameters for chunk cache
    access_plist.set_cache(0, params["CHUNK_CACHE_NELMTS"],
                           params["CHUNK_CACHE_SIZE"],
                           params["CHUNK_CACHE_PREEMPT"])

    # Set the I/O driver
    if driver == "H5FD_SEC2":
      access_plist.set_fapl_sec2()
    elif driver == "H5FD_DIRECT":
      # FIXME not sure what was going on here
      raise RuntimeError("not implemented")

      # if not H5_HAVE_DIRECT_DRIVER:
      #   raise RuntimeError("The H5FD_DIRECT driver is not available")
      # err = pt_H5Pset_fapl_direct(access_plist,
      #                             params["DRIVER_DIRECT_ALIGNMENT"],
      #                             params["DRIVER_DIRECT_BLOCK_SIZE"],
      #                             params["DRIVER_DIRECT_CBUF_SIZE"])

    #elif driver == "H5FD_LOG":
    #  if "DRIVER_LOG_FILE" not in params:
    #    H5Pclose(access_plist)
    #    raise ValueError("The DRIVER_LOG_FILE parameter is required for "
    #                     "the H5FD_LOG driver")
    #  logfile_name = encode_filename(params["DRIVER_LOG_FILE"])
    #  err = H5Pset_fapl_log(access_plist,
    #                        <char*>logfile_name,
    #                        params["DRIVER_LOG_FLAGS"],
    #                        params["DRIVER_LOG_BUF_SIZE"])

    elif driver == "H5FD_WINDOWS":
      # FIXME not sure what was going on here
      raise RuntimeError("not implemented")
      # if not H5_HAVE_WINDOWS_DRIVER:
      #   raise RuntimeError("The H5FD_WINDOWS driver is not available")
      # err = pt_H5Pset_fapl_windows(access_plist)
    elif driver == "H5FD_STDIO":
      access_plist.set_fapl_stdio()
    elif driver == "H5FD_CORE":
      access_plist.set_fapl_core(params["DRIVER_CORE_INCREMENT"], backing_store)
      if image:
        # FIXME h5py doesn't support this yet
        raise RuntimeError("not implemented")
        img_buf_len = len(image)
        img_buf_p = <void *>PyBytes_AsString(image)
        access_plist.set
        err = pt_H5Pset_file_image(access_plist, img_buf_p, img_buf_len)
        if err < 0:
          H5Pclose(access_plist)
          raise HDF5ExtError("Unable to set the file image")

    #elif driver == "H5FD_FAMILY":
    #  H5Pset_fapl_family(access_plist,
    #                     params["DRIVER_FAMILY_MEMB_SIZE"],
    #                     fapl_id)
    #elif driver == "H5FD_MULTI":
    #  err = H5Pset_fapl_multi(access_plist, memb_map, memb_fapl, memb_name,
    #                          memb_addr, relax)

    elif driver == "H5FD_SPLIT":
      raise RuntimeError("not implemented")
      err = H5Pset_fapl_split(access_plist, enc_meta_ext, meta_plist_id,
                              enc_raw_ext, raw_plist_id)

    if pymode == 'r':
      self._file = h5f.open(encname, h5f.ACC_RDONLY, access_plist)
    elif pymode == 'r+':
      self._file = h5f.open(encname, h5f.ACC_RDWR, access_plist)
    elif pymode == 'a':
      if exists:
        # A test for logging.
        ## H5Pset_sieve_buf_size(access_plist, 0)
        ## H5Pset_fapl_log (access_plist, "test.log", H5FD_LOG_LOC_WRITE, 0)
        self._file = h5f.open(encname, h5f.ACC_RDWR, access_plist)
      else:
        self._file = h5f.create(encname, h5f.ACC_TRUNC, create_plist,
                                access_plist)
    elif pymode == 'w':
      self._file = h5f.create(encname, h5f.ACC_TRUNC, create_plist,
                              access_plist)

    # Set the cache size
    cache_config = self._file.get_mdc_config()
    cache_config.set_initial_size = True
    cache_config.initial_size = params["METADATA_CACHE_SIZE"]
    self._file.set_mdc_config(cache_config)

    # Save the id for PyTables
    self.file_id = self._file.id

    # Set the maximum number of threads for Blosc
    set_blosc_max_threads(params["MAX_BLOSC_THREADS"])

  # XXX: add the possibility to pass a pre-allocated buffer
  def get_file_image(self):
    """Retrieves an in-memory image of an existing, open HDF5 file.

    .. note:: this method requires HDF5 >= 1.8.9.

    .. versionadded:: 3.0

    """

    cdef ssize_t size = 0
    cdef size_t buf_len = 0
    cdef bytes image
    cdef char* cimage

    self.flush()

    # retrieve the size of the buffer for the file image
    size = pt_H5Fget_file_image(self.file_id, NULL, buf_len)
    if size < 0:
      raise HDF5ExtError("Unable to retrieve the size of the buffer for the "
                         "file image.  Plese note that not all drivers "
                         "provide support for image files.")

    # allocate the memory buffer
    image = PyBytes_FromStringAndSize(NULL, size)
    if not image:
      raise RuntimeError("Unable to allecote meomory fir the file image")

    cimage = image
    buf_len = size
    size = pt_H5Fget_file_image(self.file_id, <void*>cimage, buf_len)
    if size < 0:
      raise HDF5ExtError("Unable to retrieve the file image. "
                         "Plese note that not all drivers provide support "
                         "for image files.")

    return image

  def get_filesize(self):
    """Returns the size of an HDF5 file.

    The returned size is that of the entire file, as opposed to only
    the HDF5 portion of the file. I.e., size includes the user block,
    if any, the HDF5 portion of the file, and any data that may have
    been appended beyond the data written through the HDF5 Library.

    .. versionadded:: 3.0

    """

    cdef herr_t err = 0
    cdef hsize_t size = 0

    err = H5Fget_filesize(self.file_id, &size)
    if err < 0:
      raise HDF5ExtError("Unable to retrieve the HDF5 file size")

    return size

  def get_userblock_size(self):
    """Retrieves the size of a user block.

    .. versionadded:: 3.0

    """

    create_plist = self._file.get_create_plist()
    return create_plist.get_userblock()


  # Accessor definitions
  def _get_file_id(self):
    return self.file_id

  def fileno(self):
    """Return the underlying OS integer file descriptor.

    This is needed for lower-level file interfaces, such as the ``fcntl``
    module.

    """

    cdef void *file_handle
    cdef uintptr_t *descriptor
    cdef herr_t err
    err = H5Fget_vfd_handle(self.file_id, H5P_DEFAULT, &file_handle)
    if err < 0:
      raise HDF5ExtError(
        "Problems getting file descriptor for file ``%s``" % self.name)
    # Convert the 'void *file_handle' into an 'int *descriptor'
    descriptor = <uintptr_t *>file_handle
    return descriptor[0]


  def _flush_file(self, scope):
    # Close the file
    H5Fflush(self.file_id, scope)


  def _close_file(self):
    # Close the file
    H5Fclose( self.file_id )
    self.file_id = 0    # Means file closed


  # This method is moved out of scope, until we provide code to delete
  # the memory booked by this extension types
  def __dealloc__(self):
    cdef int ret
    if self.file_id > 0:
      # Close the HDF5 file because user didn't do that!
      ret = H5Fclose(self.file_id)
      if ret < 0:
        raise HDF5ExtError("Problems closing the file '%s'" % self.name)

