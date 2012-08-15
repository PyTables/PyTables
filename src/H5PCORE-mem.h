/* 
 * Author: Michal Slonina <michal.slonina@gmail.com>
 * Created on July 19, 2012, 11:41 AM
 * Work on PyTables in-memory hdf5 file images file was kindly sponsored by DeltaMethod.
 */

#ifndef H5F_CORE_MEM_H
#define	H5F_CORE_MEM_H

#include <H5Tpublic.h>

#ifdef	__cplusplus
extern "C" {
#endif

hid_t H5Pset_file_inmemory_callbacks(hid_t fapl, hvl_t *udata);
int H5PCOREhasHDF5HL();
hid_t H5LTopen_file_image_proxy(void *buf_ptr, size_t buf_size, unsigned flags);

#ifdef	__cplusplus
}
#endif

#endif	/* H5F_CORE_MEM_H */

