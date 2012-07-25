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

hid_t H5Fcreate_inmemory(hvl_t *udata);

#ifdef	__cplusplus
}
#endif

#endif	/* H5F_CORE_MEM_H */

