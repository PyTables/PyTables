/* 
 * File:   h5f_core_mem.h
 * Author: af1n
 *
 * Created on July 19, 2012, 11:41 AM
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

