#include "H5LT.h"

#define FILTER_LZO 305

int register_lzo(void);

size_t lzo_deflate (unsigned flags, size_t cd_nelmts,
		    const unsigned cd_values[], size_t nbytes,
		    size_t *buf_size, void **buf);
