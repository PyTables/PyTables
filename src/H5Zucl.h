#include "H5LT.h"

#define FILTER_UCL 306

int register_ucl(void);

size_t ucl_deflate(unsigned int flags, size_t cd_nelmts,
		   const unsigned int cd_values[], size_t nbytes,
		   size_t *buf_size, void **buf);
