#include "H5LT.h"
#include "tables.h"
#include "H5Zlzo.h"
#include "H5Zucl.h"

#include <string.h>
#include <stdlib.h>

/*-------------------------------------------------------------------------
 * 
 * Private functions
 * 
 *-------------------------------------------------------------------------
 */

void *test_vltypes_alloc_custom(size_t size, void *info);
void test_vltypes_free_custom(void *mem, void *info);

/****************************************************************
**
**  test_vltypes_alloc_custom(): Test VL datatype custom memory
**      allocation routines.  This routine just uses malloc to
**      allocate the memory and increments the amount of memory
**      allocated.
** 
****************************************************************/
void *test_vltypes_alloc_custom(size_t size, void *info)
{
    void *ret_value=NULL;       /* Pointer to return */
    size_t *mem_used=(size_t *)info;  /* Get the pointer to the memory used */

    if((ret_value=malloc(size))!=NULL) {
        *(size_t *)ret_value=size;
        *mem_used+=size;
    }
    ret_value=((unsigned char *)ret_value);
    return(ret_value);
}

/* This version does not return aligned memory for Float64 variables */
void *test_vltypes_alloc_custom_orig(size_t size, void *info)
{
    void *ret_value=NULL;       /* Pointer to return */
    size_t *mem_used=(size_t *)info;  /* Get the pointer to the memory used */
    size_t extra;               /* Extra space needed */

    /*
     *  This weird contortion is required on the DEC Alpha to keep the
     *  alignment correct - QAK
     */
/*     extra=MAX(sizeof(void *),sizeof(size_t)); */
    /* I've replaced the above line by the next code */
    extra=sizeof(void *);
    if (extra < sizeof(size_t)) 
      extra = sizeof(size_t);

    if((ret_value=malloc(extra+size))!=NULL) {
        *(size_t *)ret_value=size;
        *mem_used+=size;
    }
    ret_value=((unsigned char *)ret_value)+extra;
    return(ret_value);
}

/****************************************************************
**
**  test_vltypes_free_custom(): Test VL datatype custom memory
**      allocation routines.  This routine just uses free to
**      release the memory and decrements the amount of memory
**      allocated.
** 
****************************************************************/
void test_vltypes_free_custom(void *_mem, void *info)
{
    unsigned char *mem;
    size_t *mem_used=(size_t *)info;  /* Get the pointer to the memory used */

    if(_mem!=NULL) {
        mem=((unsigned char *)_mem);
        *mem_used-=*(size_t *)mem;
        free(mem);
    } /* end if */
}

/* This version does not return aligned memory for Float64 variables */
void test_vltypes_free_custom_orig(void *_mem, void *info)
{
    unsigned char *mem;
    size_t *mem_used=(size_t *)info;  /* Get the pointer to the memory used */
    size_t extra;               /* Extra space needed */

    /*
     *  This weird contortion is required on the DEC Alpha to keep the
     *  alignment correct - QAK
     */
/*     extra=MAX(sizeof(void *),sizeof(size_t)); */
    extra=sizeof(void *);
    if (extra < sizeof(size_t)) 
      extra = sizeof(size_t);

    if(_mem!=NULL) {
        mem=((unsigned char *)_mem)-extra;
        *mem_used-=*(size_t *)mem;
        free(mem);
    } /* end if */
}

/*-------------------------------------------------------------------------
 * 
 * Public functions
 * 
 *-------------------------------------------------------------------------
 */


/*-------------------------------------------------------------------------
 * Function: H5VLARRAYmake
 *
 * Purpose: Creates and writes a dataset of a variable length type type_id
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: F. Alted
 *
 * Date: November 08, 2003
 *-------------------------------------------------------------------------
 */

herr_t H5VLARRAYmake( hid_t loc_id, 
		      const char *dset_name,
		      const char *title,
		      const char *flavor,
		      const char *obversion,    /* The Array VERSION number */
		      const int rank, 
		      const int scalar, 
		      const hsize_t *dims,
		      hid_t type_id,
		      hsize_t chunk_size,
		      void  *fill_data,
		      int   compress,
		      char  *complib,
		      int   shuffle,
		      int   fletcher32,
		      const void *data)
{

 hvl_t   vldata;
 hid_t   dataset_id, space_id, datatype, tid1;
 hsize_t dataset_dims[1];
 hsize_t maxdims[1] = { H5S_UNLIMITED };
 hsize_t dims_chunk[1];
 hid_t   plist_id;
 unsigned int cd_values[2];
 int     i;

 if (data)
   /* if data, one row will be filled initially */
   dataset_dims[0] = 1;
 else
   /* no data, so no rows on dataset initally */
   dataset_dims[0] = 0;

 dims_chunk[0] = chunk_size;

 /* Fill the vldata estructure with the data to write */
 /* This is currectly not used */
 vldata.p = (void *)data;
 vldata.len = 1;		/* Only one array type to save */

 /* Create a VL datatype */
 if (scalar == 1) {
   datatype = H5Tvlen_create(type_id);
 }
 else {
   tid1 = H5Tarray_create(type_id, rank, dims, NULL);
   datatype = H5Tvlen_create(tid1);
   H5Tclose( tid1 );   /* Release resources */
 }

 /* The dataspace */
 space_id = H5Screate_simple( 1, dataset_dims, maxdims );

 /* Modify dataset creation properties, i.e. enable chunking  */
 plist_id = H5Pcreate (H5P_DATASET_CREATE);
 if ( H5Pset_chunk ( plist_id, 1, dims_chunk ) < 0 )
   return -1;

 /* 
    Dataset creation property list is modified to use 
 */

 /* Fletcher must be first */
 if (fletcher32) {
   if ( H5Pset_fletcher32( plist_id) < 0 )
     return -1;
 }
 /* Then shuffle */
 if (shuffle) {
   if ( H5Pset_shuffle( plist_id) < 0 )
     return -1;
 }
 /* Finally compression */
 if (compress) {
   cd_values[0] = compress;
   cd_values[1] = (int)(atof(obversion) * 10);
   cd_values[2] = VLArray;
   /* The default compressor in HDF5 (zlib) */
   if (strcmp(complib, "zlib") == 0) {
     if ( H5Pset_deflate( plist_id, compress) < 0 )
       return -1;
   }
   /* The LZO compressor does accept parameters */
   else if (strcmp(complib, "lzo") == 0) {
     if ( H5Pset_filter( plist_id, FILTER_LZO, H5Z_FLAG_OPTIONAL, 3, cd_values) < 0 )
       return -1;
   }
   /* The UCL compress does accept parameters */
   else if (strcmp(complib, "ucl") == 0) {
     if ( H5Pset_filter( plist_id, FILTER_UCL, H5Z_FLAG_OPTIONAL, 3, cd_values) < 0 )
       return -1;
   }
   else {
     /* Compression library not supported */
     fprintf(stderr, "Compression library not supported\n");
     return -1;
   }
 }

 /* Create the dataset. */
 if ((dataset_id = H5Dcreate(loc_id, dset_name, datatype, space_id, plist_id )) < 0 )
   goto out;

 /* Write the dataset only if there is data to write */
 if (data) 
   if ( H5Dwrite( dataset_id, datatype, H5S_ALL, H5S_ALL, H5P_DEFAULT, &vldata ) < 0 )
     goto out;

 /* End access to the datasets and release resources used by them. */
 if ( H5Dclose( dataset_id ) < 0 )
  return -1;

 /* Terminate access to the data space. */ 
 if ( H5Sclose( space_id ) < 0 )
  return -1;

 /* Release the datatype in the case that it is not an atomic type */
 if ( H5Tclose( datatype ) < 0 )
   return -1;

 /* End access to the property list */
 if ( H5Pclose( plist_id ) < 0 )
   goto out;

/*-------------------------------------------------------------------------
 * Set the conforming array attributes
 *-------------------------------------------------------------------------
 */
    
 /* Attach the CLASS attribute */
 if ( H5LTset_attribute_string( loc_id, dset_name, "CLASS", "VLARRAY" ) < 0 )
  goto out;
   
 /* Attach the CLASS attribute */
 if ( H5LTset_attribute_string( loc_id, dset_name, "FLAVOR", flavor ) < 0 )
  goto out;
   
 /* Attach the VERSION attribute */
 if ( H5LTset_attribute_string( loc_id, dset_name, "VERSION", obversion ) < 0 )
  goto out;
     
 /* Attach the TITLE attribute */
 if ( H5LTset_attribute_string( loc_id, dset_name, "TITLE", title ) < 0 )
  goto out;

 return 0;

out:
 H5Dclose( dataset_id );
 H5Sclose( space_id ); 

 return -1;

}

/*-------------------------------------------------------------------------
 * Function: H5ARRAYappend_records
 *
 * Purpose: Appends records to an array
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmers: 
 *  Francesc Alted
 *
 * Date: October 30, 2003
 *
 * Comments: Uses memory offsets
 *
 * Modifications: 
 *
 *
 *-------------------------------------------------------------------------
 */


herr_t H5VLARRAYappend_records( hid_t loc_id, 
				const char *dset_name,
				int nobjects,
				hsize_t nrecords,
				const void *data )  
{

 hid_t    dataset_id;
 hid_t    type_id;
 hid_t    space_id;
 hid_t    mem_space_id;
 hsize_t  start[1];
 hsize_t  dataset_dims[1];
 hsize_t  dims_new[1] = {1};	/* Only a record on each append */
 hvl_t    wdata;   /* Information to write */
 int      i;

 /* Initialize VL data to write */
 wdata.p=(void *)data;
 wdata.len=nobjects;

 /* Open the dataset. */
 if ( (dataset_id = H5Dopen( loc_id, dset_name )) < 0 )
  goto out;

 /* Get the datatype */
 if ( (type_id = H5Dget_type( dataset_id )) < 0 )
  goto out;

 /* Dimension for the new dataset */
 dataset_dims[0] = nrecords + 1;

 /* Extend the dataset */
 if ( H5Dextend ( dataset_id, dataset_dims ) < 0 )
  goto out;

 /* Create a simple memory data space */
 if ( (mem_space_id = H5Screate_simple( 1, dims_new, NULL )) < 0 )
  return -1;

 /* Get the file data space */
 if ( (space_id = H5Dget_space( dataset_id )) < 0 )
  return -1;

 /* Define a hyperslab in the dataset */
 start[0] = nrecords;
 if ( H5Sselect_hyperslab( space_id, H5S_SELECT_SET, start, NULL, dims_new, NULL) < 0 )
   goto out;

 if ( H5Dwrite( dataset_id, type_id, mem_space_id, space_id, H5P_DEFAULT, &wdata ) < 0 )
     goto out;

 /* Terminate access to the dataspace */
 if ( H5Sclose( space_id ) < 0 )
  goto out;
 
 if ( H5Sclose( mem_space_id ) < 0 )
  goto out;
 
 /* Release the datatype. */
 if ( H5Tclose( type_id ) < 0 )
  return -1;

 /* End access to the dataset */
 if ( H5Dclose( dataset_id ) < 0 )
  goto out;

return 1;

out:
 H5Dclose( dataset_id );
 return -1;

}


/*-------------------------------------------------------------------------
 * Function: H5VLARRAYread
 *
 * Purpose: Reads an array from disk.
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Alted, falted@openlc.org
 *
 * Date: November 19, 2003
 *
 *-------------------------------------------------------------------------
 */

herr_t H5VLARRAYread( hid_t loc_id, 
		      const char *dset_name,
		      hsize_t start,
		      hsize_t nrecords,
		      hsize_t step,
		      hvl_t *data,
		      hsize_t *datalen)
{
 hid_t    dataset_id;
 hid_t    space_id;
 hid_t    mem_space_id;
 hid_t    type_id;
 hsize_t  dims[1];
 hsize_t  count[1];    
 hsize_t  stride[1];    
 hssize_t offset[1];
 hid_t    xfer_pid;   /* Dataset transfer property list ID */
 hsize_t  size;       /* Number of bytes which will be used */
 size_t   mem_used=0; /* Memory used during allocation */

 /* Open the dataset. */
 if ( (dataset_id = H5Dopen( loc_id, dset_name )) < 0 )
  return -1;
 
 /* Get the datatype */
 if ( (type_id = H5Dget_type(dataset_id)) < 0 )
     return -1;

  /* Get the dataspace handle */
 if ( (space_id = H5Dget_space( dataset_id )) < 0 )
  goto out;

 /* Get numbers of rows */
 if ( H5Sget_simple_extent_dims( space_id, dims, NULL) < 0 )
  goto out;

 if ( start + nrecords > dims[0] ) {
   printf("Asking for a range of rows exceeding the available ones!.\n");
   goto out;
 }

 /* Define a hyperslab in the dataset of the size of the records */
 offset[0] = start;
 count[0]  = nrecords;
 stride[0] = step;
 if ( H5Sselect_hyperslab( space_id, H5S_SELECT_SET, offset, stride, count, NULL) < 0 )
  goto out;

 /* Change to the custom memory allocation routines for reading VL data */
 if ((xfer_pid=H5Pcreate(H5P_DATASET_XFER)) < 0 )
   goto out;

 if (H5Pset_vlen_mem_manager(xfer_pid, test_vltypes_alloc_custom, 
			     &mem_used, test_vltypes_free_custom, &mem_used))
   goto out;

 /* Make certain the correct amount of memory will be used */
/*  H5Dvlen_get_buf_size(dataset_id, type_id, space_id, &size); */
/*  printf("Memory size to book: %d\n", size); */

 /* Create a memory dataspace handle */
 if ( (mem_space_id = H5Screate_simple( 1, count, NULL )) < 0 )
  goto out;

 /* These two possibilities do work: */
 if ( H5Dread( dataset_id, type_id, mem_space_id, space_id, xfer_pid, data ) < 0 )
/*  if ( H5Dread( dataset_id, type_id, mem_space_id, space_id, H5P_DEFAULT, data ) < 0 ) */
  goto out;

/*  printf("Memory size allocated after reading: %d\n", (int)mem_used); */
 *datalen = (hsize_t) mem_used;
 
 /* Terminate access to the memory dataspace */
 if ( H5Sclose( mem_space_id ) < 0 )
  goto out;

 /* Terminate access to the dataspace */
 if ( H5Sclose( space_id ) < 0 )
  goto out;

 /* End access to the dataset and release resources used by it. */
 if ( H5Dclose( dataset_id ) )
  return -1;

 /* Close the vlen type */
 if ( H5Tclose(type_id))
   return -1;

 /* Close dataset transfer property list */
 if (H5Pclose(xfer_pid))
   return -1;

 return 0;

out:
 H5Dclose( dataset_id );
 return -1;

}
  

/*-------------------------------------------------------------------------
 * Function: H5VLARRAYget_ndims
 *
 * Purpose: Gets the dimensionality of an array.
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Alted
 *
 * Date: November 19, 2003
 *
 *-------------------------------------------------------------------------
 */

herr_t H5VLARRAYget_ndims( hid_t loc_id, 
			   const char *dset_name,
			   int *rank )
{
  hid_t       dataset_id;  
  hid_t       space_id; 
  hid_t       type_id; 
  hid_t       atom_type_id; 
  H5T_class_t atom_class_id;

  /* Open the dataset. */
  if ( (dataset_id = H5Dopen( loc_id, dset_name )) < 0 )
    return -1;

  /* Get the datatype handle */
  if ( (type_id = H5Dget_type( dataset_id )) < 0 )
    goto out;

  /* Get an identifier for the datatype. */
  type_id = H5Dget_type( dataset_id );

  /* Get the type of the atomic component */
  atom_type_id = H5Tget_super( type_id );

  /* Get the class of the atomic component. */
  atom_class_id = H5Tget_class( atom_type_id );

  /* Check whether the atom is an array class object or not */ 
  if ( atom_class_id == H5T_ARRAY) {
    /* Get rank */
    if ( (*rank = H5Tget_array_ndims( atom_type_id )) < 0 )
      goto out;
  }
  else {
    *rank = 0;		/* Means scalar values */
  }

 /* Terminate access to the datatypes */
 if ( H5Tclose( atom_type_id ) < 0 )
  goto out;

 if ( H5Tclose( type_id ) < 0 )
  goto out;

 /* End access to the dataset */
 if ( H5Dclose( dataset_id ) )
  return -1;

 return 0;

out:
 H5Dclose( dataset_id );
 return -1;

}

/*-------------------------------------------------------------------------
 * Function: H5VLARRAYget_info
 *
 * Purpose: Gathers info about the VLEN type and other.
 *
 * Return: Success: 0, Failure: -1
 *
 * Programmer: Francesc Alted
 *
 * Date: November 19, 2003
 *
 *-------------------------------------------------------------------------
 */

herr_t H5VLARRAYget_info( hid_t   loc_id, 
			  char    *dset_name,
			  hsize_t *nrecords,
			  hsize_t *base_dims,
			  hid_t   *base_type_id,
			  char    *base_byteorder )
{
  hid_t       dataset_id;  
  hid_t       type_id;
  hid_t       space_id; 
  H5T_class_t base_class_id;
  H5T_class_t atom_class_id;
  hid_t       atom_type_id; 
  H5T_order_t order;
  int i;

  /* Open the dataset. */
  if ( (dataset_id = H5Dopen( loc_id, dset_name )) < 0 )
    return -1;

  /* Get the dataspace handle */
  if ( (space_id = H5Dget_space( dataset_id )) < 0 )
    goto out;

  /* Get number of records (it should be rank-1) */
  if ( H5Sget_simple_extent_dims( space_id, nrecords, NULL) < 0 )
    goto out;

/*   printf("nrecords --> %d\n", *nrecords); */

  /* Terminate access to the dataspace */
  if ( H5Sclose( space_id ) < 0 )
    goto out;
 
  /* Get an identifier for the datatype. */
  type_id = H5Dget_type( dataset_id );

  /* Get the type of the atomic component */
  atom_type_id = H5Tget_super( type_id );

  /* Get the class of the atomic component. */
  atom_class_id = H5Tget_class( atom_type_id );

  /* Check whether the atom is an array class object or not */ 
  if ( atom_class_id == H5T_ARRAY) {
    /* Get the array base component */
    *base_type_id = H5Tget_super( atom_type_id );
    /* Get the class of base component */
    base_class_id = H5Tget_class( *base_type_id );
    /* Get dimensions */
    if ( H5Tget_array_dims(atom_type_id, base_dims, NULL) < 0 )
      goto out;
    /* Release the datatypes */
    if ( H5Tclose(atom_type_id ) )
      return -1;
  }
  else {
    base_class_id = atom_class_id;
    *base_type_id = atom_type_id;
    base_dims = NULL; 		/* Is a scalar */
  }

  /* Get the byteorder */
  /* Only class integer and float can be byteordered */
  if ( (base_class_id == H5T_INTEGER) || (base_class_id == H5T_FLOAT)
       || (base_class_id == H5T_BITFIELD) ) {
    order = H5Tget_order( *base_type_id );
    if (order == H5T_ORDER_LE) 
      strcpy(base_byteorder, "little");
    else if (order == H5T_ORDER_BE)
      strcpy(base_byteorder, "big");
    else {
      fprintf(stderr, "Error: unsupported byteorder: %d\n", order);
      goto out;
    }
  }
  else {
    strcpy(base_byteorder, "non-relevant");
  }

  /* Close the VLEN datatype */
  if ( H5Tclose(type_id ) )
    return -1;

  /* End access to the dataset */
  if ( H5Dclose( dataset_id ) )
    return -1;

  return 0;

out:
 H5Tclose( type_id );
 H5Dclose( dataset_id );
 return -1;

}
