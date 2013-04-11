// This program creates nested types with gaps for testing purposes.
// F. Alted 2008-06-27

#include "hdf5.h"
#include <stdlib.h>


hid_t
create_nested_type(void) {
    hid_t tid, tid2, tid3;
    size_t offset, offset2;

    offset = 1;  offset2 = 2;
    // Create a coumpound type large enough (>= 20)
    tid = H5Tcreate(H5T_COMPOUND, 21);
    // Insert an atomic type
    tid2 = H5Tcopy(H5T_NATIVE_FLOAT);
    H5Tinsert(tid, "float", offset, tid2);
    H5Tclose(tid2);
    offset += 4 + 2;  // add two to the offset so as to create gaps
    // Insert a nested compound
    tid2 = H5Tcreate(H5T_COMPOUND, 12);
    tid3 = H5Tcopy(H5T_NATIVE_CHAR);
    H5Tinsert(tid2, "char", offset2, tid3);
    H5Tclose(tid3);
    offset2 += 2;  // add one space (for introducing gaps)
    tid3 = H5Tcopy(H5T_NATIVE_DOUBLE);
    H5Tinsert(tid2, "double", offset2, tid3);
    H5Tclose(tid3);
    offset2 += 5;  // add one space (for introducing gaps)
    H5Tinsert(tid, "compound", offset, tid2);
    H5Tclose(tid2);
    offset += 12 + 1;
    return(tid);
}

size_t
getNestedSizeType(hid_t type_id) {
    hid_t member_type_id;
    H5T_class_t class_id;
    hsize_t i, nfields;
    size_t itemsize, offset;

    nfields = H5Tget_nmembers(type_id);
    offset = 0;
    // Iterate thru the members
    for (i=0; i < nfields; i++) {
        // Get the member type
        member_type_id = H5Tget_member_type(type_id, i);
        // Get the HDF5 class
        class_id = H5Tget_class(member_type_id);
        if (class_id == H5T_COMPOUND) {
            // Get the member size for compound type
            itemsize = getNestedSizeType(member_type_id);
        }
        else {
            // Get the atomic member size
            itemsize = H5Tget_size(member_type_id);
        }
        // Update the offset
        offset = offset + itemsize;
    }
    return(offset);
}


int
main(int argc, char **argv)
{
    char file_name[256], dset_name[256];
    hid_t file_id, dataset_id, space_id, plist_id, type_id;
    hsize_t dims[1], dims_chunk[1];
    hsize_t maxdims[1] = { H5S_UNLIMITED };
    size_t disk_type_size, computed_type_size, packed_type_size;

    if (argc < 3) {
        printf("Pass the name of the file and dataset to check as arguments\n");
        return(0);
    }

    strcpy(file_name, argv[1]);
    strcpy(dset_name, argv[2]);

    dims[0] = 20;  // Create 20 records
    dims_chunk[0] = 10;

    // Create a new file
    file_id = H5Fcreate(file_name, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    // Create a simple data space with unlimited size
    space_id = H5Screate_simple(1, dims, maxdims);
    // Modify dataset creation properties, i.e. enable chunking
    plist_id = H5Pcreate (H5P_DATASET_CREATE);
    H5Pset_chunk(plist_id, 1, dims_chunk);
    // Get the nested type
    type_id = create_nested_type();
    // Create the dataset
    dataset_id = H5Dcreate(file_id, dset_name, type_id, space_id,
                           H5P_DEFAULT, plist_id, H5P_DEFAULT);
    // Free resources
    H5Sclose(space_id);
    H5Pclose(plist_id);
    H5Dclose(dataset_id);
    H5Fclose(file_id);

    // Compute type sizes for native and packed
    disk_type_size = H5Tget_size(type_id);
    computed_type_size = getNestedSizeType(type_id);
    H5Tpack(type_id);  // pack type
    packed_type_size = H5Tget_size(type_id);
    printf("Disk type size: %d\n", disk_type_size);
    printf("Packed type size: %d (should be %d)\n",
           packed_type_size, computed_type_size);

    H5Tclose(type_id);

    return(1);
}






