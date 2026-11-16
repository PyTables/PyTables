/* Generate a plain HDF5 numeric-VLEN interoperability fixture.
 * Usage: create-vlen-layouts output.h5
 * The output must not exist. No PyTables metadata or writer is used.
 */
#include <assert.h>
#include <stdio.h>
#include "hdf5.h"

#ifdef NDEBUG
#error "Build this fixture generator with assertions enabled"
#endif

int main(int argc, char **argv) {
    const int a[] = {3, 2, 1};
    const int b[] = {1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144};
    const int c[] = {-257, 513};
    const hvl_t rows[] = {{3, (void *)a}, {12, (void *)b},
                          {0, NULL}, {2, (void *)c}};
    const char *layouts[] = {"contiguous", "compact", "fixed_chunked", "extendable"};
    hid_t file, memtype;
    int endian, layout, empty;
    assert(argc == 2);
    file = H5Fcreate(argv[1], H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
    assert(file >= 0);
    memtype = H5Tvlen_create(H5T_NATIVE_INT);
    assert(memtype >= 0);
    for (endian = 0; endian < 2; ++endian) {
        hid_t type = H5Tvlen_create(endian ? H5T_STD_I32BE : H5T_STD_I32LE);
        assert(type >= 0);
        for (layout = 0; layout < 4; ++layout) {
            for (empty = 0; empty < 2; ++empty) {
                hsize_t dims[] = {empty ? 0 : 4};
                hsize_t maxdims[] = {layout == 3 ? H5S_UNLIMITED : dims[0]};
                hsize_t chunks[] = {2};
                hvl_t readback[4] = {{0, NULL}, {0, NULL}, {0, NULL}, {0, NULL}};
                char name[80];
                hid_t space, dcpl, dataset;
                unsigned int i, j;
                if (empty && layout == 2) continue;
                sprintf(name, "%s_%s%s", layouts[layout], endian ? "be" : "le",
                        empty ? "_empty" : "");
                space = H5Screate_simple(1, dims, maxdims);
                dcpl = H5Pcreate(H5P_DATASET_CREATE);
                assert(space >= 0 && dcpl >= 0);
                if (layout == 1) assert(H5Pset_layout(dcpl, H5D_COMPACT) >= 0);
                if (layout >= 2) assert(H5Pset_chunk(dcpl, 1, chunks) >= 0);
                dataset = H5Dcreate2(file, name, type, space, H5P_DEFAULT, dcpl, H5P_DEFAULT);
                assert(dataset >= 0);
                if (!empty) {
                    assert(H5Dwrite(dataset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, rows) >= 0);
                    assert(H5Dread(dataset, memtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, readback) >= 0);
                    for (i = 0; i < 4; ++i) {
                        assert(readback[i].len == rows[i].len);
                        for (j = 0; j < rows[i].len; ++j)
                            assert(((int *)readback[i].p)[j] == ((int *)rows[i].p)[j]);
                    }
                    assert(H5Treclaim(memtype, space, H5P_DEFAULT, readback) >= 0);
                }
                assert(H5Dclose(dataset) >= 0);
                assert(H5Pclose(dcpl) >= 0);
                assert(H5Sclose(space) >= 0);
                printf("verified %s\n", name);
            }
        }
        assert(H5Tclose(type) >= 0);
    }
    assert(H5Tclose(memtype) >= 0);
    assert(H5Fclose(file) >= 0);
    return 0;
}
