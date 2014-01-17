# sources
set(SOURCES blosc_filter.c)

include_directories("${PROJECT_SOURCE_DIR}/blosc")

# dependencies
find_package(HDF5 REQUIRED)
include_directories(HDF5_INCLIDE_DIRS)


# targets
add_library(blosc_filter_shared SHARED ${SOURCES})
set_target_properties(blosc_filter_shared PROPERTIES OUTPUT_NAME blosc_filter)
target_link_libraries(blosc_filter_shared blosc_shared ${HDF5_LIBRARIES})

if(BUILD_STATIC)
    add_library(blosc_filter_static ${SOURCES})
    set_target_properties(
        blosc_filter_static PROPERTIES OUTPUT_NAME blosc_filter)
    target_link_libraries(blosc_filter_static blosc_static)
endif(BUILD_STATIC)


# install
install(FILES blosc_filter.h DESTINATION include COMPONENT HDF5_FILTER_DEV)
install(TARGETS blosc_filter_static DESTINATION lib COMPONENT HDF5_FILTER)
if(BUILD_STATIC)
    install(
        TARGETS blosc_filter_shared DESTINATION lib COMPONENT HDF5_FILTER_DEV)
endif(BUILD_STATIC)


# test
if(BUILD_TESTS)
    add_executable(example example.c)
    target_link_libraries(example blosc_filter_static ${HDF5_LIBRARIES})
    add_test(test_hdf5_filter example)
endif(BUILD_TESTS)
