# sources
set(SOURCES blosc_filter.c)

include_directories("${PROJECT_SOURCE_DIR}/blosc")

# dependencies
if(MSVC)
    # FindHDF5.cmake does not find Windows installations. Try to
    # use an environment variable instead until the official "find"
    # file can be updated for Windows.
    #
    # Note that you have to set this environment variable by hand.
    file(TO_CMAKE_PATH "$ENV{HDF5_DIR}" HDF5_HINT)
    set(HDF5_DIR ${HDF5_HINT} CACHE STRING "Path to HDF5 CMake config directory.")
    find_package(HDF5 REQUIRED HINTS ${HDF5_DIR})
else(MSVC)
    find_package(HDF5 REQUIRED)
endif(MSVC)
include_directories(${HDF5_INCLUDE_DIRS})


# targets
add_library(blosc_filter_shared SHARED ${SOURCES})
set_target_properties(blosc_filter_shared PROPERTIES OUTPUT_NAME blosc_filter)
target_link_libraries(blosc_filter_shared blosc_shared ${HDF5_LIBRARIES})

if(BUILD_STATIC)
    add_library(blosc_filter_static ${SOURCES})
    set_target_properties(
        blosc_filter_static PROPERTIES OUTPUT_NAME blosc_filter)
    target_link_libraries(blosc_filter_static blosc_static ${HDF5_LIBRARIES})
endif(BUILD_STATIC)

# have to copy blosc dlls for Visual Studio
if(MSVC)
    add_custom_command(
        TARGET      blosc_filter_shared
        POST_BUILD
        COMMAND     ${CMAKE_COMMAND}
        ARGS        -E copy_if_different
                    "${PROJECT_BINARY_DIR}/blosc/\$\(Configuration\)/blosc.dll"
                    "${CMAKE_CURRENT_BINARY_DIR}/\$\(Configuration\)/blosc.dll")
endif(MSVC)

# install
install(FILES blosc_filter.h DESTINATION include COMPONENT HDF5_FILTER_DEV)
install(TARGETS blosc_filter_shared DESTINATION lib COMPONENT HDF5_FILTER)
if(BUILD_STATIC)
    install(
        TARGETS blosc_filter_static DESTINATION lib COMPONENT HDF5_FILTER_DEV)
endif(BUILD_STATIC)


# test
if(BUILD_TESTS)
    add_executable(example example.c)
    target_link_libraries(example blosc_filter_static ${HDF5_LIBRARIES})
    add_test(test_hdf5_filter example)
endif(BUILD_TESTS)
