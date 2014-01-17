# CMake build system for Blosc
# ============================
#
# Available options:
#
#   BUILD_STATIC: default ON
#       build the static version of the Blosc library
#   BUILD_HDF5_FILTER: default OFF
#       build the compression filter for the HDF5 library
#   BUILD_TESTS: default ON
#       build test programs and generates the "test" target
#   BUILD_BENCHMARKS: default ON
#       build the benchmark program
#   DEACTIVATE_LZ4: default OFF
#       do not include support for the LZ4 library
#   DEACTIVATE_SNAPPY: default OFF
#       do not include support for the Snappy library
#   DEACTIVATE_ZLIB: default OFF
#       do not include support for the Zlib library
#   PREFER_EXTERNAL_COMPLIBS: default ON
#       when found, use the installed compression libs instead of included sources
#   TEST_INCLUDE_BENCH_SINGLE_1: default ON
#       add a test that runs the benchmark program passing "single" with 1 thread
#       as first parameter
#   TEST_INCLUDE_BENCH_SINGLE_N: default ON
#       add a test that runs the benchmark program passing "single" with all threads
#       as first parameter
#   TEST_INCLUDE_BENCH_SUITE: default OFF
#       add a test that runs the benchmark program passing "suite"
#       as first parameter
#   TEST_INCLUDE_BENCH_SUITE_PARALLEL: default OFF
#       add a test that runs the benchmark program passing "parallel"
#       as first parameter
#   TEST_INCLUDE_BENCH_HARDSUITE: default OFF
#       add a test that runs the benchmark program passing "hardsuite"
#       as first parameter
#   TEST_INCLUDE_BENCH_EXTREMESUITE: default OFF
#       add a test that runs the benchmark program passing "extremesuite"
#       as first parameter
#   TEST_INCLUDE_BENCH_DEBUGSUITE: default OFF
#       add a test that runs the benchmark program passing "debugsuite"
#       as first parameter
#
# Components:
#
#    LIB: includes blosc.so
#    DEV: static includes blosc.a and blosc.h
#    HDF5_FILTER: includes blosc_filter.so
#    HDF5_FILTER_DEV: includes blosc_filter.h


cmake_minimum_required(VERSION 2.8)
project(blosc)

# parse the full version numbers from blosc.h
file(READ ${CMAKE_CURRENT_SOURCE_DIR}/blosc/blosc.h _blosc_h_contents)
string(REGEX REPLACE ".*#define[ \t]+BLOSC_VERSION_MAJOR[ \t]+([0-9]+).*"
     "\\1" BLOSC_VERSION_MAJOR ${_blosc_h_contents})
string(REGEX REPLACE ".*#define[ \t]+BLOSC_VERSION_MINOR[ \t]+([0-9]+).*"
    "\\1" BLOSC_VERSION_MINOR ${_blosc_h_contents})
string(REGEX REPLACE ".*#define[ \t]+BLOSC_VERSION_RELEASE[ \t]+([0-9]+).*"
    "\\1" BLOSC_VERSION_PATCH ${_blosc_h_contents})
string(REGEX REPLACE ".*#define[ \t]+BLOSC_VERSION_STRING[ \t]+\"([-0-9A-Za-z.]+)\".*"
    "\\1" BLOSC_VERSION_STRING ${_blosc_h_contents})

message("Configuring for Blosc version: " ${BLOSC_VERSION_STRING})

# options
option(BUILD_STATIC
    "Build a static version of the blosc library." ON)
option(BUILD_HDF5_FILTER
    "Build a blosc based compression filter for the HDF5 library" OFF)
option(BUILD_TESTS
    "Build test programs form the blosc compression library" ON)
option(BUILD_BENCHMARKS
    "Build benchmark programs form the blosc compression library" ON)
option(DEACTIVATE_LZ4
    "Do not include support for the LZ4 library." OFF)
option(DEACTIVATE_SNAPPY
    "Do not include support for the SNAPPY library." OFF)
option(DEACTIVATE_ZLIB
    "Do not include support for the ZLIB library." OFF)
option(PREFER_EXTERNAL_COMPLIBS
    "When found, use the installed compression libs instead of included sources." ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")


if(NOT PREFER_EXTERNAL_COMPLIBS)
    message(STATUS "Finding external libraries disabled.  Using internal sources.")
endif(NOT PREFER_EXTERNAL_COMPLIBS)


if(NOT DEACTIVATE_LZ4)
    if(PREFER_EXTERNAL_COMPLIBS)
        find_package(LZ4)
    endif(PREFER_EXTERNAL_COMPLIBS)
    # HAVE_LZ4 will be set to true because even if the library is
    # not found, we will use the included sources for it
    set(HAVE_LZ4 TRUE)
endif(NOT DEACTIVATE_LZ4)

if(NOT DEACTIVATE_SNAPPY)
    if(PREFER_EXTERNAL_COMPLIBS)
        find_package(Snappy)
    endif(PREFER_EXTERNAL_COMPLIBS)
    # HAVE_SNAPPY will be set to true because even if the library is not found,
    # we will use the included sources for it
    set(HAVE_SNAPPY TRUE)
endif(NOT DEACTIVATE_SNAPPY)

if(NOT DEACTIVATE_ZLIB)
    # import the ZLIB_ROOT environment variable to help finding the zlib library
    if(PREFER_EXTERNAL_COMPLIBS)
        set(ZLIB_ROOT $ENV{ZLIB_ROOT})
        find_package( ZLIB )
        if (NOT ZLIB_FOUND )
            message(STATUS "No zlib found.  Using internal sources.")
        endif (NOT ZLIB_FOUND )
    endif(PREFER_EXTERNAL_COMPLIBS)
    # HAVE_ZLIB will be set to true because even if the library is not found,
    # we will use the included sources for it
    set(HAVE_ZLIB TRUE)
endif(NOT DEACTIVATE_ZLIB)

# create the config.h file
configure_file ("blosc/config.h.in"  "blosc/config.h" )
# now make sure that you set the build directory on your "Include" path when compiling
include_directories("${PROJECT_BINARY_DIR}/blosc/")

# force the default build type to Release.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
        FORCE)
endif(NOT CMAKE_BUILD_TYPE)


# flags
# @TODO: set -Wall
# @NOTE: -O3 is enabled in Release mode (CMAKE_BUILD_TYPE="Release")

# Set the "-msse2" build flag only if the CMAKE_C_FLAGS is not already set.
# Probably "-msse2" should be appended to CMAKE_C_FLAGS_RELEASE.
if(CMAKE_C_COMPILER_ID STREQUAL GNU OR CMAKE_C_COMPILER_ID STREQUAL Clang)
     if(NOT CMAKE_C_FLAGS)
         set(CMAKE_C_FLAGS -msse2 CACHE STRING "C flags." FORCE)
     endif(NOT CMAKE_C_FLAGS)
endif(CMAKE_C_COMPILER_ID STREQUAL GNU OR CMAKE_C_COMPILER_ID STREQUAL Clang)

if(MSVC)
    if(NOT CMAKE_C_FLAGS)
        set(CMAKE_C_FLAGS "/Ox" CACHE STRING "C flags." FORCE)
    endif(NOT CMAKE_C_FLAGS)
endif(MSVC)

if(WIN32)
    # For some supporting headers
    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/blosc")
endif(WIN32)


# subdirectories
add_subdirectory(blosc)

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif(BUILD_TESTS)

if(BUILD_HDF5_FILTER)
    add_subdirectory(hdf5)
endif(BUILD_HDF5_FILTER)

if(BUILD_BENCHMARKS)
    add_subdirectory(bench)
endif(BUILD_BENCHMARKS)


# uninstall target
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)


# packaging
include(InstallRequiredSystemLibraries)

set(CPACK_GENERATOR TGZ ZIP)
set(CPACK_SOURCE_GENERATOR TGZ ZIP)
set(CPACK_PACKAGE_VERSION_MAJOR ${BLOSC_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${BLOSC_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${BLOSC_VERSION_PATCH})
set(CPACK_PACKAGE_VERSION ${BLOSC_STRING_VERSION})
set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/README.rst")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "A blocking, shuffling and lossless compression library")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSES/BLOSC.txt")
set(CPACK_SOURCE_IGNORE_FILES "/build.*;.*~;\\\\.git.*;\\\\.DS_Store")
set(CPACK_STRIP_FILES TRUE)
set(CPACK_SOURCE_STRIP_FILES TRUE)

include(CPack)
