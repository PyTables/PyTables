# CMake build system for Blosc
# ============================
#
# Available options:
#
#   BUILD_STATIC: default ON
#       build the static version of the Blosc library
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
#   PREFER_EXTERNAL_LZ4: default OFF
#       when found, use the installed LZ4 libs instead of included
#       sources
#   PREFER_EXTERNAL_SNAPPY: default ON
#       when found, use the installed Snappy libs instead of included
#       sources
#   PREFER_EXTERNAL_ZLIB: default ON
#       when found, use the installed zlib libs instead of included
#       sources
#   TEST_INCLUDE_BENCH_SHUFFLE_1: default ON
#       add a test that runs the benchmark program passing "shuffle" with 1
#       thread as second parameter
#   TEST_INCLUDE_BENCH_SHUFFLE_N: default ON
#       add a test that runs the benchmark program passing "shuffle" with all
#       threads as second parameter
#   TEST_INCLUDE_BENCH_BITSHUFFLE_1: default ON
#       add a test that runs the benchmark program passing "bitshuffle" with 1
#       thread as second parameter
#   TEST_INCLUDE_BENCH_BITSHUFFLE_N: default ON
#       add a test that runs the benchmark program passing "bitshuffle" with
#       all threads as second parameter
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


cmake_minimum_required(VERSION 2.8.10)
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
option(PREFER_EXTERNAL_LZ4
    "Find and use external LZ4 library instead of included sources." OFF)
option(PREFER_EXTERNAL_SNAPPY
    "Find and use external Snappy library instead of included sources." ON)
option(PREFER_EXTERNAL_ZLIB
    "Find and use external zlib library instead of included sources." ON)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")


if(NOT DEACTIVATE_LZ4)
    if(PREFER_EXTERNAL_LZ4)
        find_package(LZ4)
    else()
        message(STATUS "Using LZ4 internal sources.")
    endif(PREFER_EXTERNAL_LZ4)
    # HAVE_LZ4 will be set to true because even if the library is
    # not found, we will use the included sources for it
    set(HAVE_LZ4 TRUE)
endif(NOT DEACTIVATE_LZ4)

if(NOT DEACTIVATE_SNAPPY)
    if(PREFER_EXTERNAL_SNAPPY)
        find_package(Snappy)
    else()
        message(STATUS "Using Snappy internal sources.")
    endif(PREFER_EXTERNAL_SNAPPY)
    # HAVE_SNAPPY will be set to true because even if the library is not found,
    # we will use the included sources for it
    set(HAVE_SNAPPY TRUE)
endif(NOT DEACTIVATE_SNAPPY)

if(NOT DEACTIVATE_ZLIB)
    # import the ZLIB_ROOT environment variable to help finding the zlib library
    if(PREFER_EXTERNAL_ZLIB)
        set(ZLIB_ROOT $ENV{ZLIB_ROOT})
        find_package(ZLIB)
        if (NOT ZLIB_FOUND )
            message(STATUS "No zlib found.  Using internal sources.")
        endif (NOT ZLIB_FOUND )
    else()
        message(STATUS "Using zlib internal sources.")
    endif(PREFER_EXTERNAL_ZLIB)
    # HAVE_ZLIB will be set to true because even if the library is not found,
    # we will use the included sources for it
    set(HAVE_ZLIB TRUE)
endif(NOT DEACTIVATE_ZLIB)

# create the config.h file
configure_file ("blosc/config.h.in"  "blosc/config.h" )

# now make sure that you set the build directory on your "Include" path when compiling
include_directories("${PROJECT_BINARY_DIR}/blosc/")

# If the build type is not set, default to Release.
set(BLOSC_DEFAULT_BUILD_TYPE Release)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type specified. Defaulting to '${BLOSC_DEFAULT_BUILD_TYPE}'.")
    set(CMAKE_BUILD_TYPE ${BLOSC_DEFAULT_BUILD_TYPE} CACHE STRING
        "Choose the type of build." FORCE)

    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Based on the target system's processor and the compiler being used,
# set build variables indicating which hardware features can be targeted
# by the compiler. Note we DO NOT check which hardware features are supported
# by this (the host) system, because we want to be able to support compiling
# for newer hardware on older machines as well as cross-compilation.
message(STATUS "Building for system processor ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR STREQUAL i386 OR
   CMAKE_SYSTEM_PROCESSOR STREQUAL i686 OR
   CMAKE_SYSTEM_PROCESSOR STREQUAL x86_64 OR
   CMAKE_SYSTEM_PROCESSOR STREQUAL amd64 OR
   CMAKE_SYSTEM_PROCESSOR STREQUAL AMD64)
    if(CMAKE_C_COMPILER_ID STREQUAL GNU)
        set(COMPILER_SUPPORT_SSE2 TRUE)
        if(CMAKE_C_COMPILER_VERSION VERSION_GREATER 4.7 OR CMAKE_C_COMPILER_VERSION VERSION_EQUAL 4.7)
            set(COMPILER_SUPPORT_AVX2 TRUE)
        else()
            set(COMPILER_SUPPORT_AVX2 FALSE)
        endif()
    elseif(CMAKE_C_COMPILER_ID STREQUAL Clang)
        set(COMPILER_SUPPORT_SSE2 TRUE)
        if(CMAKE_C_COMPILER_VERSION VERSION_GREATER 3.2 OR CMAKE_C_COMPILER_VERSION VERSION_EQUAL 3.2)
            set(COMPILER_SUPPORT_AVX2 TRUE)
        else()
            set(COMPILER_SUPPORT_AVX2 FALSE)
        endif()
    elseif(CMAKE_C_COMPILER_ID STREQUAL Intel)
        set(COMPILER_SUPPORT_SSE2 TRUE)
        if(CMAKE_C_COMPILER_VERSION VERSION_GREATER 14.0 OR CMAKE_C_COMPILER_VERSION VERSION_EQUAL 14.0)
	    # icc (ICC) 15.0.3 does not work compiling AVX2 code
	    # (perhaps my machine does not have AVX2 and the compiler
	    # cannot generate code for that?)
            set(COMPILER_SUPPORT_AVX2 FALSE)
        else()
            set(COMPILER_SUPPORT_AVX2 FALSE)
        endif()
    elseif(MSVC)
        set(COMPILER_SUPPORT_SSE2 TRUE)
        if(CMAKE_C_COMPILER_VERSION VERSION_GREATER 18.00.30501 OR CMAKE_C_COMPILER_VERSION VERSION_EQUAL 18.00.30501)
            set(COMPILER_SUPPORT_AVX2 TRUE)
        else()
            set(COMPILER_SUPPORT_AVX2 FALSE)
        endif()
    else()
        set(COMPILER_SUPPORT_SSE2 FALSE)
        set(COMPILER_SUPPORT_AVX2 FALSE)
        # Unrecognized compiler. Emit a warning message to let the user know hardware-acceleration won't be available.
        message(WARNING "Unable to determine which ${CMAKE_SYSTEM_PROCESSOR} hardware features are supported by the C compiler (${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}).")
    endif()
else()
    # If the target system processor isn't recognized, emit a warning message to alert the user
    # that hardware-acceleration support won't be available but allow configuration to proceed.
    message(WARNING "Unrecognized system processor ${CMAKE_SYSTEM_PROCESSOR}. Cannot determine which hardware features (${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}) supports, so hardware-accelerated implementations will not be available.")
endif()

# flags
# @TODO: set -Wall
# @NOTE: -O3 is enabled in Release mode (CMAKE_BUILD_TYPE="Release")

# Set the "-msse2" build flag only if the CMAKE_C_FLAGS is not already set.
# Probably "-msse2" should be appended to CMAKE_C_FLAGS_RELEASE.
if(CMAKE_C_COMPILER_ID STREQUAL GNU OR CMAKE_C_COMPILER_ID STREQUAL Clang OR CMAKE_C_COMPILER_ID STREQUAL Intel)
     if(NOT CMAKE_C_FLAGS AND COMPILER_SUPPORT_SSE2)
         set(CMAKE_C_FLAGS -msse2 CACHE STRING "C flags." FORCE)
     endif(NOT CMAKE_C_FLAGS AND COMPILER_SUPPORT_SSE2)
endif(CMAKE_C_COMPILER_ID STREQUAL GNU OR CMAKE_C_COMPILER_ID STREQUAL Clang OR CMAKE_C_COMPILER_ID STREQUAL Intel)

if(MSVC)
    if(NOT CMAKE_C_FLAGS)
        set(CMAKE_C_FLAGS "/Ox" CACHE STRING "C flags." FORCE)
    endif(NOT CMAKE_C_FLAGS)

    # Turn off misguided "secure CRT" warnings in MSVC.
    # Microsoft wants people to use the MS-specific <function>_s
    # versions of certain C functions but this is difficult to do
    # in platform-independent code.
    add_definitions( -D_CRT_SECURE_NO_WARNINGS )
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
