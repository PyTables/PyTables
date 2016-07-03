# a simple way to detect that we are using CMAKE
add_definitions(-DUSING_CMAKE)

set(INTERNAL_LIBS ${CMAKE_SOURCE_DIR}/internal-complibs)

# Hide symbols by default unless they're specifically exported.
# This makes it easier to keep the set of exported symbols the
# same across all compilers/platforms.
set(CMAKE_C_VISIBILITY_PRESET hidden)

# includes
if(NOT DEACTIVATE_LZ4)
    if (LZ4_FOUND)
        include_directories( ${LZ4_INCLUDE_DIR} )
    else(LZ4_FOUND)
        set(LZ4_LOCAL_DIR ${INTERNAL_LIBS}/lz4-1.7.2)
        include_directories( ${LZ4_LOCAL_DIR} )
    endif(LZ4_FOUND)
endif(NOT DEACTIVATE_LZ4)

if(NOT DEACTIVATE_SNAPPY)
    if (SNAPPY_FOUND)
        include_directories( ${SNAPPY_INCLUDE_DIR} )
    else(SNAPPY_FOUND)
        set(SNAPPY_LOCAL_DIR ${INTERNAL_LIBS}/snappy-1.1.1)
        include_directories( ${SNAPPY_LOCAL_DIR} )
    endif(SNAPPY_FOUND)
endif(NOT DEACTIVATE_SNAPPY)

if(NOT DEACTIVATE_ZLIB)
    if (ZLIB_FOUND)
        include_directories( ${ZLIB_INCLUDE_DIR} )
    else(ZLIB_FOUND)
        set(ZLIB_LOCAL_DIR ${INTERNAL_LIBS}/zlib-1.2.8)
        include_directories( ${ZLIB_LOCAL_DIR} )
    endif(ZLIB_FOUND)
endif(NOT DEACTIVATE_ZLIB)

# library sources
set(SOURCES blosc.c blosclz.c shuffle-generic.c bitshuffle-generic.c)
if(COMPILER_SUPPORT_SSE2)
    message(STATUS "Adding run-time support for SSE2")
    set(SOURCES ${SOURCES} shuffle-sse2.c bitshuffle-sse2.c)
endif(COMPILER_SUPPORT_SSE2)
if(COMPILER_SUPPORT_AVX2)
    message(STATUS "Adding run-time support for AVX2")
    set(SOURCES ${SOURCES} shuffle-avx2.c bitshuffle-avx2.c)
endif(COMPILER_SUPPORT_AVX2)
set(SOURCES ${SOURCES} shuffle.c)

# library install directory
set(lib_dir lib${LIB_SUFFIX})
set(version_string ${BLOSC_VERSION_MAJOR}.${BLOSC_VERSION_MINOR}.${BLOSC_VERSION_PATCH})

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
if(WIN32)
    # try to use the system library
    find_package(Threads)
    if(NOT Threads_FOUND)
        message(STATUS "using the internal pthread library for win32 systems.")
        set(SOURCES ${SOURCES} win32/pthread.c)
    else(NOT Threads_FOUND)
        set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})
    endif(NOT Threads_FOUND)
else(WIN32)
    find_package(Threads REQUIRED)
    set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})
endif(WIN32)

if(NOT DEACTIVATE_LZ4)
    if(LZ4_FOUND)
        set(LIBS ${LIBS} ${LZ4_LIBRARY})
    else(LZ4_FOUND)
        file(GLOB LZ4_FILES ${LZ4_LOCAL_DIR}/*.c)
        set(SOURCES ${SOURCES} ${LZ4_FILES})
    endif(LZ4_FOUND)
endif(NOT DEACTIVATE_LZ4)

if(NOT DEACTIVATE_SNAPPY)
    if(SNAPPY_FOUND)
        set(LIBS ${LIBS} ${SNAPPY_LIBRARY})
    else(SNAPPY_FOUND)
        file(GLOB SNAPPY_FILES ${SNAPPY_LOCAL_DIR}/*.cc)
        set(SOURCES ${SOURCES} ${SNAPPY_FILES})
    endif(SNAPPY_FOUND)
endif(NOT DEACTIVATE_SNAPPY)

if(NOT DEACTIVATE_ZLIB)
    if(ZLIB_FOUND)
        set(LIBS ${LIBS} ${ZLIB_LIBRARY})
    else(ZLIB_FOUND)
        file(GLOB ZLIB_FILES ${ZLIB_LOCAL_DIR}/*.c)
        set(SOURCES ${SOURCES} ${ZLIB_FILES})
    endif(ZLIB_FOUND)
endif(NOT DEACTIVATE_ZLIB)

# targets
add_library(blosc_shared SHARED ${SOURCES})
set_target_properties(blosc_shared PROPERTIES OUTPUT_NAME blosc)
set_target_properties(blosc_shared PROPERTIES
        VERSION ${version_string}
        SOVERSION 1  # Change this when an ABI change happens
    )
set_property(
    TARGET blosc_shared
    APPEND PROPERTY COMPILE_DEFINITIONS BLOSC_SHARED_LIBRARY)

# Based on the target architecture and hardware features supported
# by the C compiler, set hardware architecture optimization flags
# for specific shuffle implementations.
if(COMPILER_SUPPORT_SSE2)
    if (MSVC)
        # MSVC targets SSE2 by default on 64-bit configurations, but not 32-bit configurations.
        if (${CMAKE_SIZEOF_VOID_P} EQUAL 4)
            set_source_files_properties(shuffle-sse2.c bitshuffle-sse2.c PROPERTIES COMPILE_FLAGS "/arch:SSE2")
        endif (${CMAKE_SIZEOF_VOID_P} EQUAL 4)
    else (MSVC)
        set_source_files_properties(shuffle-sse2.c bitshuffle-sse2.c PROPERTIES COMPILE_FLAGS -msse2)
    endif (MSVC)

    # Define a symbol for the shuffle-dispatch implementation
    # so it knows SSE2 is supported even though that file is
    # compiled without SSE2 support (for portability).
    set_property(
        SOURCE shuffle.c
        APPEND PROPERTY COMPILE_DEFINITIONS SHUFFLE_SSE2_ENABLED)
endif(COMPILER_SUPPORT_SSE2)
if(COMPILER_SUPPORT_AVX2)
    if (MSVC)
        set_source_files_properties(shuffle-avx2.c bitshuffle-avx2.c PROPERTIES COMPILE_FLAGS "/arch:AVX2")
    else (MSVC)
        set_source_files_properties(shuffle-avx2.c bitshuffle-avx2.c PROPERTIES COMPILE_FLAGS -mavx2)
    endif (MSVC)

    # Define a symbol for the shuffle-dispatch implementation
    # so it knows AVX2 is supported even though that file is
    # compiled without AVX2 support (for portability).
    set_property(
        SOURCE shuffle.c
        APPEND PROPERTY COMPILE_DEFINITIONS SHUFFLE_AVX2_ENABLED)
endif(COMPILER_SUPPORT_AVX2)

# When the option has been selected to compile the test suite,
# compile an additional version of blosc_shared which exports
# some normally-hidden symbols (to facilitate unit testing).
if (BUILD_TESTS)
    add_library(blosc_shared_testing SHARED ${SOURCES})
    set_target_properties(blosc_shared_testing PROPERTIES OUTPUT_NAME blosc_testing)
    set_property(
        TARGET blosc_shared_testing
        APPEND PROPERTY COMPILE_DEFINITIONS BLOSC_SHARED_LIBRARY)
    set_property(
        TARGET blosc_shared_testing
        APPEND PROPERTY COMPILE_DEFINITIONS BLOSC_TESTING)
    # TEMP : CMake doesn't automatically add -lpthread here like it does
    # for the blosc_shared target. Force it for now.
    if(UNIX)
        set_property(
            TARGET blosc_shared_testing
            APPEND PROPERTY LINK_FLAGS "-lpthread")
    endif()
endif()

target_link_libraries(blosc_shared ${LIBS})
if (BUILD_TESTS)
    target_link_libraries(blosc_shared_testing ${LIBS})
endif()

if(BUILD_STATIC)
    add_library(blosc_static STATIC ${SOURCES})
    set_target_properties(blosc_static PROPERTIES OUTPUT_NAME blosc)
    if (MSVC)
        set_target_properties(blosc_static PROPERTIES PREFIX lib)
    endif()
    target_link_libraries(blosc_static ${LIBS})
endif(BUILD_STATIC)


# install
install(FILES blosc.h blosc-export.h DESTINATION include COMPONENT DEV)
install(TARGETS blosc_shared DESTINATION ${lib_dir} COMPONENT LIB)
if(BUILD_STATIC)
    install(TARGETS blosc_static DESTINATION ${lib_dir} COMPONENT DEV)
endif(BUILD_STATIC)
