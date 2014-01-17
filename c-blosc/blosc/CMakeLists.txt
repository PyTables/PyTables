# a simple way to detect that we are using CMAKE
add_definitions(-DUSING_CMAKE)

set(INTERNAL_LIBS ${CMAKE_SOURCE_DIR}/internal-complibs)

# includes
if(NOT DEACTIVATE_LZ4)
    if (LZ4_FOUND)
        include_directories( ${LZ4_INCLUDE_DIR} )
    else(LZ4_FOUND)
        set(LZ4_LOCAL_DIR ${INTERNAL_LIBS}/lz4-r110)
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
set(SOURCES blosc.c blosclz.c shuffle.c)
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
        SOVERSION ${version_string}
    )
target_link_libraries(blosc_shared ${LIBS})

if(BUILD_STATIC)
    add_library(blosc_static STATIC ${SOURCES})
    set_target_properties(blosc_static PROPERTIES OUTPUT_NAME blosc)
    target_link_libraries(blosc_static ${LIBS})
endif(BUILD_STATIC)


# install
install(FILES blosc.h DESTINATION include COMPONENT DEV)
install(TARGETS blosc_shared DESTINATION ${lib_dir} COMPONENT LIB)
if(BUILD_STATIC)
    install(TARGETS blosc_static DESTINATION ${lib_dir} COMPONENT DEV)
endif(BUILD_STATIC)
