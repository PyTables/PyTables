cmake_minimum_required(VERSION 2.8.10)
cmake_policy(SET CMP0074 NEW)
project(blosc_hdf5)
include(ExternalProject)
include(GNUInstallDirs)

# options
option(BUILD_TESTS
    "Build test programs form the blosc filter" ON)

option(BUILD_PLUGIN
    "Build dynamically loadable plugin for HDF5 version > 1.8.11" ON)
if(BUILD_PLUGIN)
    set(PLUGIN_INSTALL_PATH "/usr/local/hdf5/lib/plugin" CACHE PATH
      "Where to install the dynamic HDF5-plugin")
endif(BUILD_PLUGIN)

set(BLOSC_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/blosc")
set(BLOSC_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/blosc")
set(BLOSC_CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${BLOSC_INSTALL_DIR})

message("BLOSC_PREFIX='${BLOSC_PREFIX}'")
message("BLOSC_INSTALL_DIR='${BLOSC_INSTALL_DIR}'")
message("BLOSC_CMAKE_ARGS='${BLOSC_CMAKE_ARGS}'")
message("GIT_EXECUTABLE='${GIT_EXECUTABLE}'")

ExternalProject_Add(project_blosc
  PREFIX ${BLOSC_PREFIX}
  GIT_REPOSITORY https://github.com/Blosc/c-blosc.git
  GIT_TAG main
  INSTALL_DIR ${BLOSC_INSTALL_DIR}
  CMAKE_ARGS ${BLOSC_CMAKE_ARGS}
)


# sources
set(SOURCES src/blosc_filter.c)
set(PLUGIN_SOURCES src/blosc_filter.c src/blosc_plugin.c )

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


# add blosc libraries
add_library(blosc_shared SHARED IMPORTED)
set_property(TARGET blosc_shared PROPERTY IMPORTED_LOCATION ${BLOSC_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}/${CMAKE_SHARED_LIBRARY_PREFIX}blosc${CMAKE_SHARED_LIBRARY_SUFFIX})
add_dependencies(blosc_shared project_blosc)
include_directories(${BLOSC_INSTALL_DIR}/include)

add_library(blosc_filter_shared SHARED ${SOURCES})
set_target_properties(
  blosc_filter_shared PROPERTIES OUTPUT_NAME blosc_filter)
target_link_libraries(blosc_filter_shared blosc_shared ${HDF5_LIBRARIES})

if(BUILD_PLUGIN)
    add_library(blosc_plugin_shared SHARED ${PLUGIN_SOURCES})
    set_target_properties(
      blosc_plugin_shared PROPERTIES OUTPUT_NAME H5Zblosc)
    target_link_libraries(blosc_plugin_shared blosc_shared ${HDF5_LIBRARIES})

    install(TARGETS blosc_plugin_shared DESTINATION ${PLUGIN_INSTALL_PATH} COMPONENT HDF5_FILTER_DEV)
endif(BUILD_PLUGIN)

# install
install(FILES src/blosc_filter.h DESTINATION include COMPONENT HDF5_FILTER_DEV)
install(TARGETS blosc_filter_shared DESTINATION lib COMPONENT HDF5_FILTER_DEV)


# test
message("LINK LIBRARIES='blosc_filter_shared ${HDF5_LIBRARIES}'")
if(BUILD_TESTS)
    enable_testing()
    set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
    find_package(Threads REQUIRED)
    set(LIBS ${LIBS} ${CMAKE_THREAD_LIBS_INIT})
    add_executable(example src/example.c)
    target_link_libraries(example blosc_filter_shared ${HDF5_LIBRARIES} ${LIBS})
    add_test(test_hdf5_filter example)
endif(BUILD_TESTS)
