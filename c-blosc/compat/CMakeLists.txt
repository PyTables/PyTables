# build the filegen utility
link_directories(${PROJECT_BINARY_DIR}/blosc)
add_executable(filegen filegen.c)
# have to copy blosc dlls on Windows
if (MSVC)
    add_custom_command(
            TARGET filegen
            POST_BUILD
            COMMAND ${CMAKE_COMMAND}
            ARGS -E copy_if_different
            "${PROJECT_BINARY_DIR}/blosc/\$\(Configuration\)/blosc.dll"
            "${CMAKE_CURRENT_BINARY_DIR}/\$\(Configuration\)/blosc.dll")
elseif (MINGW)
    add_custom_command(
            TARGET filegen
            POST_BUILD
            COMMAND ${CMAKE_COMMAND}
            ARGS -E copy_if_different
            "${PROJECT_BINARY_DIR}/blosc/libblosc.dll"
            "${CMAKE_CURRENT_BINARY_DIR}/libblosc.dll")
endif(MSVC)
target_link_libraries(filegen blosc_shared)
add_dependencies(filegen blosc_shared)

# tests
if (BUILD_TESTS)
    option(TEST_INCLUDE_COMPAT "Include compat checks in the tests" ON)
    if (TEST_INCLUDE_COMPAT)
        file(GLOB DATAFILES *.cdata)
        foreach(datafile ${DATAFILES})
            get_filename_component(fname ${datafile} NAME)
            add_test(test_compat_${fname} filegen decompress ${datafile})
        endforeach(datafile)
    endif()
endif (BUILD_TESTS)
