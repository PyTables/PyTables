# build the filegen utility
link_directories(${PROJECT_BINARY_DIR}/blosc)
add_executable(filegen filegen.c)
# have to copy blosc dlls on Windows
if (MSVC OR MINGW)
    add_custom_command(
            TARGET filegen
            POST_BUILD
            COMMAND ${CMAKE_COMMAND}
            ARGS -E copy_if_different
            "$<TARGET_FILE:blosc_shared>"
            "$<TARGET_FILE_DIR:filegen>/$<TARGET_FILE_NAME:blosc_shared>")
endif()
target_link_libraries(filegen blosc_shared)
add_dependencies(filegen blosc_shared)

# tests
if (BUILD_TESTS)
    option(TEST_INCLUDE_COMPAT "Include compat checks in the tests" ON)
    if (TEST_INCLUDE_COMPAT)
        file(GLOB DATAFILES *.cdata)
        foreach(datafile ${DATAFILES})
            # Don't test data if compressor is deactivated
            if((datafile MATCHES "lz4" AND DEACTIVATE_LZ4) OR
               (datafile MATCHES "snappy" AND DEACTIVATE_SNAPPY) OR
               (datafile MATCHES "zlib" AND DEACTIVATE_ZLIB) OR
               (datafile MATCHES "zstd" AND DEACTIVATE_ZSTD))
               continue()
            endif()
            get_filename_component(fname ${datafile} NAME)
            add_test(test_compat_${fname} filegen decompress ${datafile})
        endforeach(datafile)
    endif()
endif (BUILD_TESTS)
