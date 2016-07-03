# sources
set(SOURCES bench.c)


# targets
add_executable(bench ${SOURCES})
if(UNIX AND NOT APPLE)
  # cmake is complaining about LINK_PRIVATE in original PR
  # and removing it does not seem to hurt, so be it.
  # target_link_libraries(bench LINK_PRIVATE rt)
  target_link_libraries(bench rt)
endif(UNIX AND NOT APPLE)
target_link_libraries(bench blosc_shared)

# have to copy blosc dlls on Windows
if(MSVC)
    add_custom_command(
        TARGET      bench
        POST_BUILD
        COMMAND     ${CMAKE_COMMAND}
        ARGS        -E copy_if_different
                    "${PROJECT_BINARY_DIR}/blosc/\$\(Configuration\)/blosc.dll"
                    "${CMAKE_CURRENT_BINARY_DIR}/\$\(Configuration\)/blosc.dll")
elseif(MINGW)
    add_custom_command(
        TARGET      bench
        POST_BUILD
        COMMAND     ${CMAKE_COMMAND}
        ARGS        -E copy_if_different
                    "${PROJECT_BINARY_DIR}/blosc/libblosc.dll"
                    "${CMAKE_CURRENT_BINARY_DIR}/libblosc.dll")
endif()

# tests
if(BUILD_TESTS)

    # The commented tests below take too much time to complete
    option(TEST_INCLUDE_BENCH_SHUFFLE_1 "Include bench shuffle (1 thread) in the tests" ON)
    if(TEST_INCLUDE_BENCH_SHUFFLE_1)
        set(SHUFFLE_1_OPTS shuffle test 1)
        add_test(test_blosclz_shuffle_1 bench blosclz ${SHUFFLE_1_OPTS})
        if (HAVE_LZ4)
            add_test(test_lz4_shuffle_1 bench lz4 ${SHUFFLE_1_OPTS})
            # add_test(test_lz4hc_shuffle_1 bench lz4hc ${SHUFFLE_1_OPTS})
        endif (HAVE_LZ4)
        if (HAVE_SNAPPY)
            add_test(test_snappy_shuffle_1 bench snappy ${SHUFFLE_1_OPTS})
        endif (HAVE_SNAPPY)
        if (HAVE_ZLIB)
            # add_test(test_zlib_shuffle_1 bench zlib ${SHUFFLE_1_OPTS})
        endif (HAVE_ZLIB)
    endif(TEST_INCLUDE_BENCH_SHUFFLE_1)

    option(TEST_INCLUDE_BENCH_SHUFFLE_N "Include bench shuffle (multithread) in the tests" ON)
    if(TEST_INCLUDE_BENCH_SHUFFLE_N)
        set(SHUFFLE_N_OPTS shuffle test)
        add_test(test_blosclz_shuffle_n bench blosclz ${SHUFFLE_N_OPTS})
        if (HAVE_LZ4)
            add_test(test_lz4_shuffle_n bench lz4 ${SHUFFLE_N_OPTS})
            add_test(test_lz4hc_shuffle_n bench lz4hc ${SHUFFLE_N_OPTS})
        endif (HAVE_LZ4)
        if (HAVE_SNAPPY)
            add_test(test_snappy_shuffle_n bench snappy ${SHUFFLE_N_OPTS})
        endif (HAVE_SNAPPY)
        if (HAVE_ZLIB)
            add_test(test_zlib_shuffle_n bench zlib ${SHUFFLE_N_OPTS})
        endif (HAVE_ZLIB)
    endif(TEST_INCLUDE_BENCH_SHUFFLE_N)

    option(TEST_INCLUDE_BENCH_BITSHUFFLE_1 "Include bench bitshuffle (1 thread) in the tests" ON)
    if(TEST_INCLUDE_BENCH_BITSHUFFLE_1)
        set(BITSHUFFLE_1_OPTS bitshuffle test 1)
        add_test(test_blosclz_bitshuffle_1 bench blosclz ${BITSHUFFLE_1_OPTS})
        if (HAVE_LZ4)
            add_test(test_lz4_bitshuffle_1 bench lz4 ${BITSHUFFLE_1_OPTS})
            # add_test(test_lz4hc_bitshuffle_1 bench lz4hc ${BITSHUFFLE_1_OPTS})
        endif (HAVE_LZ4)
        if (HAVE_SNAPPY)
            add_test(test_snappy_bitshuffle_1 bench snappy ${BITSHUFFLE_1_OPTS})
        endif (HAVE_SNAPPY)
        if (HAVE_ZLIB)
            # add_test(test_zlib_bitshuffle_1 bench zlib ${BITSHUFFLE_1_OPTS})
        endif (HAVE_ZLIB)
    endif(TEST_INCLUDE_BENCH_BITSHUFFLE_1)

    option(TEST_INCLUDE_BENCH_BITSHUFFLE_N "Include bench bitshuffle (multithread) in the tests" ON)
    if(TEST_INCLUDE_BENCH_BITSHUFFLE_N)
        set(BITSHUFFLE_N_OPTS bitshuffle test)
        add_test(test_blosclz_bitshuffle_n bench blosclz ${BITSHUFFLE_N_OPTS})
        if (HAVE_LZ4)
            add_test(test_lz4_bitshuffle_n bench lz4 ${BITSHUFFLE_N_OPTS})
            # add_test(test_lz4hc_bitshuffle_n bench lz4hc ${BITSHUFFLE_N_OPTS})
        endif (HAVE_LZ4)
        if (HAVE_SNAPPY)
            add_test(test_snappy_bitshuffle_n bench snappy ${BITSHUFFLE_N_OPTS})
        endif (HAVE_SNAPPY)
        if (HAVE_ZLIB)
            # add_test(test_zlib_bitshuffle_n bench zlib ${BITSHUFFLE_N_OPTS})
        endif (HAVE_ZLIB)
    endif(TEST_INCLUDE_BENCH_BITSHUFFLE_N)

    option(TEST_INCLUDE_BENCH_SUITE "Include bench suite in the tests" OFF)
    if(TEST_INCLUDE_BENCH_SUITE)
        add_test(test_hardsuite blosc blosclz shuffle suite)
    endif(TEST_INCLUDE_BENCH_SUITE)

    option(TEST_INCLUDE_BENCH_DEBUGSUITE "Include bench debugsuite in the tests" OFF)
    if(TEST_INCLUDE_BENCH_DEBUGSUITE)
        add_test(test_debugsuite bench blosclz shuffle debugsuite)
    endif(TEST_INCLUDE_BENCH_DEBUGSUITE)
endif(BUILD_TESTS)
