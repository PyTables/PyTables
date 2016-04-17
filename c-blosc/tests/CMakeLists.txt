# sources
#aux_source_directory(. SOURCES)
file(GLOB SOURCES test_*.c)

# flags
link_directories(${PROJECT_BINARY_DIR}/blosc)

# targets and tests
foreach(source ${SOURCES})
    # Enable support for testing accelerated shuffles
    if(COMPILER_SUPPORT_SSE2)
        # Define a symbol so tests for SSE2 shuffle/unshuffle will be compiled in.
        set_property(
            SOURCE ${source}
            APPEND PROPERTY COMPILE_DEFINITIONS SHUFFLE_SSE2_ENABLED)
    endif(COMPILER_SUPPORT_SSE2)
#    if(COMPILER_SUPPORT_AVX2)
#        # Define a symbol so tests for AVX2 shuffle/unshuffle will be compiled in.
#        set_property(
#            SOURCE ${source}
#            APPEND PROPERTY COMPILE_DEFINITIONS SHUFFLE_AVX2_ENABLED)
#    endif(COMPILER_SUPPORT_AVX2)

    get_filename_component(target ${source} NAME_WE)
    add_executable(${target} ${source})

    # Define the BLOSC_TESTING symbol so normally-hidden functions
    # aren't hidden from the view of the test programs.
    set_property(
        TARGET ${target}
        APPEND PROPERTY COMPILE_DEFINITIONS BLOSC_TESTING)

    # have to copy dlls for Visual Studio
    if(MSVC)
        add_custom_command(
            TARGET      ${target}
            POST_BUILD
            COMMAND     ${CMAKE_COMMAND}
            ARGS        -E copy_if_different
                        "${PROJECT_BINARY_DIR}/blosc/\$\(Configuration\)/blosc_testing.dll"
                        "${CMAKE_CURRENT_BINARY_DIR}/\$\(Configuration\)/blosc_testing.dll")
    elseif(MINGW)
        add_custom_command(
            TARGET      ${target}
            POST_BUILD
            COMMAND     ${CMAKE_COMMAND}
            ARGS        -E copy_if_different
                        "${PROJECT_BINARY_DIR}/blosc/libblosc_testing.dll"
                        "${CMAKE_CURRENT_BINARY_DIR}/libblosc_testing.dll")
    endif()

    target_link_libraries(${target} blosc_testing)

    # If there's a CSV file present for this test, read it to get the list
    # of test parameters then add a test for each parameter set.
    # Otherwise, this is a simple test so just add it once.
    get_filename_component(source_extension ${source} EXT)
    string(REGEX REPLACE "${source_extension}$" ".csv"
        test_params_file ${source})
    if (EXISTS "${test_params_file}")
        # Read the file contents into a CMake list
        file(READ "${test_params_file}" test_params_contents)

        string(REGEX REPLACE ";" "\\\\;"
            test_params_contents "${test_params_contents}")
        string(REGEX REPLACE "\n" ";"
            test_params_contents "${test_params_contents}")

        # How many parameter sets for this test?
        # If there's not at least one (accounting for the CSV header line),
        # that's probably not correct so emit an error and stop configuring.
        list(LENGTH test_params_contents test_params_count)
        if ("${test_params_count}" LESS 2)
            message(ERROR "Invalid test parameters file: ${test_params_file}")
        endif()

        # Remove the header line.
        list(REMOVE_AT test_params_contents 0)

        # Add a test for each parameter set in the file.
        foreach(test_params_raw ${test_params_contents})
            string(REGEX REPLACE "," " " test_params "${test_params_raw}")

            # Create the test name.
            # NOTE: The documentation for add_test says the test name "may not contain
            # spaces, quotes, or other characters special in CMake syntax."
            string(REGEX REPLACE "\"| " "_" test_name_params "${test_params}")
            set(test_name "${target}_${test_name_params}")

            separate_arguments(test_params)
            add_test(${test_name} ${target} ${test_params})
        endforeach()
    else()
        add_test(${target} ${target})
    endif()
endforeach(source)
