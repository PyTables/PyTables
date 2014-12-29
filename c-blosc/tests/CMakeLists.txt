# sources
#aux_source_directory(. SOURCES)
file(GLOB SOURCES test_*.c)

# flags
link_directories(${PROJECT_BINARY_DIR}/blosc)

# targets and tests
foreach(source ${SOURCES})
    get_filename_component(target ${source} NAME_WE)
    add_executable(${target} ${source})

    # have to copy dlls for Visual Studio
    if(MSVC)
        add_custom_command(
            TARGET      ${target}
            POST_BUILD
            COMMAND     ${CMAKE_COMMAND}
            ARGS        -E copy_if_different
                        "${PROJECT_BINARY_DIR}/blosc/\$\(Configuration\)/blosc.dll"
                        "${CMAKE_CURRENT_BINARY_DIR}/\$\(Configuration\)/blosc.dll")   
    endif(MSVC)

    target_link_libraries(${target} blosc_shared)
    add_test(test_${target} ${target})
endforeach(source)
