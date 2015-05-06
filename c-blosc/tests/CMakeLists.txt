# sources
#aux_source_directory(. SOURCES)
file(GLOB SOURCES test_*.c)

# flags
link_directories(${PROJECT_BINARY_DIR}/blosc)

# targets and tests
foreach(source ${SOURCES})
    get_filename_component(target ${source} NAME_WE)
    add_executable(${target} ${source})
    target_link_libraries(${target} blosc_shared)
    add_test(test_${target} ${target})
endforeach(source)
