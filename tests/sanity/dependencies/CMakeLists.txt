set(poplib_libraries "poplibs_support;poputil;popops;poprand;popfloat;poplin;popnn;popsparse")

# Test that we only include files against linked libraries
# CMake deals with cyclic dependencies as long as we are upfront with dependencies
set(arg_string "")
foreach (lib IN LISTS poplib_libraries)
    get_target_property(deps_semi_colon_sep ${lib} LINK_LIBRARIES)
    string(REPLACE "-" "" deps_semi_colon_sep "${deps_semi_colon_sep}") # So python isn't confused
    string(APPEND arg_string "-d;${lib};${deps_semi_colon_sep};")
endforeach()

add_test(
    NAME poplibs_cyclic_dependency
    COMMAND ${PYTHON_EXECUTABLE}
            ${CMAKE_SOURCE_DIR}/tests/sanity/dependencies/check_cmake_link_include.py
            ${CMAKE_SOURCE_DIR}
            ${arg_string}
)
# T22741: There is currently a cyclic dependency with option parsing in plan constraints
set_tests_properties(poplibs_cyclic_dependency PROPERTIES DISABLED TRUE)
set(poplibs_cyclic_dependency_LABELS "python_tool")
add_parent_dir_labels(poplibs_cyclic_dependency_LABELS)
if (poplibs_cyclic_dependency_LABELS)
    set_tests_properties(poplibs_cyclic_dependency
        PROPERTIES LABELS "${poplibs_cyclic_dependency_LABELS}")
endif()
