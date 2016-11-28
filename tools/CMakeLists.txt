include(GNUInstallDirs)

find_package(Boost REQUIRED program_options)

function(add_tool name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} popnn)
endfunction()

add_tool(single_conv_layer single_conv_layer.cpp)
target_link_libraries(single_conv_layer
                      popnn_ref
                      Boost::program_options)

add_tool(fully_connected_layer fully_connected_layer.cpp)
target_link_libraries(fully_connected_layer
                      popnn_ref
                      Boost::program_options)

add_tool(max_pool_layer max_pool_layer.cpp)
target_link_libraries(max_pool_layer
                      popnn_ref
                      Boost::program_options)
