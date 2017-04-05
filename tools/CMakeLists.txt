include(GNUInstallDirs)

find_package(Boost REQUIRED program_options)

function(add_tool name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} popnn poplin popconv popreduce popstd)
endfunction()

add_tool(single_conv_layer single_conv_layer.cpp)
target_link_libraries(single_conv_layer
                      poplib_test
                      Boost::program_options)

add_tool(fully_connected_layer fully_connected_layer.cpp)
target_link_libraries(fully_connected_layer
                      poplib_test
                      Boost::program_options)


add_tool(lstm_layer lstm_layer.cpp)
target_link_libraries(lstm_layer
                      poplib_test
                      Boost::program_options)

add_tool(max_pool_layer max_pool_layer.cpp)
target_link_libraries(max_pool_layer
                      poplib_test
                      Boost::program_options)

add_tool(general_matrix_multiply general_matrix_multiply.cpp)
target_link_libraries(general_matrix_multiply
                      poplib_test
                      Boost::program_options)

add_tool(rnn_layer rnn_layer.cpp)
target_link_libraries(rnn_layer
                      poplib_test
                      Boost::program_options)
