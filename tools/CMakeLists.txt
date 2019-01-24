include(GNUInstallDirs)

find_package(Boost REQUIRED program_options)

function(add_tool name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} popnn poplin popops poputil)
  target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/tests)
endfunction()

add_tool(single_conv_layer single_conv_layer.cpp)
target_link_libraries(single_conv_layer
                      poplibs_test
                      Boost::program_options)

add_tool(fully_connected_layer fully_connected_layer.cpp)
target_link_libraries(fully_connected_layer
                      poplibs_test
                      Boost::program_options)


add_tool(lstm_layer lstm_layer.cpp)
target_link_libraries(lstm_layer
                      poplibs_test
                      Boost::program_options)

add_tool(pooling_layer pooling_layer.cpp)
target_link_libraries(pooling_layer
                      poplibs_test
                      Boost::program_options)

add_tool(general_matrix_multiply general_matrix_multiply.cpp)
target_link_libraries(general_matrix_multiply
                      poplibs_test
                      Boost::program_options)

add_tool(rnn_layer rnn_layer.cpp)
target_link_libraries(rnn_layer
                      poplibs_test
                      Boost::program_options)

add_tool(reduce_op reduce_op.cpp)
target_link_libraries(reduce_op
                      poplibs_test
                      Boost::program_options)

add_tool(collectives collectives.cpp)
target_link_libraries(collectives
                      popsys
                      poplibs_test
                      Boost::program_options)
