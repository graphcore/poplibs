include(GNUInstallDirs)

find_package(Boost 1.65.1 REQUIRED program_options)

function(add_tool name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} poprand popnn poplin popops popfloat poputil)
  if(TARGET popsparse)
    target_link_libraries(${name} popsparse)
  endif()
  target_include_directories(${name} PRIVATE ${PROJECT_SOURCE_DIR}/tests)
endfunction()

add_tool(single_conv_layer single_conv_layer.cpp)
target_link_libraries(single_conv_layer
                      poplibs_test
                      Boost::program_options)

add_tool(oct_conv_layer oct_conv_layer.cpp)
target_link_libraries(oct_conv_layer
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

add_tool(gru_layer gru_layer.cpp)
target_link_libraries(gru_layer
                      poplibs_test
                      Boost::program_options)

if(TARGET popsparse)
  add_tool(bs_matmul_test bs_matmul_test.cpp)
  target_link_libraries(bs_matmul_test
    poplibs_test
    Boost::program_options)
endif()

add_tool(pooling_layer pooling_layer.cpp)
target_link_libraries(pooling_layer
                      poplibs_test
                      Boost::program_options)

add_tool(general_matrix_multiply general_matrix_multiply.cpp)
target_link_libraries(general_matrix_multiply
                      poplibs_test
                      Boost::program_options)

add_tool(cast_to_gfloat cast_to_gfloat.cpp)
target_link_libraries(cast_to_gfloat
                      poplibs_test
                      Boost::program_options)

add_tool(cast_to_gfloat_sr cast_to_gfloat_sr.cpp)
target_link_libraries(cast_to_gfloat_sr
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
                      poplibs_test
                      Boost::program_options)

add_tool(replicated_collectives replicated_collectives.cpp)
target_link_libraries(replicated_collectives
                      poplibs_test
                      Boost::program_options)

add_tool(random_generator random_generator.cpp)
target_link_libraries(random_generator
                     poplibs_test
                     Boost::program_options)

add_tool(embedding_layer embedding_layer.cpp)
target_link_libraries(embedding_layer
                      poplibs_support
                      poplibs_test
                      Boost::program_options)

add_tool(norm_layer norm_layer.cpp)
target_link_libraries(norm_layer
                      poplibs_test
                      Boost::program_options)
