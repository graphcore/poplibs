find_package(PythonInterp 3)

include(GNUInstallDirs)

function(add_tool name)
  add_executable(${name} ${ARGN})
  TargetAddMarchExtensions(${name})
  target_link_libraries(${name} poprand popnn poplin popops popfloat poputil Boost::filesystem)
  if(TARGET popsparse)
    target_link_libraries(${name} popsparse)
  endif()
  install(TARGETS "${name}"
          DESTINATION internal_tools
          COMPONENT internal_tools)
endfunction()

set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_INSTALL_RPATH $ORIGIN)

add_tool(single_conv_layer single_conv_layer.cpp)
target_link_libraries(single_conv_layer
                      poplibs_support poplibs_test
                      Boost::program_options)

add_tool(multi_conv_layer multi_conv_layer.cpp)
target_link_libraries(multi_conv_layer
                      poplibs_support poplibs_test
                      Boost::program_options)

add_tool(oct_conv_layer oct_conv_layer.cpp)
target_link_libraries(oct_conv_layer
                      poplibs_support poplibs_test
                      Boost::program_options)

add_tool(fully_connected_layer fully_connected_layer.cpp)
target_link_libraries(fully_connected_layer
                      poplibs_support poplibs_test
                      Boost::program_options)


add_tool(lstm_layer lstm_layer.cpp)
target_link_libraries(lstm_layer
                      poplibs_support poplibs_test
                      Boost::program_options)

add_tool(gru_layer gru_layer.cpp)
target_link_libraries(gru_layer
                      poplibs_support poplibs_test
                      Boost::program_options)

if(TARGET popsparse)
  add_tool(bs_matmul_test bs_matmul_test.cpp)
  target_link_libraries(bs_matmul_test
    poplibs_support poplibs_test
    spdlog::spdlog_header_only
    Boost::program_options)
endif()

add_tool(pooling_layer pooling_layer.cpp)
target_link_libraries(pooling_layer
                      poplibs_support poplibs_test
                      Boost::program_options)

add_tool(general_matrix_multiply general_matrix_multiply.cpp)
target_link_libraries(general_matrix_multiply
                      poplibs_support poplibs_test
                      Boost::program_options)

add_tool(cast_to_gfloat cast_to_gfloat.cpp)
target_link_libraries(cast_to_gfloat
                      poplibs_support poplibs_test
                      Boost::program_options)

add_tool(cast_to_gfloat_sr cast_to_gfloat_sr.cpp)
target_link_libraries(cast_to_gfloat_sr
                      poplibs_support poplibs_test
                      Boost::program_options)

add_tool(rnn_layer rnn_layer.cpp)
target_link_libraries(rnn_layer
                      poplibs_support poplibs_test
                      Boost::program_options)

add_tool(reduce_op reduce_op.cpp)
target_link_libraries(reduce_op
                      poplibs_support poplibs_test
                      Boost::program_options)

add_tool(random_generator random_generator.cpp)
target_link_libraries(random_generator
                     poplibs_support poplibs_test
                     Boost::program_options)

add_tool(embedding_layer embedding_layer.cpp)
target_link_libraries(embedding_layer
                      poplibs_support
                      poplibs_test
                      Boost::program_options)

add_tool(norm_layer norm_layer.cpp)
target_link_libraries(norm_layer
                      poplibs_support poplibs_test
                      Boost::program_options)

add_tool(ctc_loss ctc_loss.cpp)
target_link_libraries(ctc_loss
                      poplibs_support poplibs_test
                      Boost::program_options)

add_tool(ctc_beam_search ctc_beam_search.cpp)
target_link_libraries(ctc_beam_search
                      poplibs_support poplibs_test
                      Boost::program_options)

add_tool(ctc_model_validate ctc_model_validate.cpp)
target_link_libraries(ctc_model_validate
                      poplibs_support poplibs_test
                      Boost::program_options)

add_tool(topk topk.cpp)
target_link_libraries(topk
                      poplibs_support poplibs_test
                      Boost::program_options)

add_tool(matrix_solver matrix_solver.cpp)
target_link_libraries(matrix_solver
                      poplibs_support poplibs_test
                      Boost::program_options)

add_tool(consistent_executable_tool consistent_executable_tool.cpp)
target_link_libraries(consistent_executable_tool poplibs_support)

if (TARGET popsparse)
  add_tool(sparse_fc_layer sparse_fc_layer.cpp)
  target_link_libraries(sparse_fc_layer
                        poplibs_support
                        poplibs_test
                        popsolver
                        spdlog::spdlog_header_only
                        Boost::program_options)

  add_tool(sparse_matmul sparse_matmul.cpp)
  target_link_libraries(sparse_matmul
                        poplibs_support
                        poplibs_test
                        Boost::program_options)

  add_tool(sparse_embedding_tied_matmul sparse_embedding_tied_matmul.cpp)
  target_link_libraries(sparse_embedding_tied_matmul
                        poplibs_support
                        poplibs_test
                        Boost::program_options)
endif()
