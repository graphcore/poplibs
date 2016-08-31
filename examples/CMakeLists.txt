include(GNUInstallDirs)

function(add_example name)
  add_executable(${name} ${ARGN})
  target_include_directories(${name}
    PRIVATE
      ${CMAKE_SOURCE_DIR}/include
      ${Boost_INCLUDE_DIRS})
  target_link_libraries(${name} popnn)
endfunction()

add_example(conv_mnist
  conv_mnist.cpp
  mnist.cpp
  mnist.h)

add_example(fc_mnist
  fc_mnist.cpp
  mnist.cpp
  mnist.h)

add_custom_command(
  OUTPUT
    get_mnist.sh
  COMMAND
    ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/get_mnist.sh get_mnist.sh
  DEPENDS
    get_mnist.sh
)

add_custom_target(example_scripts DEPENDS get_mnist.sh)

add_dependencies(conv_mnist example_scripts)
