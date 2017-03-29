include(GNUInstallDirs)
find_package(Boost REQUIRED program_options)

add_library(enigma SHARED
  Optimizer.cpp
  TensorOp.hpp
  ${CMAKE_SOURCE_DIR}/include/enigma/exceptions.hpp
  ${CMAKE_SOURCE_DIR}/include/enigma/Optimizer.hpp
)

target_link_libraries(enigma poplar popnn popconv poplin popstd
                      Boost::program_options)

target_include_directories(enigma
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    .
)

install(TARGETS enigma
        EXPORT enigma
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT enigma
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT enigma DESTINATION ${CMAKE_INSTALL_LIBDIR}
        COMPONENT enigma)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/enigma
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT enigma)
