cmake_minimum_required(VERSION 3.5.0)
project(poplibs)

include(CheckCXXCompilerFlag)

# Add path for custom modules
set(CMAKE_MODULE_PATH
        ${CMAKE_MODULE_PATH}
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules"
        )

enable_testing()

if(${BUILD_SHARED_LIBS})
  set(Boost_USE_STATIC_LIBS OFF)
else()
  set(Boost_USE_STATIC_LIBS ON)
endif()

find_package(poplar REQUIRED)
find_package(Boost REQUIRED regex)
find_package(Sparsehash REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")

check_cxx_compiler_flag(-Wno-maybe-uninitialized COMPILER_SUPPORTS_W_NMI)
if(COMPILER_SUPPORTS_W_NMI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized")
endif()

check_cxx_compiler_flag(-Werror=vla COMPILER_SUPPORTS_W_VLA)
if(COMPILER_SUPPORTS_W_VLA)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror=vla")
endif()

add_subdirectory(lib)
add_subdirectory(tests)
add_subdirectory(tools)

install(FILES)
