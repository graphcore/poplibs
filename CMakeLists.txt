cmake_minimum_required(VERSION 3.6.0)
project(poplibs)

# Add path for custom modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules" )
include(FindRuntimeAndArchPaths)

enable_testing()

if(${BUILD_SHARED_LIBS})
  set(Boost_USE_STATIC_LIBS OFF)
else()
  set(Boost_USE_STATIC_LIBS ON)
endif()

find_package(poplar REQUIRED)
find_package(Boost REQUIRED regex)
find_package(TBB REQUIRED)

# For Ubuntu 16 we need to help the linker find TBB. On Ubuntu 18 and
# OSX the linker manages to find it (presumably using the rpath that is
# embedded in libpoplar?)
get_filename_component(TBB_LIB_DIR ${TBB_LIBRARY} DIRECTORY)
set(CMAKE_BUILD_RPATH "${TBB_LIB_DIR}")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Require C++11.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Add some warning flags. check_cxx_compiler_flag isn't very good at detecting
# unsupported flags that don't cause an error, so it is simpler to switch based
# on the compiler ID.

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  string(APPEND CMAKE_CXX_FLAGS " -Wall -Werror=vla -Wno-maybe-uninitialized")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  string(APPEND CMAKE_CXX_FLAGS " -Wall -Werror=vla")
endif()

string(APPEND CMAKE_CXX_FLAGS " -I ${POPLAR_INCLUDE_PATH}")

# Add an explicit exported symbols file (a linker flag)

if(APPLE)
  string(APPEND CMAKE_SHARED_LINKER_FLAGS
         " -Wl,-exported_symbols_list,"
         "${CMAKE_CURRENT_SOURCE_DIR}/exported_symbols_osx.lds")
else()
  string(APPEND CMAKE_SHARED_LINKER_FLAGS
         " -Wl,--version-script,"
         "${CMAKE_CURRENT_SOURCE_DIR}/exported_symbols_linux.lds")
endif()

# Set up popc flags.
set(POPC_FLAGS -I ${CMAKE_SOURCE_DIR}/include -I ${ARCH_MAN_PATH}/include -I ${POPLAR_INCLUDE_PATH})
if(CMAKE_BUILD_TYPE STREQUAL Debug)
  list(APPEND POPC_FLAGS -g)
elseif(CMAKE_BUILD_TYPE STREQUAL RelWithDebInfo)
  list(APPEND POPC_FLAGS -O3 -g)
else()
  list(APPEND POPC_FLAGS -O3)
endif()

# If requested, optimise the debug builds too.
if (OPTIMISE_IPU_DEBUG AND CMAKE_BUILD_TYPE STREQUAL Debug)
  list(APPEND POPC_FLAGS -O3)
endif()

# For poplibs, we don't want surprises in code-generation.
# -Wdouble-promotion will catch any kind of implicit floating point promotion.
list(APPEND POPC_FLAGS -Wdouble-promotion -Werror=double-promotion)

add_definitions("-DBOOST_ICL_USE_STATIC_BOUNDED_INTERVALS")

add_subdirectory(lib)
add_subdirectory(tests)
add_subdirectory(tools)

install(FILES)
