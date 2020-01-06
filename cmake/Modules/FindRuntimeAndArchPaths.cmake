# Search for runtime and arch_man path

# this module defines the following variables:
#   - RUNTIME_PATH
#   - POPLAR_INCLUDE_PATH
#   - ARCH_MAN_PATH
#   - ENABLED_IPU_ARCH_NAMES e.g. "ipu0;ipu1"
#   - ENABLED_IPU_ARCH_NAMES_COMMA_SEPARATED e.g. "ipu0,ipu1"

foreach(path ${CMAKE_PREFIX_PATH})
  if(EXISTS "${path}/lib/graphcore/include/stddef.h")
    set(RUNTIME_PATH ${path})
  endif()

  file(GLOB ARCH_MAN_IPU_PATHS ${path}/include/arch/gc_tilearch_ipu*.h)
  if(ARCH_MAN_IPU_PATHS)
    set(ARCH_MAN_PATH ${path}/lib/graphcore)
    foreach(IPU_PATH ${ARCH_MAN_IPU_PATHS})
      string(REGEX MATCH ".*(ipu[0-9]+).h$" _ ${IPU_PATH})
      set(GC_TILEARCH_IPU ${CMAKE_MATCH_1})
      list(APPEND ENABLED_IPU_ARCH_NAMES ${GC_TILEARCH_IPU})
    endforeach()
    list(REMOVE_ITEM ENABLED_IPU_ARCH_NAMES "ipu0") # We have stopped supporting ipu0
    string(REPLACE ";" "," ENABLED_IPU_ARCH_NAMES_COMMA_SEPARATED "${ENABLED_IPU_ARCH_NAMES}")
  endif()

  if(EXISTS "${path}/lib/graphcore/include/arch/gc_tile_defines.h")
    set(POPLAR_INCLUDE_PATH ${path}/lib/graphcore/include)
  endif()
endforeach(path)

if(NOT RUNTIME_PATH)
  message(WARNING "Could not find runtime path")
endif()

if(NOT ARCH_MAN_PATH)
  message(WARNING "Could not find arch man path")
endif()

if(NOT POPLAR_INCLUDE_PATH)
  message(WARNING "Could not find poplar include path")
endif()
