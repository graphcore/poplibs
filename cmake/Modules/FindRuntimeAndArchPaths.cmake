# Search for runatime and arch_man path
foreach(path ${CMAKE_PREFIX_PATH})
  if(EXISTS "${path}/lib/graphcore/include/stddef.h")
    set(RUNTIME_PATH ${path})
  endif()
  if(EXISTS "${path}/include/colossus/tileimplconsts.h")
    set(ARCH_MAN_PATH ${path})
  endif()
endforeach(path)
