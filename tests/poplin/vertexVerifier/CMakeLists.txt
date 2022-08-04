include(GNUInstallDirs)

set(GP_OBJECTS "${CMAKE_INSTALL_FULL_LIBDIR}")
get_target_property(VV_EXECUTABLE vertex_verifier LOCATION)

add_test(NAME VVConvPartialQuarterHalf1x1Out
  COMMAND ${VV_EXECUTABLE}
    --input-file=${GP_OBJECTS}/poplin.gp
    --vertex=poplin::ConvPartial1x1Out<quarter,half,true,false,16,8,false>
    --target=ipu21
    --target=ipu21:cpp
    --add-overread-zone=each
    --constraints-file=${CMAKE_CURRENT_SOURCE_DIR}/ConvPartialQuarterHalf1x1Out.vv
)

add_test(NAME VVConvPartialQuarterHalfNx1
  COMMAND ${VV_EXECUTABLE}
    --input-file=${GP_OBJECTS}/poplin.gp
    --vertex=poplin::ConvPartialnx1<quarter,half,true,false,16,8,false>
    --target=ipu21
    --target=ipu21:cpp
    --add-overread-zone=each
    --constraints-file=${CMAKE_CURRENT_SOURCE_DIR}/ConvPartialQuarterHalfnx1.vv
)

add_test(NAME VVConvPartialQuarterHalf1xNStride1
  COMMAND ${VV_EXECUTABLE}
    --input-file=${GP_OBJECTS}/poplin.gp
    --vertex=poplin::ConvPartial1xNSLIC<quarter,half,1,true,4,4,8,false>
    --target=ipu21
    --target=ipu21:cpp
    --add-overread-zone=each
    --constraints-file=${CMAKE_CURRENT_SOURCE_DIR}/ConvPartialQuarterHalf1xnStride1.vv
)

add_test(NAME VVConvPartialQuarterHalf1xNStride2
  COMMAND ${VV_EXECUTABLE}
    --input-file=${GP_OBJECTS}/poplin.gp
    --vertex=poplin::ConvPartial1xNSLIC<quarter,half,2,true,4,4,8,false>
    --target=ipu21
    --target=ipu21:cpp
    --add-overread-zone=each
    --constraints-file=${CMAKE_CURRENT_SOURCE_DIR}/ConvPartialQuarterHalf1xnStride2.vv
)
