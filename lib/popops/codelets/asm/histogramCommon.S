// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

// Shared macros, constants used by Histogram vertices

// Macros to implement the abs value of $a0 or $a0:1 if required by the vertex
.macro CONDITIONAL_ABSv2HALF IS_ABS
.ifc "\IS_ABS", "true"
  f16v2absadd $a0, $azero, $a0
.endif
.endm

.macro CONDITIONAL_ABSv4HALF_BUNDLE IS_ABS
.ifc "\IS_ABS", "true"
{ nop
  f16v4absadd $a0:1, $azeros, $a0:1
}
.endif
.endm

.macro CONDITIONAL_ABSv4HALF IS_ABS
.ifc "\IS_ABS", "true"
  f16v4absadd $a0:1, $azeros, $a0:1
.endif
.endm

.macro CONDITIONAL_ABSv1FLOAT IS_ABS
.ifc "\IS_ABS", "true"
  f32absadd $a0, $azero, $a0
.endif
.endm

.macro CONDITIONAL_ABSv2FLOAT_BUNDLE IS_ABS
.ifc "\IS_ABS", "true"
{ nop
  f32v2absadd $a0:1, $azeros, $a0:1
}
.endif
.endm

.macro CONDITIONAL_ABSv2FLOAT IS_ABS
.ifc "\IS_ABS", "true"
  f32v2absadd $a0:1, $azeros, $a0:1
.endif
.endm
