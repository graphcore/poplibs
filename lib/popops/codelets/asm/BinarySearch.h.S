// Copyright (c) 2021 Graphcore Ltd. All rights reserved.

#ifndef BinarySearch_h_S
#define BinarySearch_h_S
#ifdef __IPU__

// These functions are used by multiUpdate and multiUpdateOp. The registers
// are carefully aliased so that there is minimal register spill. 

#include "poplar/AvailableVTypes.h"
#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"

// Binary search functions
.extern lowerBinarySearch
.extern upperBinarySearch

// binary search register aliases
#define mBS_indicesPtr     m0
#define mBS_numIndices     m1
#define mBS_targetValue    m2
#define mBS_retLr          m11
#define mBS_startIndex     m7
#define mBS_endIndex       m10

#endif // __IPU__
#endif // BinarySearch_h_S
