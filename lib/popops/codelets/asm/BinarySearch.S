// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifdef __IPU__

#include "poplar/TileConstants.hpp"
#include "poplar/StackSizeDefs.hpp"
#include "BinarySearch.h.S"
#include "CommonPoplibsMacros.h.S"

// These functions are used by multiUpdate and multiUpdateOp. The registers
// are carefully aliased so that there is minimal register spill. 

// register aliases
#define low           m3
#define high          m4
#define mid           m6
#define temp          m9
#define val           m3
#define midValue      m8

FN_WORKER_ENTRY_POINT lowerBinarySearch

setzi $low, 0                            // low = 0
sub $high, $mBS_numIndices, 1            // high = numIndices - 1 
sub $mBS_startIndex, $mzero, 1           // startIndex = -1 

LowerSearchBegin:
sub $temp, $high, $low                   // temp = high - low
brneg $temp, LowerSearchEnd              // if !(low <= high)

shr $temp, $temp, 1
add $mid, $temp, $low                    // mid = (high - low)/2 + low

ld32 $midValue, $mzero, $mBS_indicesPtr, $mid // indicesPtr[mid]
sub $temp, $midValue, $mBS_targetValue
brneg $temp, LowerUpdateLow              // indicesPtr[mid] < targetValue
sub $high, $mid, 1
mov $mBS_startIndex, $mid
bri LowerSearchBegin                     // continue with binary search

LowerUpdateLow:
add $low, $mid, 1
// continue to while loop
bri LowerSearchBegin

LowerSearchEnd:
brpos $mBS_startIndex, LowerCheckDuplicateEntries
mov $mBS_startIndex, $mBS_numIndices
bri LowerEnd

// Check for duplicate entries
LowerCheckDuplicateEntries:
// we are guaranteed that startIndex will move through 0, hence we can check
// for equality with 0.
brz $mBS_startIndex, LowerEnd
add $temp, $mBS_startIndex, -1               // startIndex - 1
ld32 $val, $mzero, $mBS_indicesPtr, $temp    // indicesPtr[startIndex - 1]
sub $val, $val, $mBS_targetValue
brneg $val, LowerEnd                         // indicesPtr[startIndex - 1] < tatgetVal
mov $mBS_startIndex, $temp                   // --startIndex
bri LowerCheckDuplicateEntries

LowerEnd:
// return value is in startIndex
br $mBS_retLr
FN_SIZE lowerBinarySearch


//------------------------------------------------------------------------------

FN_WORKER_ENTRY_POINT upperBinarySearch

setzi $low, 0                        // low = 0
sub $high, $mBS_numIndices, 1            // high = numIndices - 1 
sub $mBS_endIndex, $mzero, 1           // endIndex = -1 

UpperSearchBegin:
sub $temp, $high, $low               // temp = high - low
brneg $temp, UpperSearchEnd          // if !(low <= high)

shr $temp, $temp, 1
add $mid, $temp, $low                // mid = (high - low)/2 + low

ld32 $midValue, $mzero, $mBS_indicesPtr, $mid // indicesPtr[mid]
sub $temp, $midValue, $mBS_targetValue
brneg $temp, UpperUpdateLow            // indicesPtr[mid] < targetValue
sub $high, $mid, 1
bri UpperSearchBegin                 // continue with binary search

UpperUpdateLow:
add $low, $mid, 1
mov $mBS_endIndex, $mid

// continue to while loop
bri UpperSearchBegin

UpperSearchEnd:
brpos $mBS_endIndex, UpperCheckDuplicateEntries
mov $mBS_endIndex, $mzero
br $mBS_retLr

// Check for duplicate entries
UpperCheckDuplicateEntries:
add $temp, $mBS_endIndex, 1               // endIndex + 1
sub $val, $temp, $mBS_numIndices
brpos $val, UpperEnd

ld32 $val, $mzero, $mBS_indicesPtr, $temp
sub $val, $mBS_targetValue, $val
brneg $val, UpperEnd
mov $mBS_endIndex, $temp
bri UpperCheckDuplicateEntries

UpperEnd:
// return value is in endIndex
add $mBS_endIndex, $mBS_endIndex, 1
br $mBS_retLr

FN_SIZE upperBinarySearch
#endif // __IPU__
