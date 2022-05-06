// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef __SCALED_ADD_SUPERVISOR_INC__
#define __SCALED_ADD_SUPERVISOR_INC__

// Shared register across files for the scale value
#define k a6

// Flag for memConstraints
#define MEM_CONSTRAINTS_MASK 0x1
// Common register between files to indicate memConstraints were applied
#define memConstraints m11

#endif
