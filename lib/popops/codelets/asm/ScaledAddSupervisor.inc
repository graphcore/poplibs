// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef __SCALED_ADD_SUPERVISOR_INC__
#define __SCALED_ADD_SUPERVISOR_INC__

// Shared register across files for the scale value
#define k a6
// For running the worker functions
#define sv_workerFunction m6

// Flag for memConstraints
#define MEM_CONSTRAINTS_MASK 0x1
// Common register between files to indicate memConstraints were applied
#define memConstraints m11

// Bit to force all accumulators to zero when written in FP_CLR register
#define ZAACC_BITMASK (CSR_W_FP_CLR__ZAACC__MASK << CSR_W_FP_CLR__ZAACC__SHIFT)

#endif
