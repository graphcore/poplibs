// Copyright (c) 2018 Graphcore Ltd. All rights reserved.

#ifndef poplibs_support_TileConstants_hpp
#define poplibs_support_TileConstants_hpp

#ifdef __IPU__
#include "arch/gc_tile_defines.h"
#else
#define CTXT_WORKERS (6)
#endif
// TODO: T12860 Consider merging NUM_WORKERS with CTXT_WORKERS.
#define NUM_WORKERS (CTXT_WORKERS)

#define CONV_UNIT_INPUT_LOAD_ELEMS_FLOAT (2)
#define CONV_UNIT_INPUT_LOAD_ELEMS_HALF (4)

#define TMEM_ELEMSIZE (0x4000)

#define TMEM_BYTE_MAX_ADDRESS_WIDTH (21)

#define CSR_W_FP_CTL__INDEX (258)

#define NUM_STRIDE_BITS (10)
// end namespace poplibs

#endif // poplibs_support_TileConstants_hpp
