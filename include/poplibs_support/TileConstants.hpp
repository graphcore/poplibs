// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poplibs_support_TileConstants_hpp
#define poplibs_support_TileConstants_hpp

#ifdef __IPU__
#include "arch/gc_tile_defines.h"
#else
#define CTXT_WORKERS                     (6)
#endif
//TODO is there any reason why NUM_WORKERS
// might be different from CTXT_WORKERS?
#define NUM_WORKERS           (CTXT_WORKERS)

#define CONV_UNIT_INPUT_LOAD_ELEMS_FLOAT (2)
#define CONV_UNIT_INPUT_LOAD_ELEMS_HALF  (4)


// end namespace poplibs

#endif // poplibs_support_TileConstants_hpp
