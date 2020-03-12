// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#ifndef popops_EncodingConstants_hpp
#define popops_EncodingConstants_hpp

// code point for masked index (i.e. index to be ignored)
#define MASKED_LABEL_CODE 0xFFFFFFFFU

// Small constants used in natural logarithm computation
#define EPS_LOG_N_FLOAT (1.17549435e-38F)
#define EPS_LOG_N_HALF (0.000000059605F)

#endif // popops_EncodingConstants_hpp
