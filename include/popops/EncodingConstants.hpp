// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Constants used by encoding functions.
 *
 */

#ifndef popops_EncodingConstants_hpp
#define popops_EncodingConstants_hpp

/// Code point for masked index (an index to be ignored).
#define MASKED_LABEL_CODE 0xFFFFFFFFU

/// Small constant used in natural logarithm computation.
/// @{
#define EPS_LOG_N_FLOAT (1.17549435e-38F)
#define EPS_LOG_N_HALF (0.000000059605F)
/// @}

#endif // popops_EncodingConstants_hpp
