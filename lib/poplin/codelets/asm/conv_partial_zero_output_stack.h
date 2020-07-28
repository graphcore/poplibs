// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
// Defines vertex stack states offsets shared across multuple
// codelets
//
#ifdef __IPU__
#ifndef __CONV_PARTIAL_ZERO_OUTPUT_STACK_DEF_S__
#define __CONV_PARTIAL_ZERO_OUTPUT_STACK_DEF_S__

// Shared stack between supervisor and workers
#define WKR_ZERO_INFO 0   // word
#define WKR_OUTCHAN_PTR 4 // word
#define WKR_ZERO_OUTPUT_STACK (WKR_OUTCHAN_PTR + 4)

#endif // __CONV_PARTIAL_ZERO_OUTPUT_STACK_DEF_S__
#endif // __IPU__
