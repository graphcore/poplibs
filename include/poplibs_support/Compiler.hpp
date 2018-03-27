// Copyright (c) 2018, Graphcore Ltd, All rights reserved.

#ifndef poplibs_support_Compiler_hpp
#define poplibs_support_Compiler_hpp

// This file provides useful macros to use in the Poplar libraries

#ifdef NDEBUG
#define POPLIB_UNREACHABLE() __builtin_unreachable()
#else
#define POPLIB_UNREACHABLE() __builtin_trap()
#endif

#endif // poplibs_support_Compiler_hpp
