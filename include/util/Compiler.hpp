#ifndef __popstd_Compiler_hpp__
#define __popstd_Compiler_hpp__

// This file provides useful macros to use in the Poplar libraries

#ifdef NDEBUG
#define POPLIB_UNREACHABLE() __builtin_unreachable()
#else
#define POPLIB_UNREACHABLE() __builtin_trap()
#endif

#endif // _popstd_Compiler_hpp__
