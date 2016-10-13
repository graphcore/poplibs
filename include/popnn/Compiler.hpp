#ifndef __Compiler_hpp__
#define __Compiler_hpp__

#ifdef NDEBUG
#define POPNN_UNREACHABLE() __builtin_unreachable()
#else
#define POPNN_UNREACHABLE() __builtin_trap()
#endif

#endif // _Compiler_hpp__
