// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef popnn_CTCInferenceDefs_hpp
#define popnn_CTCInferenceDefs_hpp

#ifndef INCLUDE_IN_ASSEMBLER
#include <limits>

namespace popnn {
namespace ctc_infer {
// A reserved class index representing "nothing" in beam search decoding
inline constexpr auto voidSymbol = std::numeric_limits<unsigned>::max();
// A reserved class index representing an invalid candidate or beam addend
// in beam search decoding
inline constexpr auto invalidSymbol = std::numeric_limits<unsigned>::max() - 1;
} // namespace ctc_infer
} // namespace popnn

#endif

// Equivalent definition for assembler inclusion
#define VOID_SYMBOL 0xffffffff
#define INVALID_SYMBOL 0xfffffffe

#endif // popnn_CTCInferenceDefs_hpp
