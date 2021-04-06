// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef popnn_CTCInferenceDefs_hpp
#define popnn_CTCInferenceDefs_hpp

#include <limits>

namespace popnn {
namespace ctc_infer {
// A reserved class index representing "nothing" in beam search decoding
inline constexpr auto voidSymbol = std::numeric_limits<unsigned>::max();
} // namespace ctc_infer
} // namespace popnn
#endif // popnn_CTCInferenceDefs_hpp
