// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef popops_RearrangeInternal_hpp
#define popops_RearrangeInternal_hpp

#include <vector>

namespace popops {
namespace internal {

// Can a a number of transpositions be split amongst workers?
bool canSplitTranspose(unsigned numTranspositions, unsigned numWorkers);

// Split a number of transpositions amongst workers such that each worker gets
// at most a single slice of a transposition.
std::vector<unsigned> createSplitTranspose1DWorkList(unsigned rows,
                                                     unsigned cols,
                                                     unsigned numTranspositions,
                                                     unsigned numWorkers,
                                                     unsigned blockSize);

} // namespace internal
} // namespace popops
#endif
