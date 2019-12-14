// Copyright (c) Graphcore Ltd, All rights reserved.
#include "poplibs_test/Embedding.hpp"

#include <poputil/exceptions.hpp>

namespace poplibs_test {
namespace embedding {

void multiSlice(const boost::multi_array<double, 2> &embeddingMatrix,
                const std::vector<unsigned> &indices,
                boost::multi_array<double, 2> &result) {
  const auto size = embeddingMatrix.shape()[1];
  if (size != result.shape()[1]) {
    throw poputil::poplibs_error("Inner-most dimension of the result does not "
                                 "match the same dim in the embedding matrix");
  }

  if (indices.size() != result.shape()[0]) {
    throw poputil::poplibs_error("Number of indices does not match the number "
                                 "of rows in the output");
  }

  for (unsigned i = 0; i < indices.size(); ++i) {
    if (indices[i] >= embeddingMatrix.size()) {
      throw poputil::poplibs_error("Index is out-of-bounds.");
    }

    const auto &row = embeddingMatrix[indices[i]];
    std::copy_n(std::begin(row), size, result.data() + i * size);
  }
}

void multiUpdateAdd(const boost::multi_array<double, 2> &deltas,
                    const std::vector<unsigned> &indices, const double scale,
                    boost::multi_array<double, 2> &embeddingMatrix) {
  const auto size = deltas.shape()[1];
  if (size != embeddingMatrix.shape()[1]) {
    throw poputil::poplibs_error("Inner-most dimension of the deltas does not "
                                 "match the same dim in the embedding matrix");
  }

  if (indices.size() != deltas.shape()[0]) {
    throw poputil::poplibs_error("Number of indices does not match the number "
                                 "of rows in the deltas");
  }

  for (unsigned i = 0; i < indices.size(); ++i) {
    if (indices[i] >= embeddingMatrix.size()) {
      throw poputil::poplibs_error("Index is out-of-bounds.");
    }

    for (unsigned j = 0; j < size; ++j) {
      embeddingMatrix[indices[i]][j] += deltas[i][j] * scale;
    }
  }
}

} // namespace embedding
} // namespace poplibs_test
