// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Functions and data types to for performing operations on fully-connected
 * layers.
 *
 */

#ifndef poplin_FullyConnected_hpp
#define poplin_FullyConnected_hpp

#include "poplin/MatMul.hpp"

namespace poplin {
namespace fc {

/// Parameters to describe a fully-connected layer.
struct FullyConnectedParams {
  /// The number of groups (where each group represents a fully-connected layer
  /// of the same shape) that operate in one layer. Each group is totally
  /// independent of the others and so \p numGroups is a common dimension among
  /// the inputs, weights, and outputs for the layer.
  std::size_t numGroups;
  /// Number of samples in the input to the layer.
  std::size_t batchSize;
  /// Size of the input in each batch into the layer.
  std::size_t inputSize;
  /// Size of the output in each batch from the layer.
  std::size_t outputSize;
};

/**
 * Predict what matrix multiplications will be needed for the given parameters
 * and return a list of corresponding matmul() parameters and options.
 *
 * \param parameters      Parameters for the fully-connected layer.
 * \param matmulOptions   Option flags are the same as those from matmul().
 *                        They are passed through to the underlying matmul,
 *                        updating the `fullyConnectedPass` option only.
 * \param type            Input and output datatype.
 * \param inferenceOnly   Whether the fully-connected layer is for inference
 *                        only. If true, we can ignore backwards and
 *                        weight-update passes.
 *
 * \returns               Vector of pairs of [\c MatMulParams, \c OptionFlags]
 *                        representing the complete set of matmul parameters
 *                        for planning.
 */
std::vector<std::pair<MatMulParams, poplar::OptionFlags>>
getMatMulPrePlanParameters(FullyConnectedParams parameters,
                           poplar::OptionFlags matmulOptions, poplar::Type type,
                           bool inferenceOnly);

} // namespace fc
} // namespace poplin

#endif // poplin_FullyConnected_hpp
