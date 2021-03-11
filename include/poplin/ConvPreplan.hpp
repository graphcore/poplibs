// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Functions and data types to support performing convolution preplanning
 *
 */

#ifndef poplin_ConvPreplan_hpp
#define poplin_ConvPreplan_hpp
#include "Convolution.hpp"
#include "MatMul.hpp"

#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <set>
#include <tuple>

namespace poplin {

/**
 * Plan the specified convolutions & matmuls.
 *
 * \param convs   A set of tuples of:
 *                  - conv-specific target for tile / IPU sizing
 *                  - convolution parameters
 *                  - implementation options. See createWeights().
 *
 *                All entries must have matching machine parameters.
 * \param matmuls A set of tuples of:
 *                  - matmul-specific target for tile / IPU sizing
 *                  - convolution parameters
 *                  - implementation options. See createWeights().
 *
 *                All entries must have matching machine parameters.
 * \param cache   The planning cache to update.
 */
void preplan(const std::set<ConvPlanParams> &convs,
             const std::set<MatMulPlanParams> &matmuls, PlanningCache &cache);

} // namespace poplin

#endif // poplin_ConvPreplan_hpp
