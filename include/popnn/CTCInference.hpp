// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
/** \file
 *
 * Support for Connectionist Temporal Classification (CTC) Beam search decoder.
 *
 */

#ifndef popnn_CTCInference_hpp
#define popnn_CTCInference_hpp

#include "CTCPlan.hpp"

#include <poplar/Graph.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Program.hpp>

namespace popnn {
namespace ctc {

/** Create a plan for implementing the CTC Beam search inference function
 *
 * **CTC Beam search inference options**
 *
 *    * `partialsType` poplar::Type [=poplar::FLOAT]
 *
 *      The type to use for partial results.
 *
 *    * `availableMemoryProportion` Decimal between 0 and 1 (inclusive) [=0.6]
 *
 *      The maximum proportion of available memory on each tile that this
 *      layer should consume temporarily during the course of the operation.
 *
 * \param graph            The graph the operation will be added to
 * \param inType           The data type of the probability data input
 * \param batchSize        The size of the batch to be processed at once
 * \param maxTime          The maximum time of any sequence input
 * \param numClasses       The number of symbols/classes in the "alphabet",
 *                         including the blankClass
 * \param beamwidth        The number of beams to maintain during beamsearch
 * \param options          Any implementation/debug options for the operation
 *
 * \return plan            The plan produced, which will specify how the
 *                         operation is to be implemented
 */
Plan plan(const poplar::Graph &graph, const poplar::Type &inType,
          unsigned batchSize, unsigned maxTime, unsigned numClasses,
          unsigned beamwidth, const poplar::OptionFlags &options = {});

} // namespace ctc
} // namespace popnn

#endif // popnn_CTCLoss_hpp
