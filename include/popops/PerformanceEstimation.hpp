// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef _popops_performance_estimation_h_
#define _popops_performance_estimation_h_

#include "popops/Expr.hpp"
#include <poplar/Target.hpp>

namespace popops {

// All functions within the internal namespace are liable to be changed at any
// point without prior notice.
namespace internal {

/** Supervisor context cycle estimate for BinaryOp and UnaryOp supervisor
 * codelets.
 *
 * \param isScaledPtr64Type  Is true if vector vertex field is ScaledPtr64
 *                           and false otherwise.
 *
 * \returns Estimated number of cycles.
 */
std::uint64_t basicOpSupervisorOverhead(const bool isScaledPtr64Type = false);

/** Cycle cost for processing an arbitrary number of elements given the cycle
 * cost for processing a vector of elements simultaneously.
 *
 * \param numElems        Number of elements in the operand.
 * \param vectorSize      Size of vector of elements that can be processed
 *                        simultaneously.
 * \param cyclesPerVector Cycle cost per vector of elements.
 *
 * \returns Estimated number of cycles.
 */
std::uint64_t basicOpLoopCycles(const unsigned numElems,
                                const unsigned vectorSize,
                                const unsigned cyclesPerVector);

/** Cycle cost for processing an arbitrary number of elements given the cycle
 * cost for processing a vector of elements simultaneously.
 *
 * \param target          The target on which the operation should be estimated
 * \param type            Data type of elements.
 * \param cyclesPerVector Cycle cost per vector of elements.
 * \param vectorSize      Size of vector of elements that be processed
 *                        simultaneously.
 * \param numElems        Number of elements in the operand.
 * \param overheadPerLoop Cycles overhead per loop.
 *
 * \returns Estimated number of cycles.
 */
std::uint64_t binaryOpInnerLoopCycles(const poplar::Target &target,
                                      const poplar::Type &type,
                                      const unsigned cyclesPerVector,
                                      const bool vectorize,
                                      const unsigned numElems,
                                      const std::uint64_t overheadPerLoop);

/** Cycle estimate for Dynamic Slice 1D vertex.
 *
 * \param target         The target on which the operation should be estimated.
 * \param type           Data type of elements.
 * \param regionSize     Number of elements in tensor to be sliced.
 * \param numSubElements Number of elements per slice.
 *
 * \returns Estimated number of cycles.
 */
std::uint64_t getDynamicSlice1DEstimate(const poplar::Target &target,
                                        const poplar::Type &type,
                                        const unsigned regionSize,
                                        const unsigned numSubElements);

/** Cycle estimate for Binary-1D In-Place MultiVertex.
 *
 * \param target   The target on which the operation should be estimated.
 * \param type     Data type of elements.
 * \param op       Binary operator type.
 * \param numElems Total number of output elements.
 *
 * \returns Estimated number of cycles.
 */
std::uint64_t getBinaryOp1DInPlaceSupervisorEstimate(
    const poplar::Target &target, const poplar::Type &type,
    const popops::expr::BinaryOpType op, const unsigned numElems);

} // namespace internal
} // namespace popops

#endif // _popops_performance_estimation_h_
