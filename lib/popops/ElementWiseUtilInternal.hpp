// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#ifndef _popops_ElementWiseUtilInternal_hpp_
#define _popops_ElementWiseUtilInternal_hpp_

#include "ExprOpUtil.hpp"
#include <poplar/Graph.hpp>
#include <vector>

namespace popops {

/** Return the section of a contiguous region starting from a specific offset
 *  and with a specific length. At the end of a call to this function, the
 *  parameters "index" and "offset" are updated to point immediately following
 *  the extracted section.
 *
 *   Note 1: secLength must be greater than 0
 *   Note 2: User inputs should be such that the section does not exceed the
 *           bounds of the region, as the function does not verify this.
 *
 *   \param region     The region that needs to be sliced into sections
 *   \param secLength  Length of the section that needs to be returned
 *   \param index      Index into the vector of intervals from the start of
 *                     region to the start of the section. The index is updated
 *                     by the end of this function
 *   \param offset     Offset from the beginning of the indexed interval
 *                     "region[index]" to the beginning of the section. The
 *                     offset is updated by the end of this function.
 *   \param regIndex   Not used by the program directly, but appropriately
 *                     incremented if all the intervals in the present region
 *                     have been used up..
 *
 *   \return A section of the region starting at the position specified by
 *           "index" and "offset".
 *
 */
std::vector<poplar::Interval>
cutRegionSection(const std::vector<poplar::Interval> &region,
                 const unsigned secLength, unsigned &index, unsigned &offset,
                 unsigned &regIndex);

/** Generate vertices to perform an element-wise operation where
 *  the second operand is just one underlying unique element.
 *
 *  This assumes each element of the outer vector in `intervals`
 *  contains regions which are both contiguous in memory and
 *  cover a single unique underlying element in in2.
 *
 *  \param graph             The graph to add vertices to.
 *  \param in1               LHS input operand.
 *  \param in2               RHS input operand, the input that is broadcast.
 *  \param out               Output operand. If in-place this will be the same
 *                           as the LHS input operand `in1`.
 *  \param intervals         Contiguous regions for the output operand on this
 *                           tile.
 *  \param tile              The tile to add vertices to.
 *  \param cs                The compute set to add vertices to.
 *  \param op                Binary operation to perform.
 *  \param inPlace           Whether or not this operation is performed in-place
 *                           on the LHS input operand.
 *  \param uniformScalar     Whether or not the scalar for each contiguous
 *                           region in `intervals` is the same. If true this
 *                           allows use of smaller vertices in the 2-dimensional
 *                           case.
 *  \param exitIfInefficient Fail in finding an appropriate vertex if neither
 *                           supervisor nor worker vertices are efficient.
 *
 *  \return true if vertex was found and false otherwise.
 */
bool createVertexBinaryOpBroadcastScalar(
    poplar::Graph &graph, const poplar::Tensor &in1, const poplar::Tensor &in2,
    const poplar::Tensor &out,
    const std::vector<std::vector<poplar::Interval>> &intervals, unsigned tile,
    const poplar::ComputeSet &cs, expr::BinaryOpType op, bool inPlace = false,
    bool uniformScalar = false, bool exitIfInefficient = false);

} // end namespace popops

#endif // _popops_ElementWiseUtilInternal_hpp_
