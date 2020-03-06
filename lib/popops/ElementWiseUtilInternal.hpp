// Copyright (c) 2019 Graphcore Ltd, All rights reserved.

#ifndef _popops_ElementWiseUtilInternal_hpp_
#define _popops_ElementWiseUtilInternal_hpp_

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

} // end namespace popops

#endif // _popops_ElementWiseUtilInternal_hpp_
