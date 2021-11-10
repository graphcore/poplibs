// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cassert>
#include <cmath>
#include <poplar/HalfFloat.hpp>
#include <poplar/Vertex.hpp>
#include <type_traits>

#include "elemwiseBinaryOps.hpp"
#include "poplar/TileConstants.hpp"
#include "poplibs_support/ExternalCodelet.hpp"
#include "popops/ExprOp.hpp"

using namespace poplar;

namespace popops {

template <expr::BinaryOpType op, class FPType>
class [[poplar::constraint("elem(*data) != elem(*out)")]] BroadcastVectorInner1D
    : public MultiVertex {
public:
  BroadcastVectorInner1D();

  Input<Vector<FPType, SPAN, 8>> B;
  Input<Vector<FPType, ONE_PTR, 8>> data;
  // The division of work among 6 workers has been done when creating the vertex
  // (contrary to other types of vertices that do that in the vertex code
  // itself).
  //
  // The amount of work to do is expressed by:
  //       totalBlockCount = data.size() / B.size();
  // i.e. how many times the 'B' vector fits inside 'data'
  // This has been divided by 6; the quotient and remainder of this division
  // has been packed into 'dataBlockCountPacked'
  //
  //                         15 14 13 12 11 10            4  3  2  1  0
  //                        +--+--+--+--+--+--+--  .... +--+--+--+--+--+
  // dataBlockCountPacked:  |           13 bits               | 3 bits |
  //                        +--+--+--+--+--+--+--  .... +--+--+--+--+--+
  //
  //                        |                                 |        |
  //                        +---------------+-----------------+----+---+
  //                                        |                      |
  //                          floor(totalBlockCount/6)    totalBlockCount % 6
  //                                (dataBlockCount)       (remainingBlocks)
  //
  const uint16_t dataBlockCountPacked;
  Output<Vector<FPType, ONE_PTR, 8>> out;

  IS_EXTERNAL_CODELET(true);

  bool compute(unsigned wid) {
    unsigned BSize = B.size();
    // Each worker will process a contiguous span from data[offs], of at least
    // 'dataBlockCount' blocks, but the first few (wid=0 to
    // wid=remaining_blocks-1) workers will process 1 extra block.
    unsigned dataBlockCount = (dataBlockCountPacked >> 3);
    const unsigned remainingBlocks = dataBlockCountPacked & 0x07;
    unsigned offs = wid * dataBlockCount +
                    ((wid < remainingBlocks) ? wid : remainingBlocks);
    unsigned numBlocks = dataBlockCount + (wid < remainingBlocks);

    const FPType *dataPtr = &data[offs * BSize];
    FPType *outPtr = &out[offs * BSize];

    for (unsigned k = 0; k != BSize; ++k) {
      for (unsigned j = 0; j != numBlocks; ++j) {
        outPtr[j * BSize + k] =
            BinaryOpFn<op, FPType, architecture::active>::fn(
                dataPtr[j * BSize + k], B[k]);
      }
    }
    return true;
  }
};

// See the comment before the template specializations in
// BroadcastVectorInner2D.cpp, about the old SCALED_ADD operation type.
template class BroadcastVectorInner1D<expr::BinaryOpType::ADD, float>;
template class BroadcastVectorInner1D<expr::BinaryOpType::ADD, half>;
template class BroadcastVectorInner1D<expr::BinaryOpType::DIVIDE, float>;
template class BroadcastVectorInner1D<expr::BinaryOpType::DIVIDE, half>;
template class BroadcastVectorInner1D<expr::BinaryOpType::MULTIPLY, float>;
template class BroadcastVectorInner1D<expr::BinaryOpType::MULTIPLY, half>;
template class BroadcastVectorInner1D<expr::BinaryOpType::SUBTRACT, float>;
template class BroadcastVectorInner1D<expr::BinaryOpType::SUBTRACT, half>;

} // namespace popops
