#ifndef __popnn_Loss_hpp__
#define __popnn_Loss_hpp__

namespace popnn {

enum LossType {
  SUM_SQUARED_LOSS,
  SOFTMAX_CROSS_ENTROPY_LOSS
};

} // end namespace popnn

#ifndef __POPC__
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>

namespace popnn {

poplar::program::Program
calcLoss(poplar::Graph &graph,
         const poplar::Tensor& activations,
         const poplar::Tensor& expected,
         const poplar::Tensor& loss,
         const poplar::Tensor& deltas,
         const poplar::Tensor& numCorrect,
         const poplar::Type& activationType,
         const poplar::Type& expectedType,
         LossType lossType,
         const std::string &debugPrefix = "");

} // end namespace popnn

#endif // !__POPC__


#endif // __Loss_hpp__
