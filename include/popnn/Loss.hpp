#ifndef __popnn_Loss_hpp__
#define __popnn_Loss_hpp__
#include "poplar/Program.hpp"
#include "poplar/Tensor.hpp"

namespace popnn {

enum LossType {
  SUM_SQUARED_LOSS,
  SOFTMAX_CROSS_ENTROPY_LOSS
};

#ifndef __POPC__

poplar::program::Program
calcLoss(poplar::Graph &graph,
         const poplar::Tensor& activations,
         const poplar::Tensor& expected,
         const poplar::Tensor& loss,
         const poplar::Tensor& deltas,
         const poplar::Tensor& numCorrect,
         const std::string& activationType,
         const std::string& expectedType,
         LossType lossType,
         const std::string &debugPrefix = "");

#endif // !__POPC__

}

#endif // __Loss_hpp__
