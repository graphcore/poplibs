#ifndef __Loss_hpp__
#define __Loss_hpp__
#include "poplar/Program.hpp"
#include "poplar/Tensor.hpp"
#include "popnn/NetDef.hpp"

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

#endif // __Loss_hpp__
