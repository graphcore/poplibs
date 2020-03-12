// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include "poplin/FullyConnected.hpp"

#include "poplibs_support/Compiler.hpp"

namespace poplin {
namespace fc {

// For a FullyConnected layer, we can express it with the following parameters:
//     {groupSize, batchSize, inputSize, outputSize}
//
// These are independent from the pass (fwd, bwd, wu), but can be mapped to
// a.shape() and b.shape().
//
// As an example, for a fwd pass matrix multiplication:
//     A x B = OUT
// We can represent it with these parameters:
//     A(rows) => numGroups, A(cols) => inputSize
//     B(rows) => inputSize, B(cols) => outputSize
//
// Shown is one of the groups:
//
//                                          outputSize
//               inputSize                    _ _ _                   outputSize
//            _ _ _ _ _ _ _ _               |       |                   _ _ _
//           |               |              |       |                 |       |
// batchSize |               |      X       |       |     =           |       |
//           |       A       |              |   B   |                 |  OUT  |
//           |               |              |       |       batchSize |       |
//           |_ _ _ _ _ _ _ _|    inputSize |       |                 | _ _ _ |
//                                          |       |
//                                          | _ _ _ |
//
// We can also map between forward/backward/weight update passes as follows:
//     - bwd pass we swap (fwd parameters) input & output size
//     - wu pass we swap (fwd parameters) input & batch size

enum class Pass { FWD, BWD, WU };

static MatMulParams toMatMulParamsFwdPass(FullyConnectedParams p,
                                          poplar::Type type) {
  return {type,
          type,
          {p.numGroups, p.batchSize, p.inputSize},
          {p.numGroups, p.inputSize, p.outputSize}};
}

static MatMulParams toMatMulParams(FullyConnectedParams p, poplar::Type type,
                                   Pass pass) {
  switch (pass) {
  case Pass::FWD:
    return toMatMulParamsFwdPass(p, type);
  case Pass::BWD: // Swap the input and output size
    return toMatMulParamsFwdPass(
        {p.numGroups, p.batchSize, p.outputSize, p.inputSize}, type);
  case Pass::WU: // Swap the input and batch size
    return toMatMulParamsFwdPass(
        {p.numGroups, p.inputSize, p.batchSize, p.outputSize}, type);
  default:
    POPLIB_UNREACHABLE();
  }
}

// Generate set of matmuls for each pass required.
// Override fullyConnectedPass of the matmul options, keeping the rest of the
// options
std::vector<std::pair<MatMulParams, poplar::OptionFlags>>
getMatMulPrePlanParameters(FullyConnectedParams params,
                           poplar::OptionFlags matmulOptions, poplar::Type type,
                           bool inferenceOnly) {
  std::vector<std::pair<MatMulParams, poplar::OptionFlags>> matmuls;

  // Fwd pass
  const auto fwdPassParams = toMatMulParams(params, type, Pass::FWD);
  poplar::OptionFlags fwdPassOpt = matmulOptions;
  fwdPassOpt.set("fullyConnectedPass",
                 inferenceOnly ? "INFERENCE_FWD" : "TRAINING_FWD");
  matmuls.push_back(std::make_pair(fwdPassParams, fwdPassOpt));

  // Bwd and Wu passes
  if (!inferenceOnly) {
    const auto bwdPassParams = toMatMulParams(params, type, Pass::BWD);
    poplar::OptionFlags bwdPassOpt = matmulOptions;
    bwdPassOpt.set("fullyConnectedPass", "TRAINING_BWD");

    const auto wuPassParams = toMatMulParams(params, type, Pass::WU);
    poplar::OptionFlags wuPassOpt = matmulOptions;
    wuPassOpt.set("fullyConnectedPass", "TRAINING_WU");

    matmuls.push_back(std::make_pair(bwdPassParams, bwdPassOpt));
    matmuls.push_back(std::make_pair(wuPassParams, wuPassOpt));
  }
  return matmuls;
}
} // namespace fc
} // namespace poplin
