#include "ConvReducePlan.hpp"
#include <tuple>
#include <cmath>

namespace poplin {


/// Return the number of reduce stages to use for a reduction of the specified
/// reduction depth.
static unsigned getNumReduceStages(unsigned partialsDepth) {
  /// Using more reduce stages affects code size as follows.
  /// If the reduction depth is p then a single stage reduction requires each
  /// tile to receive p messages. If instead we break the reduction down into n
  /// stages then each stage involves a reduction of reduce p^(1/n) messages.
  /// The total number of messages is n*p^(1/n). For large p, increase n
  /// will reducing the total number of messages received which is turn likely
  /// to also reduce the exchange code size. The thresholds below have been
  /// chosen based on benchmarking.
  if (partialsDepth >= 125)
    return 3;
  if (partialsDepth >= 16)
    return 2;
  return 1;
}

/// Return a plan for how to split a reduction into multiple stages along with
/// an estimate of the cost of the plan. The first member of the pair is a
/// vector of the depth of each partials tensor in each intermediate stage.
/// If the vector is empty there are no intermediate stages and the reduction
/// is performed in a single step. The second member of the pair is an
/// estimated cost. The cost is an estimate of the average number of messages
/// required per tile.
std::pair<std::vector<unsigned>, float>
getMultiStageReducePlanAndCost(unsigned partialsDepth, unsigned numStages) {
  if (numStages == 1) {
    return {{}, partialsDepth};
  }
  auto nextDepthRoundDown =
      static_cast<unsigned>(
        std::pow(static_cast<double>(partialsDepth),
                 (numStages - 1.0) / numStages)
      );
  std::vector<unsigned> roundDownPlan, roundUpPlan;
  float roundDownCost, roundUpCost;
  std::tie(roundDownPlan, roundDownCost) =
      getMultiStageReducePlanAndCost(nextDepthRoundDown, numStages - 1);
  roundDownCost += static_cast<float>(partialsDepth) / nextDepthRoundDown;
  auto nextDepthRoundUp = nextDepthRoundDown + 1;
  std::tie(roundUpPlan, roundUpCost) =
      getMultiStageReducePlanAndCost(nextDepthRoundUp, numStages - 1);
  roundUpCost += static_cast<float>(partialsDepth) / nextDepthRoundUp;
  if (roundDownCost < roundUpCost) {
    roundDownPlan.insert(roundDownPlan.begin(), nextDepthRoundDown);
    return {roundDownPlan, roundDownCost};
  }
  roundUpPlan.insert(roundUpPlan.begin(), nextDepthRoundUp);
  return {roundUpPlan, roundUpCost};
}

std::vector<unsigned>
getMultiStageReducePlan(unsigned partialsDepth) {
  const auto numStages = getNumReduceStages(partialsDepth);
  return getMultiStageReducePlanAndCost(partialsDepth, numStages).first;
}

}
