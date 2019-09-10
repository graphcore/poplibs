#ifndef ConvReduce_hpp
#define ConvReduce_hpp

#include <string>
#include <vector>

#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include <poplar/Tensor.hpp>
#include <poplar/Type.hpp>

namespace poplin {

poplar::Tensor
multiStageGroupedReduce(poplar::Graph &graph,
                        poplar::Tensor partials,
                        const poplar::Type &resultType,
                        std::vector<poplar::program::Sequence> &
                          preComputeProgs,
                        std::vector<poplar::ComputeSet> &computeSets,
                        const std::string &debugPrefix);

}

#endif // ConvReduce_hpp
