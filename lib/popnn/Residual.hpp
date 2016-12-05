#ifndef __Residual_hpp__
#define __Residual_hpp__
#include "popnn/ResidualDef.hpp"
#include "poplar/Program.hpp"

namespace residual {
/// Add the input activations together.
/// \a in0, \a in1 and \a out must have identical dimensions
poplar::program::Program
joinResidual(poplar::Graph &graph,
             poplar::Tensor in1,
             poplar::Tensor in2,
             poplar::Tensor out,
             const std::string &debugPrefix = "");

/// Add the input deltas together.
/// Only the channel elements present in in1 are updated in
/// \outIn0. if \a in1 is smaller in x and y only the
/// appropriate elements in \a outIn0 are updated.
poplar::program::Program
joinDeltas(poplar::Graph &graph,
           poplar::Tensor outIn0,
           poplar::Tensor in1,
           const std::string &debugPrefix);
std::uint64_t getNumberOfAdds(unsigned outDimY, unsigned outDimX,
                              unsigned outNumChans, bool forwardOnly);

uint64_t getFlops(unsigned batchSize,
                  unsigned inDimY, unsigned inDimX, unsigned inNumChans,
                  bool forwardOnly);

/// Convert a Tensor to the required dimensions
poplar::Tensor
arrangeResidualInput(poplar::Graph &graph,
                     poplar::Tensor resIn,
                     std::vector<size_t> outDims,
                     std::string dType,
                     ResidualMethod resMethod);

double getPerfectCycleCount(const poplar::Graph &graph,
                            std::string dType, unsigned batchSize,
                            unsigned inDimY, unsigned inDimX,
                            unsigned numChannels,
                            bool forwardOnly);

}

#endif // __Residual_hpp__
