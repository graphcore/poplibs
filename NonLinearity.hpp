#ifndef __NonLinearity_hpp__
#define __NonLinearity_hpp__
#include "neural_net_common.h"
#include "DeviceInfo.hpp"
#include "poplar/IPUModelEngine.hpp"
#include "poplar/Program.hpp"

poplar::program::Program
bwdNonLinearity(poplar::Graph &graph,
                poplar::IPUModelEngineBuilder::TileMapping &mapping,
                DeviceInfo &deviceInfo,
                std::string dType,
                poplar::Tensor z, poplar::Tensor deltasIn,
                poplar::Tensor zDeltas,
                NonLinearityType nonLinearityType);

#endif // __NonLinearity_hpp__
