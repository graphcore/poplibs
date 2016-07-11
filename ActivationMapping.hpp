#ifndef __ActivationMapping_hpp__
#define __ActivationMapping_hpp__
#include <vector>
#include "DeviceInfo.hpp"
#include "poplar/Tensor.hpp"
#include "poplar/IPUModelEngine.hpp"


std::vector<unsigned> computeActivationsMapping(poplar::Tensor t,
                                                const DeviceInfo &deviceInfo);

void mapActivations(poplar::Tensor t,
                    poplar::IPUModelEngineBuilder::TileMapping &mapping,
                    const DeviceInfo &deviceInfo);


void mapTensor(poplar::Tensor t,
               poplar::IPUModelEngineBuilder::TileMapping &mapping,
               const DeviceInfo &deviceInfo);

#endif // __ActivationMapping_hpp__
