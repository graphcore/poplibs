#ifndef __DeviceInfo_hpp__
#define __DeviceInfo_hpp__
#include <poplar/IPUModelEngine.hpp>
#include <cassert>

class DeviceInfo {
  poplar::IPUModelEngineBuilder &ipuEB;
public:
  unsigned dataPathWidth = 64;
  unsigned convUnitPipelineDepth = 4;
  unsigned fp16AccumConvUnitsPerTile = 8;
  unsigned fp32AccumConvUnitsPerTile = 4;
  bool sharedConvWeights = true;
  
  DeviceInfo(poplar::IPUModelEngineBuilder &ipuEB,
             unsigned dataPathWidth,
             unsigned convUnitPipelineDepth,
             unsigned fp16AccumConvUnitsPerTile,
             unsigned fp32AccumConvUnitsPerTile) :
    ipuEB(ipuEB), dataPathWidth(dataPathWidth),
    convUnitPipelineDepth(convUnitPipelineDepth),
    fp16AccumConvUnitsPerTile(fp16AccumConvUnitsPerTile),
    fp32AccumConvUnitsPerTile(fp32AccumConvUnitsPerTile) {}

  unsigned getTilesPerIPU() const { return ipuEB.getTilesPerIPU(); }
  unsigned getNumIPUs() const { return ipuEB.getNumIPUs(); }
  unsigned getNumTiles() const { return getTilesPerIPU() * getNumIPUs(); }
  unsigned getNumWorkerContexts() const {
    return ipuEB.getNumWorkerContexts();
  }
  unsigned getIPUExchangeBandwidth() const {
    return ipuEB.getIPUExchangeBandwidth();
  }
  unsigned getFloatVectorWidth() const {
    assert(dataPathWidth % 32 == 0);
    return dataPathWidth / 32;
  }

  unsigned getHalfVectorWidth() const {
    assert(dataPathWidth % 16 == 0);
    return dataPathWidth / 16;
  }

  unsigned getInputChannelsPerConvUnit() const {
    return getHalfVectorWidth() * convUnitPipelineDepth;
  }

};

#endif // __DeviceInfo_hpp__
