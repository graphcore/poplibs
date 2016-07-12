#ifndef __ConvReuse_hpp__
#define __ConvReuse_hpp__
#include <poplar/Graph.hpp>
#include <poplar/Program.hpp>
#include "neural_net_common.h"

struct ReusableLayer {
  poplar::program::Program prog;
  std::vector<poplar::Tensor> inputs;
  std::vector<poplar::Tensor> outputs;
  ReusableLayer(poplar::program::Program prog,
                std::vector<poplar::Tensor> inputs,
                std::vector<poplar::Tensor> outputs) :
    prog(prog),
    inputs(std::move(inputs)),
    outputs(std::move(outputs)) {}
};

class ConvImplSpec {
public:
  unsigned inNumChans, inNumChanGroups, inDimX, inDimY;
  unsigned outNumChans, outNumChanGroups, outDimX, outDimY;
  unsigned resNumChans, resNumChanGroups, resDimX, resDimY;
  unsigned kernelSize, stride, padding;
  NonLinearityType nonLinearityType;
  ResidualMethod resMethod;
  ConvImplSpec(unsigned inNumChans, unsigned inNumChanGroups,
               unsigned inDimX, unsigned inDimY,
               unsigned outNumChans, unsigned outNumChanGroups,
               unsigned outDimX, unsigned outDimY,
               unsigned resNumChans, unsigned resNumChanGroups,
               unsigned resDimX, unsigned resDimY,
               unsigned kernelSize, unsigned stride, unsigned padding,
               NonLinearityType nonLinearityType,
               ResidualMethod resMethod) :
    inNumChans(inNumChans), inNumChanGroups(inNumChanGroups),
    inDimX(inDimX), inDimY(inDimY),
    outNumChans(outNumChans), outNumChanGroups(outNumChanGroups),
    outDimX(outDimX), outDimY(outDimY),
    resNumChans(resNumChans), resNumChanGroups(resNumChanGroups),
    resDimX(resDimX), resDimY(resDimY),
    kernelSize(kernelSize), stride(stride), padding(padding),
    nonLinearityType(nonLinearityType), resMethod(resMethod) {}

  bool operator<(const ConvImplSpec &other) const;
};


#endif // __ConvReuse_hpp__
